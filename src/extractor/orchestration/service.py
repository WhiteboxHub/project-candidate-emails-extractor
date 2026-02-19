import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..connectors.http_api import get_api_client
from ..connectors.imap_gmail import GmailIMAPConnector
from ..core.settings import get_config
from ..email.cleaner import EmailCleaner
from ..email.reader import EmailReader
from ..extraction.contacts import ContactExtractor
from ..filtering.rules import EmailFilter
from ..orchestration.candidate_runner import CandidateRunner
from ..persistence.db_candidate_source import DatabaseCandidateSource
from ..persistence.job_activity import JobActivityLogUtil
from ..persistence.vendor_contacts import VendorUtil
from ..reporting.email_reporter import EmailReporter
from ..state.uid_tracker import get_uid_tracker
from ..state.cache import DeduplicationCache
from ..workflow.manager import WorkflowManager

logger = logging.getLogger(__name__)

# Project root = 3 levels above this file (src/extractor/orchestration/service.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EmailExtractionService:
    """Main service for candidate email extraction workflow."""

    def __init__(
        self,
        candidate_source: DatabaseCandidateSource,
        workflow_manager: Optional[WorkflowManager] = None,
        run_id: Optional[str] = None,
        workflow_id: Optional[int] = None,
        runtime_parameters: Optional[Dict] = None,
    ):
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)

        self.candidate_source = candidate_source
        self.workflow_manager = workflow_manager
        self.run_id = run_id
        self.workflow_id = workflow_id
        self.runtime_parameters = runtime_parameters or {}

        self.api_client = None
        self.vendor_util = None
        self.job_activity_log_util = None
        self.cleaner = None
        self.extractor = None
        self.email_filter = None
        self.uid_tracker = None
        self.deduplication_cache = None
        self.candidate_runner = None
        self.email_reporter = None

        self._initialize_components()

    def _load_config(self) -> Dict:
        config_loader = get_config()
        try:
            return config_loader.load()
        except FileNotFoundError:
            fallback_path = _PROJECT_ROOT / "configs" / "config.yaml"
            config_loader.config_path = fallback_path
            logger.info("Primary config not found; using fallback config path: %s", fallback_path)
            return config_loader.load()

    def _initialize_components(self):
        try:
            self.api_client = get_api_client()
            self.vendor_util = VendorUtil(self.api_client)
            self.job_activity_log_util = JobActivityLogUtil(self.api_client)
            self.logger.info("API-backed persistence initialized")
        except Exception as error:
            self.logger.warning("API client unavailable, persistence disabled: %s", error)
            self.api_client = None

        self.cleaner = EmailCleaner()
        self.extractor = ContactExtractor(self.config)
        self.email_filter = EmailFilter(self.config)

        # ── UID tracker — always use an absolute path so it resolves correctly
        # regardless of the working directory the script is launched from.
        default_tracker = str(_PROJECT_ROOT / "last_run.json")
        tracker_file = self.runtime_parameters.get("uid_tracker_file", default_tracker)
        self.uid_tracker = get_uid_tracker(tracker_file)
        self.logger.info("UID tracker file: %s", tracker_file)

        self.deduplication_cache = DeduplicationCache()

        self.candidate_runner = CandidateRunner(
            config=self.config,
            cleaner=self.cleaner,
            extractor=self.extractor,
            email_filter=self.email_filter,
            uid_tracker=self.uid_tracker,
            deduplication_cache=self.deduplication_cache,
            vendor_util=self.vendor_util,
            connector_cls=GmailIMAPConnector,
            reader_cls=EmailReader,
        )

        # Initialize SMTP email reporter from environment variables
        smtp_config = {
            "SMTP_SERVER": os.getenv("SMTP_SERVER"),
            "SMTP_PORT": os.getenv("SMTP_PORT", "587"),
            "SMTP_USERNAME": os.getenv("SMTP_USERNAME"),
            "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD"),
            "REPORT_FROM_EMAIL": os.getenv("REPORT_FROM_EMAIL"),
            "REPORT_TO_EMAIL": os.getenv("REPORT_TO_EMAIL"),
        }
        self.email_reporter = EmailReporter(smtp_config)
        self.logger.info("Service components initialized")

    def run(
        self,
        candidate_id: Optional[int] = None,
        candidate_email: Optional[str] = None,
    ) -> Dict:
        execution_metadata = {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "parameters": self.runtime_parameters,
            "candidates": [],
            "started_at": datetime.utcnow().isoformat(),
        }

        total_contacts = 0
        total_positions = 0
        total_failed = 0
        total_emails_fetched = 0

        try:
            # ── Global dedup cache warm-up ─────────────────────────────────────
            if self.vendor_util and self.deduplication_cache:
                self.logger.info("Initializing global deduplication cache...")
                recent_emails = self.vendor_util.get_recent_vendor_emails(limit=5000)
                self.deduplication_cache.add_known_db_emails(recent_emails)

            candidates = self.candidate_source.get_active_candidates(
                candidate_id=candidate_id,
                candidate_email=candidate_email,
            )

            if not candidates:
                self.logger.warning("No candidates found with email credentials")
                summary = self._finalize_summary(execution_metadata, 0, 0, 0, 0)
                self._update_run_status(
                    status="success",
                    records_processed=0,
                    records_failed=0,
                    error_summary="No candidates found",
                    execution_metadata=summary,
                )
                self._persist_execution_log(summary)
                report = self._generate_json_report(summary)
                self._save_json_report(report)
                return summary

            # ── Phase 1: Extract from ALL candidates ───────────────────────────
            # Nothing is saved to DB here. Each runner returns its contacts.
            all_extracted_contacts: List[Dict] = []
            candidate_results = []

            for candidate in candidates:
                result = self.candidate_runner.run(candidate)
                candidate_results.append(result)
                execution_metadata["candidates"].append(result.to_metadata())
                total_emails_fetched += result.emails_fetched

                if result.status != "success":
                    total_failed += 1
                    self.logger.error(
                        "Candidate %s (%s) FAILED: %s",
                        result.candidate_id,
                        result.email,
                        result.error,
                    )
                else:
                    all_extracted_contacts.extend(result.extracted_contacts)

            # ── Phase 2: Single bulk save for ALL candidates ───────────────────
            self.logger.info("=" * 70)
            self.logger.info("BULK SAVE: %d contacts from all candidates", len(all_extracted_contacts))
            self.logger.info("=" * 70)

            save_result = {"contacts_inserted": 0, "contacts_skipped": 0,
                           "positions_inserted": 0, "positions_skipped": 0}

            if all_extracted_contacts and self.vendor_util:
                try:
                    save_result = self.vendor_util.save_contacts(all_extracted_contacts)
                    total_contacts = save_result.get("contacts_inserted", 0)
                    total_positions = save_result.get("positions_inserted", 0)
                    self.logger.info(
                        "Bulk save complete: %d contacts inserted, %d positions inserted, "
                        "%d skipped",
                        total_contacts,
                        total_positions,
                        save_result.get("contacts_skipped", 0),
                    )
                except Exception as save_error:
                    self.logger.error("Bulk save failed: %s", save_error, exc_info=True)
            elif not self.vendor_util:
                self.logger.warning("vendor_util not available — skipping DB/API save")
            else:
                self.logger.info("No contacts extracted — nothing to save")

            # Update candidate-level contacts_saved / positions_saved in metadata
            # (We distribute the totals to the summary; per-candidate is already 0)
            for meta in execution_metadata["candidates"]:
                meta["bulk_contacts_inserted"] = total_contacts
                meta["bulk_positions_inserted"] = total_positions

            # ── Phase 3: Log activity per candidate ──────────────────────────
            for result in candidate_results:
                self._log_activity(
                    candidate_id=result.candidate_id,
                    email=result.email,
                    contacts_saved=result.contacts_saved,
                    positions_saved=result.positions_saved,
                    emails_fetched=result.emails_fetched,
                    filter_stats=result.filter_stats,
                    error_message=result.error,
                )

            overall_status = "success"
            if total_failed > 0:
                overall_status = "partial_success"
            if total_failed == len(candidates):
                overall_status = "failed"

            error_summary = None
            error_details = None
            if total_failed > 0:
                failed_results = [
                    c for c in execution_metadata["candidates"] if c.get("status") != "success"
                ]
                error_summary = f"{total_failed} candidates failed"
                error_details = "\n".join(
                    [f"{c.get('candidate_email')}: {c.get('error')}" for c in failed_results]
                )

            summary = self._finalize_summary(
                execution_metadata,
                total_contacts,
                total_positions,
                total_failed,
                total_emails_fetched,
            )
            self._update_run_status(
                status=overall_status,
                records_processed=total_contacts,
                records_failed=total_failed,
                error_summary=error_summary,
                error_details=error_details,
                execution_metadata=summary,
            )
            self._persist_execution_log(summary)

            # ── Report: JSON save + SMTP email ────────────────────────────────
            report = self._generate_json_report(summary)
            self._save_json_report(report)
            self.email_reporter.send_report(report)

            return summary

        except Exception as error:
            self.logger.error("Service execution failed: %s", error, exc_info=True)
            execution_metadata["fatal_error"] = str(error)
            summary = self._finalize_summary(
                execution_metadata,
                total_contacts,
                total_positions,
                total_failed,
                total_emails_fetched,
            )
            self._update_run_status(
                status="failed",
                records_processed=total_contacts,
                records_failed=total_failed,
                error_summary=str(error)[:255],
                error_details=str(error),
                execution_metadata=summary,
            )
            self._persist_execution_log(summary)
            report = self._generate_json_report(summary)
            self._save_json_report(report)
            self.email_reporter.send_report(report)
            raise

    def _update_run_status(
        self,
        status: str,
        records_processed: int,
        records_failed: int,
        error_summary: Optional[str] = None,
        error_details: Optional[str] = None,
        execution_metadata: Optional[Dict] = None,
    ):
        if not self.workflow_manager or not self.run_id:
            return
        self.workflow_manager.update_run_status(
            self.run_id,
            status,
            records_processed=records_processed,
            records_failed=records_failed,
            error_summary=error_summary,
            error_details=error_details,
            execution_metadata=execution_metadata,
        )

    def _finalize_summary(
        self,
        execution_metadata: Dict,
        total_contacts: int,
        total_positions: int,
        total_failed: int,
        total_emails_fetched: int,
    ) -> Dict:
        candidates = execution_metadata.get("candidates", [])
        success_candidates = [
            item.get("candidate_email") for item in candidates if item.get("status") == "success"
        ]
        failed_candidates = [
            item.get("candidate_email") for item in candidates if item.get("status") != "success"
        ]
        execution_metadata["summary"] = {
            "total_candidates": len(candidates),
            "success_count": len(success_candidates),
            "failure_count": len(failed_candidates),
            "total_contacts_inserted": total_contacts,
            "total_positions_inserted": total_positions,
            "total_emails_fetched": total_emails_fetched,
            "total_candidates_failed": total_failed,
            "successful_candidates": success_candidates,
            "failed_candidates": failed_candidates,
        }
        execution_metadata["finished_at"] = datetime.utcnow().isoformat()
        return execution_metadata

    def _persist_execution_log(self, execution_metadata: Dict):
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            output_dir = _PROJECT_ROOT / "output" / date_str
            output_dir.mkdir(parents=True, exist_ok=True)
            run_label = self.run_id or "manual"
            output_file = output_dir / f"run_{run_label}.json"
            with open(output_file, "w", encoding="utf-8") as output_stream:
                json.dump(execution_metadata, output_stream, indent=2, default=str)
            self.logger.info("Saved execution log to %s", output_file)
        except Exception as error:
            self.logger.error("Failed to save execution log to file: %s", error)

    def _generate_json_report(self, execution_metadata: Dict) -> Dict:
        """Generate a comprehensive JSON report for the extraction run."""
        candidates_data = execution_metadata.get("candidates", [])

        total_emails_fetched = sum(c.get("emails_fetched", 0) for c in candidates_data)
        total_duplicates = sum(c.get("duplicates_skipped", 0) for c in candidates_data)
        total_non_vendor = sum(c.get("non_vendor_filtered", 0) for c in candidates_data)
        total_inserted = sum(c.get("emails_inserted", 0) for c in candidates_data)

        successful = [c for c in candidates_data if c.get("status") == "success"]
        failed = [c for c in candidates_data if c.get("status") != "success"]

        started = execution_metadata.get("started_at")
        finished = execution_metadata.get("finished_at")
        duration_seconds = None
        if started and finished:
            try:
                start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                finish_dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
                duration_seconds = (finish_dt - start_dt).total_seconds()
            except Exception:
                pass

        return {
            "run_metadata": {
                "run_id": self.run_id,
                "workflow_id": self.workflow_id,
                "workflow_key": "email_extractor",
                "started_at": started,
                "finished_at": finished,
                "duration_seconds": duration_seconds,
            },
            "summary": {
                "total_candidates": len(candidates_data),
                "successful_candidates": len(successful),
                "failed_candidates": len(failed),
                "total_emails_fetched": total_emails_fetched,
                "total_emails_inserted": total_inserted,
                "total_duplicates": total_duplicates,
                "total_non_vendor": total_non_vendor,
            },
            "candidates": candidates_data,
            "successful_candidates": [c.get("candidate_email") for c in successful],
            "failed_candidates": [c.get("candidate_email") for c in failed],
            "failed_candidate_details": [
                {
                    "candidate_email": c.get("candidate_email"),
                    "candidate_id": c.get("candidate_id"),
                    "candidate_name": c.get("candidate_name"),
                    "error": c.get("error"),
                }
                for c in failed
            ],
        }

    def _save_json_report(self, report: Dict):
        """Save the JSON report to the output/reports directory."""
        try:
            reports_dir = _PROJECT_ROOT / "output" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"extraction_report_{timestamp}.json"
            latest_report = reports_dir / "latest_extraction_report.json"

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            with open(latest_report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info("=" * 80)
            self.logger.info("📊 EXTRACTION REPORT SAVED")
            self.logger.info("=" * 80)
            self.logger.info("📁 Timestamped: %s", report_file)
            self.logger.info("📄 Latest:      %s", latest_report)
            self.logger.info("=" * 80)
        except Exception as error:
            self.logger.error("Failed to save JSON report: %s", error)

    def _log_activity(
        self,
        candidate_id: Optional[int],
        email: str,
        contacts_saved: int,
        positions_saved: int,
        emails_fetched: int,
        filter_stats: Dict,
        error_message: Optional[str],
    ):
        if not self.job_activity_log_util:
            return
        if not candidate_id:
            self.logger.warning("Missing candidate_id for %s - skipping job activity log", email)
            return

        notes_parts = [
            f"contacts_inserted={contacts_saved}",
            f"positions_inserted={positions_saved}",
            f"emails_fetched={emails_fetched}",
        ]
        if filter_stats:
            notes_parts.append(
                "filters="
                f"passed:{filter_stats.get('passed', 0)},"
                f"junk:{filter_stats.get('junk', 0)},"
                f"not_recruiter:{filter_stats.get('not_recruiter', 0)},"
                f"calendar:{filter_stats.get('calendar_invites', 0)}"
            )
        if error_message:
            notes_parts.append(f"error={error_message}")

        notes = " | ".join(notes_parts)
        self.job_activity_log_util.log_activity(
            candidate_id=candidate_id,
            contacts_extracted=contacts_saved,
            notes=notes,
        )
