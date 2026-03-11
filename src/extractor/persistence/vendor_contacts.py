"""
vendor_contacts.py
==================
Persist extracted vendor/recruiter contacts plus raw job positions.

Design:
- save_contacts() is called ONCE by service.py after all candidates run.
- All three tables are populated in bulk:
  1. automation_contact_extracts  (audit / dedup table) — via API (INSERT IGNORE)
  2. vendor_contact via /api/vendor_contact/bulk
  3. raw_positions  via /api/raw-positions/bulk
"""

from typing import Dict, List, Optional
import logging
import json

from ..connectors.http_api import APIClient
from ..filtering.repository import get_filter_repository
from .duckdb_raw_listings import RawJobListingsDuckDB
from datetime import datetime
import sys
import time
from pathlib import Path
from ..extraction.ner_validator import NERValidator

# Auto-log generator (scripts/generate_duckdb_log.py)
try:
    _SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from generate_duckdb_log import write_duckdb_log as _write_duckdb_log, _print_summary as _print_duckdb_summary
    _HAS_DUCKDB_LOG = True
except Exception:
    _HAS_DUCKDB_LOG = False

logger = logging.getLogger(__name__)


class VendorUtil:
    """Persist extracted vendor/recruiter contacts and related raw positions in bulk."""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        self.filter_repo = get_filter_repository()

    # ─────────────────────────────────────────────────────────────────────────
    # Deduplication helpers — now use API calls instead of raw SQL
    # ─────────────────────────────────────────────────────────────────────────

    def get_existing_emails(self, source_email: str) -> set:
        """
        Fetch existing vendor emails for this candidate to avoid re-inserting.
        Calls GET /api/automation-extracts?source_email=<email>
        """
        if not source_email:
            return set()
        try:
            response = self.api_client.get(
                f"/api/automation-extracts?source_email={source_email}"
            )
            records = response if isinstance(response, list) else (response or {}).get("data", [])
            existing = {r["email"].strip().lower() for r in records if r.get("email")}
            self.logger.info("Loaded %d existing contacts for deduplication", len(existing))
            return existing
        except Exception as e:
            self.logger.error("Failed to fetch existing contacts: %s", e)
            return set()

    def get_recent_vendor_emails(self, limit: int = 5000) -> set:
        """
        Fetch recently extracted unique vendor emails for global deduplication cache.
        Falls back to get_globally_existing_emails() with an empty list — the caller
        should use get_globally_existing_emails() directly for a targeted check.
        """
        try:
            # Note: The backend endpoint might not support limit/order_by yet, 
            # but we use the general automation-extracts endpoint.
            response = self.api_client.get(
                f"/api/automation-extracts"
            )
            records = response if isinstance(response, list) else (response or {}).get("data", [])
            existing = {r["email"].strip().lower() for r in records if r.get("email")}
            self.logger.info("Loaded %d global vendor emails for cache", len(existing))
            return existing
        except Exception as e:
            self.logger.error("Failed to fetch global vendor cache: %s", e)
            return set()

    def get_globally_existing_emails(self, emails: List[str]) -> set:
        """
        Check which emails from the provided list already exist globally.
        Calls POST /api/automation-extracts/check-emails — one bulk API call.
        """
        if not emails:
            return set()
        try:
            response = self.api_client.post(
                "/api/automation-extracts/check-emails",
                {"emails": emails},
            )
            found = response.get("existing_emails", []) if isinstance(response, dict) else []
            return {e.strip().lower() for e in found if e}
        except Exception as e:
            self.logger.error("Failed to check global existing emails: %s", e)
            return set()

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point — called ONCE after all candidates have been processed
    # ─────────────────────────────────────────────────────────────────────────

    def save_contacts(self, contacts: List[Dict], candidate_id: Optional[int] = None) -> Dict[str, int]:
        """
        Bulk-save all extracted contacts to three destinations:
          1. automation_contact_extracts  (INSERT IGNORE — audit/dedup table)
          2. vendor_contact via API        (truly new contacts only)
          3. raw_positions  via API        (truly new contacts only)

        contacts is the flat list accumulated from ALL candidate runs.
        Each contact dict may carry a 'candidate_id' key set by candidate_runner.

        Returns dict with insert / skip counts.
        """
        result = {
            "contacts_inserted": 0,
            "contacts_skipped": 0,
            "positions_inserted": 0,
            "positions_skipped": 0,
            "extracts_inserted": 0,
            "extracts_skipped": 0,
            "positions_finalized": 0,
            "ner_fallback_inserted": 0,
        }
        if not contacts:
            self.logger.info("No contacts to save")
            return result

        # ── Step 1: Validate and local-dedup ─────────────────────────────────
        filtered_contacts: List[Dict] = []
        seen_keys: set = set()

        for contact in contacts:
            if not self._is_valid_contact(contact):
                result["contacts_skipped"] += 1
                continue
            if not self._is_vendor_recruiter_contact(contact):
                result["contacts_skipped"] += 1
                continue

            email_key = (contact.get("email") or "").strip().lower()
            linkedin_key = (contact.get("linkedin_id") or "").strip().lower()
            dedupe_key = email_key or f"li:{linkedin_key}"
            if dedupe_key and dedupe_key in seen_keys:
                result["contacts_skipped"] += 1
                continue
            if dedupe_key:
                seen_keys.add(dedupe_key)

            filtered_contacts.append(contact)

        if not filtered_contacts:
            self.logger.info("No vendor/recruiter contacts after validation")
            return result

        # ── Step 2: Global DB dedup ──────────────────────────────────────────
        candidate_emails = [
            c.get("email").strip().lower()
            for c in filtered_contacts
            if c.get("email")
        ]
        existing_global_emails = self.get_globally_existing_emails(candidate_emails)

        truly_new_contacts = [
            c for c in filtered_contacts
            if (c.get("email") or "").strip().lower() not in existing_global_emails
        ]
        result["contacts_skipped"] += len(filtered_contacts) - len(truly_new_contacts)

        # ── Step 3: Bulk INSERT IGNORE → automation_contact_extracts ─────────
        # ALL filtered contacts are recorded (new ones as 'new', duplicates
        # are silently ignored by INSERT IGNORE via the unique index).
        ext_inserted, ext_skipped = self._bulk_insert_contact_extracts(filtered_contacts, existing_global_emails)
        result["extracts_inserted"] = ext_inserted
        result["extracts_skipped"] = ext_skipped

        if not truly_new_contacts:
            self.logger.info("No truly new contacts — all duplicates recorded in audit table.")
            return result

        # ── Step 4: Bulk POST → vendor_contact API ────────────────────────────
        bulk_contacts = self._build_vendor_contacts_payload(truly_new_contacts)
        if bulk_contacts:
            try:
                self.logger.info("Sending %s contacts to /api/vendor_contact/bulk", len(bulk_contacts))
                response = self.api_client.post("/api/vendor_contact/bulk", {"contacts": bulk_contacts})
                inserted, skipped = self._extract_insert_skip_counts(response, default_inserted=len(bulk_contacts))
                result["contacts_inserted"] = inserted
                result["contacts_skipped"] += skipped
            except Exception as error:
                self.logger.error("API error saving vendor contacts: %s", error)
                return result
        else:
            self.logger.info("No contacts prepared for vendor_contact bulk insert")

        # ── Step 5: raw_job_listings ──────────────────────────────────────────
        # candidate_id may be None (run across multiple candidates) — each
        # contact carries its own 'candidate_id' set by candidate_runner.
        raw_job_listings = self._build_raw_job_listings_payload(truly_new_contacts)
        if raw_job_listings:
            # ── Step 5a: Local DuckDB only (API path disabled for now) ────────
            # TODO: When ready to push to production, uncomment Step 5b below.
            try:
                duckdb_store = RawJobListingsDuckDB()
                duckdb_inserted = duckdb_store.insert_bulk(raw_job_listings)
                duckdb_stats = duckdb_store.get_stats()
                duckdb_store.close()
                self.logger.info(
                    "DuckDB: %d/%d raw_job_listings rows stored | cumulative stats: %s",
                    duckdb_inserted,
                    len(raw_job_listings),
                    duckdb_stats,
                )
                result["positions_inserted"] = duckdb_inserted

                # ── Auto-generate duckdb_logs.json ────────────────────────
                if _HAS_DUCKDB_LOG:
                    try:
                        import json as _json
                        log_path = _write_duckdb_log()
                        self.logger.info("DuckDB log written → %s", log_path)
                        # Print colored summary to terminal so user sees results immediately
                        _log_data = _json.loads(log_path.read_text(encoding="utf-8"))
                        _print_duckdb_summary(_log_data)
                        run_num = _log_data.get("run_number", "?")
                        print(f"  ✓ DuckDB run #{run_num} log saved → {log_path.name}")
                        print(f"    (Latest alias: data/duckdb_logs.json)\n")
                    except Exception as log_err:
                        self.logger.warning("Could not write duckdb_logs.json: %s", log_err)
            except Exception as duckdb_error:
                self.logger.warning("DuckDB write failed: %s", duckdb_error)

            # ── Step 5b: Bulk POST → /api/raw-positions/bulk  [DISABLED] ─────
            # Uncomment the block below when you are ready to push to MySQL:
            #
            # try:
            #     self.logger.info("Sending %s raw job listings to /api/raw-positions/bulk", len(raw_job_listings))
            #     response = self.api_client.post("/api/raw-positions/bulk", {"positions": raw_job_listings})
            #     inserted, skipped = self._extract_insert_skip_counts(response, default_inserted=len(raw_job_listings))
            #     result["positions_inserted"] = inserted
            #     result["positions_skipped"] = skipped
            # except Exception as error:
            #     self.logger.error("Error saving raw job listings: %s", error)
        else:
            self.logger.info("No raw job listings produced from filtered vendor contacts")

        # ── Step 6: Job Finalization (NER Path) ──────────────────────────────
        # Route high-quality jobs to job_listings, and leftovers to email_positions
        if truly_new_contacts:
            self.logger.info("Running NER validation and routing for %d new contacts...", len(truly_new_contacts))
            
            ner_validator = NERValidator(use_gliner=False) # Use regex-based NER for speed in extractor
            ner_fallback_positions = []
            finalized_positions = []
            
            for contact in truly_new_contacts:
                job_data = self._map_contact_to_job_data(contact)
                
                # Mock raw_job for validator
                raw_job_mock = {
                    "id": 0,
                    "raw_description": contact.get("raw_body", ""),
                    "raw_payload": contact
                }
                # Mock llm_result
                llm_result_mock = {
                    "extracted_title": contact.get("job_position", ""),
                    "score": contact.get("data_quality_score", 0) / 100.0 if contact.get("data_quality_score") else 0.5
                }
                
                val_result = ner_validator.validate_and_finalize(raw_job_mock, job_data, llm_result_mock)
                
                if val_result["is_finalized"]:
                    try:
                        self.logger.info("✓ Finalized job found (NER passed) -> /api/positions/")
                        self.api_client.post("/api/positions/", val_result["job_data"])
                        finalized_positions.append(val_result["job_data"])
                        result["positions_finalized"] += 1
                    except Exception as e:
                        self.logger.error("Failed to save finalized job: %s", e)
                else:
                    # NER failed: Check for email, title, and company to fallback to email_positions
                    # Use sanitized values from the validator (recovered titles/companies)
                    final_data = val_result["job_data"]
                    email = (final_data.get("contact_email") or "").strip()
                    title = (final_data.get("title") or "").strip()
                    company = (final_data.get("company_name") or "").strip()
                    
                    if email and title and company:
                        fallback_entry = self._map_contact_to_email_pos(final_data)
                        ner_fallback_positions.append(fallback_entry)
                    else:
                        missing = []
                        if not email: missing.append("email")
                        if not title: missing.append("title")
                        if not company: missing.append("company")
                        self.logger.warning("Skipping email_positions fallback - missing required fields: %s", ", ".join(missing))
            
            # Bulk save fallback positions (LLM and NER fallbacks both go here)
            if ner_fallback_positions:
                try:
                    self.logger.info("Sending %d fallback records to /api/email-positions/bulk", len(ner_fallback_positions))
                    self.api_client.post("/api/email-positions/bulk", {"positions": ner_fallback_positions})
                    result["ner_fallback_inserted"] = len(ner_fallback_positions)
                except Exception as e:
                    self.logger.error("Failed to save bulk email positions: %s", e)

            # ── Step 6c: Save Categorized JSON Result Files ──────────────────
            self._save_categorized_results({
                "finalized": finalized_positions,
                "ner_fallback": ner_fallback_positions
            }, result)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Mapping Helpers for NER Path
    # ─────────────────────────────────────────────────────────────────────────

    def _map_contact_to_job_data(self, contact: Dict) -> Dict:
        """Map contact dict to job_listings schema."""
        def normalize_position_type(raw: str) -> str:
            raw = (raw or '').lower().replace(' ', '_').replace('-', '_')
            if any(x in raw for x in ['w2', 'w_2', 'contract_to_hire', 'c2h', 'contract to hire']):
                return 'contract_to_hire' if 'hire' in raw else 'contract'
            if any(x in raw for x in ['c2c', 'corp', '1099', 'independent']):
                return 'contract'
            if 'full' in raw:
                return 'full_time'
            if 'intern' in raw:
                return 'internship'
            if 'contract' in raw:
                return 'contract'
            return 'full_time'

        def normalize_employment_mode(raw: str) -> str:
            raw = (raw or '').lower()
            if 'remote' in raw:
                return 'remote'
            if 'onsite' in raw or 'on-site' in raw or 'on site' in raw or 'office' in raw:
                return 'onsite'
            return 'hybrid'

        return {
            "title": (contact.get("job_position") or "")[:200],
            "description": contact.get("raw_body"),
            "company_name": (contact.get("company") or "")[:200],
            "position_type": normalize_position_type(contact.get("employment_type") or ""),
            "employment_mode": normalize_employment_mode(contact.get("work_mode") or ""),
            "source": "email",
            "source_uid": contact.get("extracted_from_uid") or "",
            "source_job_id": str(contact.get('linkedin_id') or ''),
            "created_from_raw_id": 0,
            "location": contact.get("location") or "",
            "zip": contact.get("zip_code") or "",
            "country": "USA",
            "contact_email": (contact.get("email") or "").strip(),
            "contact_phone": contact.get("phone") or "",
            "job_url": contact.get("job_url") or "",
            "notes": f"Extracted from {contact.get('extraction_source')}",
            "confidence_score": contact.get("data_quality_score", 0) / 100.0 if contact.get("data_quality_score") else 0.5,
            "candidate_id": contact.get("candidate_id"),
            "raw_payload": contact
        }

    def _map_contact_to_email_pos(self, job_data: Dict) -> Dict:
        """Map sanitized job_data to email_positions schema (fallback)."""
        return {
            "title": job_data.get("title"),
            "company": job_data.get("company_name"),
            "location": job_data.get("location"),
            "email": job_data.get("contact_email"),
            "source": "email",
            "candidate_id": job_data.get("candidate_id"),
            "raw_payload": job_data.get("raw_payload")  # Original contact dict included in job_data mock
        }

    def _save_categorized_results(self, records: Dict[str, List[Dict]], stats: Dict):
        """Save categorized records to JSON files for auditing, similar to LLM classifier."""
        try:
            project_root = Path(__file__).resolve().parents[3]
            output_dir = project_root / "output" / "extraction_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            for category, data in records.items():
                if not data:
                    continue
                    
                filename = output_dir / f"extraction_{category}_{timestamp}.json"
                
                result_package = {
                    "summary": {
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_extracted": len(records.get("finalized", [])) + len(records.get("ner_fallback", [])),
                        "positions_finalized": stats.get("positions_finalized", 0),
                        "ner_fallback_inserted": stats.get("ner_fallback_inserted", 0),
                        "file_category": category,
                        "file_record_count": len(data)
                    },
                    "records": data
                }
                
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result_package, f, indent=2, ensure_ascii=False)
                self.logger.info("Saved %s records to: %s", category, filename)
                
        except Exception as e:
            self.logger.error("Failed to save categorized results: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Bulk API insert — automation_contact_extracts
    # ─────────────────────────────────────────────────────────────────────────

    def _bulk_insert_contact_extracts(
        self,
        contacts: List[Dict],
        existing_global_emails: set,
    ) -> tuple[int, int]:
        """
        Bulk-insert all contacts into automation_contact_extracts via API.
        The backend does INSERT IGNORE so duplicates are silently skipped.
        Returns (inserted, skipped).
        """
        if not contacts:
            return 0, 0

        rows = []
        for contact in contacts:
            email_lc = (contact.get("email") or "").strip().lower()
            status = "duplicate" if email_lc in existing_global_emails else "new"
            rows.append({
                "full_name":       contact.get("name"),
                "email":           contact.get("email"),
                "phone":           contact.get("phone"),
                "company_name":    contact.get("company"),
                "job_title":       contact.get("job_position"),
                "city":            contact.get("location"),
                "postal_code":     contact.get("zip_code"),
                "linkedin_id":     contact.get("linkedin_id"),
                "source_type":     "email",
                "source_reference": contact.get("source"),
                "raw_payload":     contact,
                "processing_status": status,
                "classification":  "unknown",
            })

        try:
            response = self.api_client.post(
                "/api/automation-extracts/bulk",
                {"extracts": rows},
            )
            if isinstance(response, dict):
                inserted = response.get("inserted", 0)
                duplicates = response.get("duplicates", 0)
                failed = response.get("failed", 0)
                self.logger.info(
                    "automation-extracts bulk: %d rows → %d inserted, %d duplicates, %d failed",
                    response.get("total", len(rows)),
                    inserted,
                    duplicates,
                    failed,
                )
                return inserted, (duplicates + failed)
        except Exception as e:
            self.logger.error("Error in audit bulk insert: %s", e)
        
        return 0, len(rows)  # Default on error


    # ─────────────────────────────────────────────────────────────────────────
    # Payload builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_vendor_contacts_payload(self, contacts: List[Dict]) -> List[Dict]:
        payload = []
        for contact in contacts:
            full_name = (contact.get("name") or "").strip()
            if not full_name:
                continue
            item = {
                "full_name": full_name,
                "source_email": contact.get("source"),
                "email": contact.get("email"),
                "phone": contact.get("phone"),
                "linkedin_id": contact.get("linkedin_id"),
                "company_name": contact.get("company"),
                "location": contact.get("location"),
                "extraction_date": datetime.now().date().isoformat(),
                "job_source": "Bot Candidate Email Extractor",
            }
            item = {k: v for k, v in item.items() if v not in (None, "")}
            if "full_name" in item:
                payload.append(item)
        return payload

    def _build_raw_job_listings_payload(self, contacts: List[Dict]) -> List[Dict]:
        """Build one raw job listing per contact that has job-content signals."""
        payload = []
        for contact in contacts:
            has_position_signal = any(
                contact.get(field) for field in ("job_position", "raw_body", "location", "company")
            )
            if not has_position_signal:
                continue

            # Use the candidate_id tagged per-contact by candidate_runner
            cid = contact.get("candidate_id")

            contact_info = {
                "name": contact.get("name"),
                "email": contact.get("email"),
                "phone": contact.get("phone"),
                "linkedin": contact.get("linkedin_id"),
            }
            payload.append({
                "candidate_id": cid,
                "source": "email",
                "source_uid": contact.get("extracted_from_uid"),
                "extractor_version": "v1.0",
                "raw_title": contact.get("job_position"),
                "raw_company": contact.get("company"),
                "raw_location": contact.get("location"),
                "raw_zip": contact.get("zip_code"),
                "raw_description": contact.get("raw_body"),
                "raw_contact_info": json.dumps(contact_info),
                "raw_notes": f"Extracted from {contact.get('extraction_source')}",
                "raw_payload": contact,
                "processing_status": "new",
            })
        return payload

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_insert_skip_counts(self, response: Dict, default_inserted: int) -> tuple:
        if isinstance(response, dict):
            inserted = int(response.get("inserted", response.get("saved", default_inserted)) or 0)
            skipped = int(response.get("skipped", 0) or 0)
            return inserted, skipped
        return default_inserted, 0

    def _is_vendor_recruiter_contact(self, contact: Dict) -> bool:
        source_email = (contact.get("source") or "").strip().lower()
        contact_email = (contact.get("email") or "").strip().lower()
        if source_email and contact_email and source_email == contact_email:
            return False
        if contact_email:
            action = self.filter_repo.check_email(contact_email)
            if action == "block":
                return False
        return True

    def _is_valid_contact(self, contact: Dict) -> bool:
        """Validate contact has minimum required quality."""
        try:
            email = (contact.get("email") or "").strip()
            linkedin = (contact.get("linkedin_id") or "").strip()
            name = (contact.get("name") or "").strip()

            if not email and not linkedin:
                return False

            if email:
                if "@" not in email or "." not in email:
                    return False
                email_lower = email.lower()
                blocked_local_parts = ["noreply", "no-reply", "info@", "support@", "admin@"]
                if any(token in email_lower for token in blocked_local_parts):
                    return False

            if linkedin and (" " in linkedin or len(linkedin) > 80):
                return False

            if name:
                words = name.split()
                if len(words) < 2 or len(words) > 6:
                    return False
                if any(char.isdigit() for char in name):
                    return False
            return True
        except Exception as error:
            self.logger.error("Error validating contact: %s", error)
            return False
