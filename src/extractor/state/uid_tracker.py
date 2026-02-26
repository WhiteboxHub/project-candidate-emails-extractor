"""
UID Tracker - Manages last processed email UID per candidate
Prevents re-processing same emails on subsequent runs.

Fallback recovery: when last_run.json is missing, the tracker will try
to rebuild it from the most recent automation_workflow_log entry via the
backend API, so that we never re-process emails already handled.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UIDTracker:
    """Track last processed UID per email account"""

    def __init__(
        self,
        tracker_file: str = "last_run.json",
        api_client=None,
        workflow_id: Optional[int] = None,
    ):
        """
        Initialize UID tracker.

        Args:
            tracker_file: Path to JSON file storing last run data.
            api_client:   Optional APIClient instance.  When provided AND
                          last_run.json doesn't exist, the tracker will call
                          GET /api/automation-workflow-log/latest?workflow_id=…
                          and reconstruct UIDs from execution_metadata so we
                          don't re-process emails from scratch.
            workflow_id:  ID of the workflow, required for the API fallback.
        """
        self.tracker_file = Path(tracker_file)
        self.api_client = api_client
        self.workflow_id = workflow_id
        self.data = self._load()
        self.logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal load / save
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> Dict:
        """Load last run data from JSON file; fall back to API if missing."""
        if not self.tracker_file.exists():
            logger.info("No last_run.json found — attempting API recovery...")
            recovered = self._recover_from_api()
            if recovered:
                logger.info(
                    "Recovered %d accounts from workflow log API. Saving to %s",
                    len(recovered),
                    self.tracker_file,
                )
                self.data = recovered
                self._save()
                return recovered
            else:
                logger.warning(
                    "No API recovery data available — starting fresh (will process ALL emails)"
                )
                return {}

        try:
            with open(self.tracker_file, "r") as f:
                data = json.load(f)
            logger.info("Loaded last_run.json with %d accounts", len(data))
            return data
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s: %s", self.tracker_file, e)
            return {}
        except Exception as e:
            logger.error("Error loading %s: %s", self.tracker_file, e)
            return {}

    def _recover_from_api(self) -> Dict:
        """
        Call GET /api/automation-workflow-log/latest?workflow_id={id} and
        reconstruct a last_run.json-compatible dict from execution_metadata.

        The execution_metadata.candidates[] contains:
          { candidate_email, last_uid, status, ... }

        Returns an empty dict if no suitable log entry exists or the call fails.
        """
        if not self.api_client or not self.workflow_id:
            logger.debug(
                "API fallback skipped: api_client=%s, workflow_id=%s",
                self.api_client, self.workflow_id,
            )
            return {}

        try:
            response = self.api_client.get(
                "/api/automation-workflow-log/latest",
                params={"workflow_id": self.workflow_id},
            )
        except Exception as exc:
            # 404 is expected when there are no previous runs — not an error
            logger.info("No previous workflow log found (first ever run): %s", exc)
            return {}

        if not isinstance(response, dict):
            logger.warning("Unexpected response format from /latest endpoint")
            return {}

        execution_metadata = response.get("execution_metadata") or {}
        if isinstance(execution_metadata, str):
            try:
                execution_metadata = json.loads(execution_metadata)
            except json.JSONDecodeError:
                logger.warning("Could not parse execution_metadata JSON")
                return {}

        candidates = execution_metadata.get("candidates", [])
        if not candidates:
            logger.info("Latest workflow log has no candidates in execution_metadata")
            return {}

        finished_at = response.get("finished_at") or execution_metadata.get(
            "finished_at", datetime.utcnow().isoformat()
        )

        recovered: Dict[str, Any] = {}
        for candidate in candidates:
            email = (candidate.get("candidate_email") or "").strip().lower()
            last_uid = candidate.get("last_uid")
            if not email or last_uid is None:
                continue
            recovered[email] = {
                "last_uid": str(last_uid),
                "last_run": finished_at,
            }

        logger.info(
            "Recovered UIDs for %d candidates from run_id=%s (finished %s)",
            len(recovered),
            response.get("run_id", "?"),
            finished_at,
        )
        return recovered

    def _save(self):
        """Save last run data to JSON file"""
        try:
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tracker_file, "w") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            logger.debug("Saved last_run.json with %d accounts", len(self.data))
        except Exception as e:
            logger.error("Error saving %s: %s", self.tracker_file, e)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — unchanged interface
    # ─────────────────────────────────────────────────────────────────────────

    def get_last_uid(self, email: str) -> Optional[str]:
        """
        Get last processed UID for an email account.

        Returns:
            Last UID string or None if first run.
        """
        email = email.strip().lower()

        if email not in self.data:
            logger.info("First run for %s — will process all emails", email)
            return None

        last_uid = self.data[email].get("last_uid")
        last_run = self.data[email].get("last_run", "unknown")

        logger.info("Last run for %s: UID %s on %s", email, last_uid, last_run)
        return last_uid

    def update_last_uid(self, email: str, uid: str, force_timestamp: bool = False):
        """
        Update last processed UID for an email account.

        Args:
            email:           Email address (lowercase)
            uid:             Last processed UID
            force_timestamp: If True, always update last_run even if UID hasn't advanced.
        """
        email = email.strip().lower()

        try:
            new_uid_int = int(uid)
        except (ValueError, TypeError):
            logger.warning("Invalid UID format for update: %s for %s", uid, email)
            return

        current_data = self.data.get(email)
        if current_data:
            last_uid_str = current_data.get("last_uid")
            try:
                if last_uid_str and int(last_uid_str) > new_uid_int:
                    # Stored UID is strictly higher — don't regress it,
                    # but still refresh the last_run timestamp if forced.
                    if force_timestamp:
                        self.data[email]["last_run"] = datetime.now().isoformat()
                        self._save()
                        logger.debug("Updated timestamp only for %s (UID not advanced)", email)
                    else:
                        logger.debug(
                            "Skipping UID update for %s: stored %s > new %s",
                            email, last_uid_str, uid,
                        )
                    return
            except (ValueError, TypeError):
                pass  # Corrupted stored UID — allow overwrite

        self.data[email] = {
            "last_uid": str(uid),
            "last_run": datetime.now().isoformat(),
        }
        self._save()
        logger.info("Updated %s: last_uid=%s", email, uid)

    def get_all_tracked_accounts(self) -> list:
        """Get list of all tracked email accounts"""
        return list(self.data.keys())

    def remove_account(self, email: str):
        """Remove tracking for an account (forces full re-process next run)"""
        email = email.strip().lower()
        if email in self.data:
            del self.data[email]
            self._save()
            logger.info("Removed tracking for %s", email)

    def get_stats(self) -> Dict:
        """Get statistics about tracked accounts"""
        if not self.data:
            return {"total_accounts": 0, "oldest_run": None, "newest_run": None}

        run_dates = []
        for account_data in self.data.values():
            last_run = account_data.get("last_run")
            if last_run:
                try:
                    run_dates.append(datetime.fromisoformat(last_run))
                except Exception:
                    pass

        return {
            "total_accounts": len(self.data),
            "oldest_run": min(run_dates).isoformat() if run_dates else None,
            "newest_run": max(run_dates).isoformat() if run_dates else None,
        }

    def reset_all(self):
        """Reset all tracking (forces full re-process for all accounts)"""
        self.data = {}
        self._save()
        logger.warning("Reset all UID tracking — will process all emails on next run")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────

_tracker_instances: dict = {}


def get_uid_tracker(
    tracker_file: str = "last_run.json",
    api_client=None,
    workflow_id: Optional[int] = None,
) -> UIDTracker:
    """
    Return a UIDTracker for the given file path, reusing an existing instance
    if possible.  Pass api_client + workflow_id on first call to enable API
    fallback recovery when last_run.json is missing.
    """
    global _tracker_instances
    resolved = str(Path(tracker_file).resolve())
    if resolved not in _tracker_instances:
        _tracker_instances[resolved] = UIDTracker(
            tracker_file, api_client=api_client, workflow_id=workflow_id
        )
    return _tracker_instances[resolved]
