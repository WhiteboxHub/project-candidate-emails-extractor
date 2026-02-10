from typing import List, Dict, Optional
import logging
from ..connectors.http_api import APIClient

logger = logging.getLogger(__name__)


class ActivityLog(Dict):
    pass

class JobActivityLogUtil:
    """
    Utility for logging job activity via API (BULK INSERT)
    Tracks vendor contacts extracted per candidate per day

    Uses SAME API:
    POST /api/job_activity_logs
    """

    def __init__(
        self,
        api_client: APIClient,
        job_unique_id: str = 'bot_candidate_email_extractor'
    ):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        self.job_unique_id = job_unique_id
        self.employee_id = api_client.employee_id
        self._job_type_id = None  # Cache job_type_id

    # --------------------------------------------------
    # Job type lookup (unchanged)
    # --------------------------------------------------
    def _get_job_type_id(self) -> Optional[int]:
        if self._job_type_id:
            return self._job_type_id

        try:
            job_types = self.api_client.get('/api/job-types')

            for job_type in job_types:
                if job_type.get('unique_id') == self.job_unique_id:
                    self._job_type_id = job_type['id']
                    self.logger.info(
                        f"Found job_type_id {self._job_type_id} for '{self.job_unique_id}'"
                    )
                    return self._job_type_id

            self.logger.error(
                f"Job type not found with unique_id: {self.job_unique_id}"
            )
            return None

        except Exception as e:
            self.logger.error(f"Error fetching job_type_id: {str(e)}")
            return None

    # --------------------------------------------------
    # BULK INSERT METHOD
    # --------------------------------------------------
    def log_activities_bulk(
        self,
        activities: List[Dict],
        notes: Optional[str] = None
    ) -> Optional[dict]:
        """
        Log job activities in BULK (single API call)

        Args:
            activities: List of dicts
                [
                  {
                    "candidate_id": int,
                    "contacts_extracted": int
                  }
                ]
            notes: Optional notes for all logs

        Returns:
            API response or None
        """
        if not activities:
            self.logger.info("No activities to log")
            return None

        job_type_id = self._get_job_type_id()
        if not job_type_id:
            self.logger.error("Cannot log activities: job_type_id not found")
            return None

        from datetime import date
        today = date.today().isoformat()

        bulk_logs = []

        for activity in activities:
            candidate_id = activity.get("candidate_id")
            contacts_extracted = activity.get("contacts_extracted", 0)

            if not candidate_id:
                continue

            log_data = {
                "job_id": job_type_id,
                "candidate_id": candidate_id,
                "employee_id": self.employee_id,
                "activity_date": today,
                "activity_count": contacts_extracted
            }

            if notes:
                log_data["notes"] = notes

            bulk_logs.append(log_data)

        if not bulk_logs:
            self.logger.info("No valid activity logs prepared")
            return None

        try:
            self.logger.info(
                f"Bulk logging {len(bulk_logs)} job activity records"
            )

            # SAME API, but BULK payload (list)
            response = self.api_client.post(
                "/api/job_activity_logs/bulk",
                {"logs": bulk_logs}
            )

            self.logger.info(
                f"Bulk activity logging completed for {len(bulk_logs)} candidates"
            )

            return response

        except Exception as e:
            self.logger.error(
                f"API error during bulk activity logging: {str(e)}"
            )
            return None

    # --------------------------------------------------
    # (Optional) Keep single-log method for compatibility
    # --------------------------------------------------
    def log_activity(
        self,
        candidate_id: int,
        contacts_extracted: int,
        notes: Optional[str] = None
    ):
        """
        Backward-compatible single insert (wraps bulk)
        """
        return self.log_activities_bulk(
            activities=[
                {
                    "candidate_id": candidate_id,
                    "contacts_extracted": contacts_extracted
                }
            ],
            notes=notes
        )

    # --------------------------------------------------
    # Summary method (unchanged)
    # --------------------------------------------------
    def get_today_summary(self) -> dict:
        try:
            job_type_id = self._get_job_type_id()
            if not job_type_id:
                return {}

            logs = self.api_client.get(
                f'/api/job_activity_logs/job/{job_type_id}'
            )

            if not logs:
                return {}

            from datetime import date
            today = date.today().isoformat()

            today_logs = [
                log for log in logs
                if log.get('activity_date') == today
            ]

            if not today_logs:
                return {}

            unique_candidates = {
                log['candidate_id'] for log in today_logs
            }
            total_contacts = sum(
                log['activity_count'] for log in today_logs
            )

            return {
                'candidates_processed': len(unique_candidates),
                'total_contacts_extracted': total_contacts
            }

        except Exception as e:
            self.logger.error(f"API error fetching summary: {str(e)}")
            return {}
