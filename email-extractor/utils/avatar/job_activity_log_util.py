from typing import List, Dict, Optional
import logging
from utils.api_client import APIClient

logger = logging.getLogger(__name__)

class JobActivityLogUtil:
    """
    Utility for logging job activity via API
    Tracks vendor contacts extracted per candidate per day
    """
    
    def __init__(self, api_client: APIClient, job_unique_id: str = 'bot_candidate_email_extractor'):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        self.job_unique_id = job_unique_id
        self.employee_id = api_client.employee_id
        self._job_type_id = None  # Cache the job_type_id
    
    def _get_job_type_id(self) -> Optional[int]:
        """
        Get job_type_id by unique_id from the API
        
        Returns:
            job_type_id or None if not found
        """
        if self._job_type_id:
            return self._job_type_id
        
        try:
            # GET /api/job-types to find our job by unique_id
            job_types = self.api_client.get('/api/job-types')
            
            for job_type in job_types:
                if job_type.get('unique_id') == self.job_unique_id:
                    self._job_type_id = job_type['id']
                    self.logger.info(f"Found job_type_id {self._job_type_id} for '{self.job_unique_id}'")
                    return self._job_type_id
            
            self.logger.error(f"Job type not found with unique_id: {self.job_unique_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching job_type_id: {str(e)}")
            return None
    
    def log_activity(self, candidate_id: int, contacts_extracted: int, notes: Optional[str] = None):
        """
        Log job activity for a candidate via API
        
        Args:
            candidate_id: candidate.id (FK)
            contacts_extracted: Number of vendor contacts saved to database
            notes: Optional notes (errors, filter stats, etc.)
        """
        try:
            job_type_id = self._get_job_type_id()
            
            if not job_type_id:
                self.logger.error("Cannot log activity: job_type_id not found")
                return None
            
            # POST /api/job_activity_logs
            from datetime import date
            
            log_data = {
                "job_id": job_type_id,  # API expects 'job_id' not 'job_type_id'
                "candidate_id": candidate_id,
                "employee_id": self.employee_id,
                "activity_date": date.today().isoformat(),  # Required field (YYYY-MM-DD format)
                "activity_count": contacts_extracted
            }
            
            # Add notes if provided
            if notes:
                log_data["notes"] = notes
            
            response = self.api_client.post('/api/job_activity_logs', log_data)
            
            self.logger.info(
                f"Activity logged for candidate_id {candidate_id}: "
                f"{contacts_extracted} inserted"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"API error logging activity for candidate {candidate_id}: {str(e)}")
            return None
    
    def get_today_summary(self) -> dict:
        """
        Get summary of today's extraction activity via API
        
        Returns:
            Dictionary with summary statistics
        """
        try:
            job_type_id = self._get_job_type_id()
            
            if not job_type_id:
                return {}
            
            # GET /api/job_activity_logs/job/{job_type_id}
            logs = self.api_client.get(f'/api/job_activity_logs/job/{job_type_id}')
            
            if not logs:
                return {}
            
            # Filter for today and calculate summary
            from datetime import date
            today = date.today().isoformat()
            
            today_logs = [log for log in logs if log.get('activity_date') == today]
            
            if not today_logs:
                return {}
            
            unique_candidates = set(log['candidate_id'] for log in today_logs)
            total_contacts = sum(log['activity_count'] for log in today_logs)
            
            return {
                'candidates_processed': len(unique_candidates),
                'total_contacts_extracted': total_contacts
            }
            
        except Exception as e:
            self.logger.error(f"API error fetching summary: {str(e)}")
            return {}

