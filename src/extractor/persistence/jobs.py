import logging
from typing import List, Dict, Optional
from ..connectors.http_api import APIClient

logger = logging.getLogger(__name__)

class JobPersistence:
    """
    Handles API interactions for job classification tasks.
    Fetches raw job listings and persists classified valid jobs.
    """

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)

    def fetch_raw_jobs(self, limit: int = 50) -> List[Dict]:
        """
        Fetch raw job listings with status 'new'.
        """
        try:
            self.logger.info(f"Fetching up to {limit} raw jobs with status 'new'")
            response = self.api_client.get(
                "/api/raw-positions/", 
                params={"processing_status": "new", "limit": limit}
            )
            
            # Extract results based on common API patterns
            if isinstance(response, list):
                return response
            elif isinstance(response, dict):
                return response.get('results', response.get('data', []))
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to fetch raw jobs: {e}")
            return []

    def save_valid_job(self, job_data: Dict) -> bool:
        """
        Save a classified valid job to the main job_listings table.
        Endpoint: /api/positions/
        """
        try:
            self.api_client.post("/api/positions/", job_data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save valid job: {e}")
            return False

    def update_raw_status(self, raw_id: int, status: str) -> bool:
        """
        Update the processing status of a raw job listing.
        """
        try:
            self.api_client.put(
                f"/api/raw-positions/{raw_id}/", 
                {"processing_status": status}
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to update raw status for ID {raw_id}: {e}")
            return False
