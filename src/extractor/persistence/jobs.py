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

    def fetch_raw_jobs(self, limit: int = 50, skip: int = 0) -> List[Dict]:
        """
        Fetch raw job listings with status 'new'.
        Supports skip-based pagination.
        """
        try:
            self.logger.info(f"Fetching up to {limit} raw jobs (skip={skip}) with status 'new'")
            response = self.api_client.get(
                "/api/raw-positions/", 
                params={"processing_status": "new", "limit": limit, "skip": skip}
            )
            
            # Extract results based on common API patterns
            jobs = []
            if isinstance(response, list):
                jobs = response
            elif isinstance(response, dict):
                jobs = response.get('results', response.get('data', []))
            
            # Debug logging for statuses
            if jobs:
                sample_statuses = [f"ID {j.get('id')}: {j.get('processing_status')}" for j in jobs[:3]]
                self.logger.info(f"Sample statuses from API: {', '.join(sample_statuses)}")

            # Specific Fix: Client-side filter to ensure only 'new' records are processed
            filtered_jobs = [j for j in jobs if j.get('processing_status') == 'new']
            
            if len(filtered_jobs) < len(jobs):
                self.logger.warning(f"Filtered out {len(jobs) - len(filtered_jobs)} records that were not 'new'")
                
            return filtered_jobs, len(jobs)
            
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
            # Server definition: @router.put("/{raw_job_listing_id}")
            # Router prefix: /raw-positions
            # NO trailing slash should be used here.
            self.api_client.put(
                f"/api/raw-positions/{raw_id}", 
                {"processing_status": status}
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to update raw status for ID {raw_id}: {e}")
            return False
