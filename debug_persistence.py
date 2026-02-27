import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.extractor.connectors.http_api import get_api_client
from src.extractor.persistence.jobs import JobPersistence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_persistence():
    load_dotenv()
    client = get_api_client()
    persistence = JobPersistence(client)
    
    logger.info("--- Cycle 1 ---")
    jobs = persistence.fetch_raw_jobs(limit=1)
    if not jobs:
        logger.info("No jobs found.")
        return
        
    job = jobs[0]
    raw_id = job['id']
    logger.info(f"Fetched ID: {raw_id}")
    
    # Update status
    logger.info(f"Updating ID {raw_id} to 'parsed'...")
    success = persistence.update_raw_status(raw_id, "parsed")
    logger.info(f"Update returned: {success}")
    
    # Check item immediately
    item = persistence.api_client.get(f"/api/raw-positions/{raw_id}")
    new_status = item.get('processing_status')
    logger.info(f"Check Item Status via API: {new_status}")
    if new_status == 'parsed':
        logger.info("SUCCESS: Item updated to 'parsed'.")
    else:
        logger.error(f"FAILURE: Item status is '{new_status}' (expected 'parsed').")
        
    logger.info("--- Cycle 2 ---")
    jobs_2 = persistence.fetch_raw_jobs(limit=1)
    if not jobs_2:
        logger.info("Cycle 2: No jobs found (Good!).")
    else:
        new_id = jobs_2[0]['id']
        logger.info(f"Cycle 2 fetch ID: {new_id}")
        if new_id == raw_id:
            logger.error("FAILURE: List still returns same ID!")
        else:
            logger.info("SUCCESS: Fetched different ID.")
            logger.info("Applying DIRECT UPDATE using client.session.put...")
            try:
                # Use same logic as successful probe
                endpoint = f"/api/raw-positions/{raw_id}"
                url = f"{client.base_url}{endpoint}"
                resp = client.session.put(
                    url,
                    json={"processing_status": "parsed"},
                    follow_redirects=True
                )
                logger.info(f"Manual PUT Status: {resp.status_code}")
                # logger.info(f"Manual PUT History: {[r.status_code for r in resp.history]}")
                
            except Exception as e:
                logger.error(f"Manual PUT Failed: {e}")
            else:
                logger.info("SUCCESS: Fetched different ID.")

if __name__ == "__main__":
    test_persistence()
