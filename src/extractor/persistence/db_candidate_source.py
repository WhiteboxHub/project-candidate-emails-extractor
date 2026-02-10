import logging
from typing import List, Dict
from ..core.database import get_db_client

logger = logging.getLogger(__name__)

class DatabaseCandidateSource:
    """
    Fetches candidate credentials using a raw SQL query provided by the workflow configuration.
    """
    
    def __init__(self, credentials_sql: str):
        self.db_client = get_db_client()
        self.credentials_sql = credentials_sql

    def get_active_candidates(self) -> List[Dict]:
        """
        Execute the stored SQL to get candidates.
        Expecting columns: candidate_id, email, password, imap_server (optional)
        """
        if not self.credentials_sql:
            logger.warning("No credentials SQL provided to DatabaseCandidateSource")
            return []
            
        try:
            results = self.db_client.execute_query(self.credentials_sql)
            
            candidates = []
            for row in results:
                # Map raw DB row to expected dictionary format
                candidate = {
                    'id': row.get('candidate_id'), # Using candidate_id as the main ID
                    'candidate_id': row.get('candidate_id'),
                    'email': row.get('email'),
                    'imap_password': row.get('password'), # Mapping password to imap_password
                    'imap_server': row.get('imap_server', 'imap.gmail.com'),
                    'name': row.get('email') # Default name to email if not present
                }
                
                # Basic validation
                if candidate.get('email') and candidate.get('imap_password'):
                    candidates.append(candidate)
                else:
                    logger.warning(f"Skipping invalid candidate row: {row}")

            logger.info(f"Fetched {len(candidates)} candidates from database source")
            return candidates
            
        except Exception as e:
            logger.error(f"Error fetching candidates from database: {e}")
            return [] 
