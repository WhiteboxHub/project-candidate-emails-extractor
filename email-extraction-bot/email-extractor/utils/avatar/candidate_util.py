import mysql.connector
import logging
from typing import List, Dict, Optional

class CandidateUtil:
    """
    Utility for fetching candidate marketing accounts
    """
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.logger = logging.getLogger(__name__)
    
    def get_active_candidates(self) -> List[Dict]:
        """
        Fetch ALL candidates with email credentials (ignores status)
        
        Returns:
            List of candidate dictionaries with email and imap_password
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT 
                    cm.id,
                    cm.candidate_id,
                    cm.email,
                    cm.imap_password,
                    cm.status,
                    cm.priority
                FROM candidate_marketing cm
                WHERE cm.email IS NOT NULL 
                  AND cm.email != ''
                  AND cm.imap_password IS NOT NULL 
                  AND cm.imap_password != ''
                ORDER BY cm.priority ASC, cm.id ASC
            """
            
            cursor.execute(query)
            candidates = cursor.fetchall()
            
            # Set name to email if not available
            for candidate in candidates:
                candidate['name'] = candidate.get('email', 'Unknown')
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"Fetched {len(candidates)} candidates with email credentials")
            return candidates
            
        except mysql.connector.Error as err:
            self.logger.error(f"Database error fetching candidates: {err}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching candidates: {str(e)}")
            return []
    
    def get_candidate_by_id(self, candidate_id: int) -> Optional[Dict]:
        """
        Fetch a specific candidate by ID
        
        Args:
            candidate_id: Candidate marketing ID
            
        Returns:
            Candidate dictionary or None
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT 
                    cm.id,
                    cm.candidate_id,
                    cm.email,
                    cm.imap_password,
                    cm.status
                FROM candidate_marketing cm
                WHERE cm.id = %s
            """
            
            cursor.execute(query, (candidate_id,))
            candidate = cursor.fetchone()
            
            if candidate:
                candidate['name'] = candidate.get('email', 'Unknown')
            
            cursor.close()
            conn.close()
            
            return candidate
            
        except mysql.connector.Error as err:
            self.logger.error(f"Database error fetching candidate: {err}")
            return None
