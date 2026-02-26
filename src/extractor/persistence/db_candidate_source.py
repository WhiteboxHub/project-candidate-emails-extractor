import logging
from typing import List, Dict, Optional
from ..core.database import get_db_client

logger = logging.getLogger(__name__)

class DatabaseCandidateSource:
    """
    Fetches candidate credentials using a raw SQL query provided by the workflow configuration.
    """
    
    def __init__(self, credentials_sql: str):
        self.db_client = get_db_client()
        self.credentials_sql = credentials_sql

    def get_active_candidates(
        self,
        candidate_id: Optional[int] = None,
        candidate_email: Optional[str] = None,
    ) -> List[Dict]:
        """
        Execute the stored SQL to get candidates.
        SQL comes from automation_workflows.credentials_list_sql and can use
        different aliases for the same logical fields.
        """
        if not self.credentials_sql:
            logger.warning("No credentials SQL provided to DatabaseCandidateSource")
            return []
            
        try:
            results = self.db_client.execute_query(self.credentials_sql)
            
            if results:
                logger.debug(f"DB Columns found: {list(results[0].keys())}")
            
            candidates = []
            for row in results:
                resolved_candidate_id = self._pick(
                    row,
                    ["candidate_id", "id", "candidate_marketing_id", "candidateMarketingId"],
                )
                resolved_email = self._pick(row, ["email", "imap_email", "candidate_email", "username"])
                resolved_password = self._pick(
                    row,
                    ["imap_password", "password", "app_password", "email_password"],
                )
                resolved_imap_server = self._pick(
                    row,
                    ["imap_server", "email_server", "server"],
                    default="imap.gmail.com",
                )
                resolved_name = self._pick(
                    row,
                    ["name", "full_name", "candidate_name"],
                    default=resolved_email,
                )

                candidate = {
                    "id": resolved_candidate_id,
                    "candidate_id": resolved_candidate_id,
                    "email": resolved_email,
                    "imap_password": resolved_password,
                    "imap_server": resolved_imap_server,
                    "name": resolved_name,
                }
                
                if not self._matches_candidate_filter(candidate, candidate_id, candidate_email):
                    continue

                if candidate.get("email") and candidate.get("imap_password"):
                    candidates.append(candidate)
                else:
                    logger.warning(f"Skipping invalid candidate row: {row}")

            logger.info(f"Fetched {len(candidates)} candidates from database source")
            return candidates
            
        except Exception as e:
            logger.error(f"Error fetching candidates from database: {e}")
            return []

    def _pick(self, row: Dict, keys: List[str], default=None):
        for key in keys:
            if key in row and row.get(key) not in (None, ""):
                return row.get(key)
        return default

    def _matches_candidate_filter(
        self,
        candidate: Dict,
        candidate_id: Optional[int],
        candidate_email: Optional[str],
    ) -> bool:
        if candidate_id is not None:
            current_id = candidate.get("candidate_id")
            try:
                if current_id is None or int(current_id) != int(candidate_id):
                    return False
            except (TypeError, ValueError):
                return False

        if candidate_email:
            current_email = (candidate.get("email") or "").strip().lower()
            if current_email != candidate_email.strip().lower():
                return False

        return True
