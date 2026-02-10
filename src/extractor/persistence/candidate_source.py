from typing import List, Dict, Optional
import logging
from utils.api_client import APIClient

logger = logging.getLogger(__name__)

class CandidateUtil:
    """
    Utility for fetching candidate marketing accounts via API
    """
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        # Session-based cache to reduce API calls
        self._cached_candidates: Optional[List[Dict]] = None
        self._cache_loaded = False
    
    def get_active_candidates(self) -> List[Dict]:
        """
        Fetch ALL candidates with email credentials via API
        Uses session-based caching to reduce API calls.
        
        Returns:
            List of candidate dictionaries with email and imap_password
        """
        # Return cached data if already loaded
        if self._cache_loaded and self._cached_candidates is not None:
            self.logger.info(f"Returning {len(self._cached_candidates)} candidates from cache")
            return self._cached_candidates
        
        try:
            # GET /candidate/marketing with pagination
            # Fetch all pages
            all_candidates = []
            page = 1
            limit = 100
            
            while True:
                response = self.api_client.get(
                    '/api/candidate/marketing',
                    params={'page': page, 'limit': limit}
                )
                
                # Handle different response formats
                if isinstance(response, dict):
                    candidates = response.get('data', response.get('items', []))
                    total = response.get('total', 0)
                else:
                    candidates = response
                    total = len(candidates)
                
                if not candidates:
                    break
                
                # Filter for candidates with email credentials
                for candidate in candidates:
                    if candidate.get('email') and candidate.get('imap_password'):
                        # Get full_name from nested candidate object if available
                        full_name = candidate.get('email')
                        if candidate.get('candidate') and isinstance(candidate.get('candidate'), dict):
                            full_name = candidate.get('candidate').get('full_name') or full_name

                        # Map API response to expected format
                        all_candidates.append({
                            'id': candidate.get('id'),
                            'candidate_id': candidate.get('candidate_id'),
                            'email': candidate.get('email'),
                            'imap_password': candidate.get('imap_password'),
                            'status': candidate.get('status'),
                            'priority': candidate.get('priority', 100),
                            'name': full_name
                        })
                
                # Check if we've fetched all pages
                if len(candidates) < limit:
                    break
                
                page += 1
            
            self.logger.info(f"Fetched {len(all_candidates)} candidates with email credentials from API")
            
            # Cache the results for this session
            self._cached_candidates = all_candidates
            self._cache_loaded = True
            
            return all_candidates
            
        except Exception as e:
            self.logger.error(f"API error fetching candidates: {str(e)}")
            return []
    
    def get_candidate_by_id(self, candidate_id: int) -> Optional[Dict]:
        """
        Fetch a specific candidate by ID via API
        
        Args:
            candidate_id: Candidate marketing ID
            
        Returns:
            Candidate dictionary or None
        """
        try:
            response = self.api_client.get(f'/api/candidate/marketing/{candidate_id}')
            
            if response:
                candidate = response
                # Map to expected format
                return {
                    'id': candidate.get('id'),
                    'candidate_id': candidate.get('candidate_id'),
                    'email': candidate.get('email'),
                    'imap_password': candidate.get('imap_password'),
                    'status': candidate.get('status'),
                    'name': candidate.get('email', 'Unknown')
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"API error fetching candidate {candidate_id}: {str(e)}")
            return None
    
    def clear_cache(self):
        """
        Clear the session cache to force a fresh fetch on next call.
        Useful for testing or if candidate data needs to be refreshed.
        """
        self._cached_candidates = None
        self._cache_loaded = False
        self.logger.info("Candidate cache cleared")
