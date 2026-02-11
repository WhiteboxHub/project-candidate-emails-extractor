"""
Raw Positions Persistence - Save extracted job positions to database via API
"""

import logging
from typing import List, Dict, Optional
from ..connectors.http_api import APIClient

logger = logging.getLogger(__name__)


class RawPositionsUtil:
    """Utility for saving raw position data via API"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
    
    def save_positions_bulk(self, positions: List[Dict], candidate_id: int = None) -> int:
        """
        Save raw positions in bulk via API
        
        Args:
            positions: List of position dictionaries with extracted data
            candidate_id: Optional candidate ID to associate positions with
            
        Returns:
            Number of positions saved
        """
        try:
            if not positions:
                self.logger.warning("No positions to save")
                return 0
            
            # Prepare positions for API
            prepared_positions = []
            for pos in positions:
                position_data = {
                    'job_position': pos.get('job_position'),
                    'location': pos.get('location'),
                    'employment_type': pos.get('employment_type'),
                    'zip_code': pos.get('zip_code'),
                    'company': pos.get('company'),
                    'email': pos.get('email'),
                    'source_email': pos.get('source'),
                    'candidate_id': candidate_id,
                    'raw_data': {
                        'subject': pos.get('extracted_from_subject'),
                        'extraction_source': pos.get('extraction_source'),
                        'name': pos.get('name'),
                        'phone': pos.get('phone'),
                        'linkedin_id': pos.get('linkedin_id')
                    }
                }
                
                # Only save if we have at least job_position or location
                if position_data['job_position'] or position_data['location']:
                    prepared_positions.append(position_data)
            
            if not prepared_positions:
                self.logger.warning("No valid positions to save after filtering")
                return 0
            
            # POST to bulk endpoint
            response = self.api_client.post('/api/raw-positions/bulk', {
                'positions': prepared_positions
            })
            
            saved_count = response.get('saved', 0) if response else 0
            self.logger.info(f"✓ Saved {saved_count} raw positions via API")
            
            return saved_count
            
        except Exception as e:
            self.logger.error(f"Error saving raw positions: {str(e)}")
            return 0
    
    def save_position(self, position: Dict, candidate_id: int = None) -> bool:
        """
        Save a single raw position
        
        Args:
            position: Position dictionary with extracted data
            candidate_id: Optional candidate ID
            
        Returns:
            True if saved successfully
        """
        try:
            result = self.save_positions_bulk([position], candidate_id)
            return result > 0
        except Exception as e:
            self.logger.error(f"Error saving position: {str(e)}")
            return False
