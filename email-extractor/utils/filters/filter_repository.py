import logging
import re
from typing import List, Dict, Optional
from utils.api_client import get_api_client

logger = logging.getLogger(__name__)

class FilterRepository:
    """Repository for loading and caching email filters from database"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._filters = None
        self._filters_by_priority = None
        
    def load_filters(self) -> bool:
        """Load filters from API and cache them"""
        try:
            api_client = get_api_client()
            
            # Fetch all filters from API
            response = api_client.get('/api/job-automation-keywords')
            
            if not response:
                self.logger.error("No response from filters API")
                return False
            
            # Filter active filters for email_extractor source
            all_filters = response if isinstance(response, list) else response.get('data', [])
            
            self._filters = [
                f for f in all_filters 
                if f.get('is_active') == 1 and f.get('source') == 'email_extractor'
            ]
            
            # Sort by priority (lower number = higher priority)
            self._filters.sort(key=lambda x: x.get('priority', 999))
            
            # Group by priority for efficient processing
            self._filters_by_priority = {}
            for filter_item in self._filters:
                priority = filter_item.get('priority', 999)
                if priority not in self._filters_by_priority:
                    self._filters_by_priority[priority] = []
                self._filters_by_priority[priority].append(filter_item)
            
            self.logger.info(f"Loaded {len(self._filters)} active filters from database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load filters from API: {str(e)}")
            return False
    
    def get_filters(self) -> List[Dict]:
        """Get all cached filters"""
        if self._filters is None:
            self.load_filters()
        return self._filters or []
    
    def get_filters_by_category(self, category: str) -> List[Dict]:
        """Get filters by category"""
        filters = self.get_filters()
        return [f for f in filters if f.get('category') == category]
    
    def check_email(self, email: str) -> Optional[str]:
        """
        Check email against all filters in priority order
        
        Args:
            email: Email address to check
            
        Returns:
            'allow' or 'block' if matched, None if no match
        """
        if not email:
            return None
        
        filters = self.get_filters()
        email_lower = email.lower()
        
        # Extract parts for different matching strategies
        try:
            local_part, domain = email_lower.split('@', 1)
        except:
            return 'block'  # Invalid email format
        
        # Process filters in priority order
        for filter_item in filters:
            category = filter_item.get('category', '')
            keywords_str = filter_item.get('keywords', '')
            match_type = filter_item.get('match_type', 'contains')
            action = filter_item.get('action', 'block')
            
            if not keywords_str:
                continue
            
            # Split keywords by comma
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
            
            # Determine what to match against based on category
            if 'localpart' in category.lower():
                match_target = local_part
            elif 'domain' in category.lower():
                match_target = domain
            elif 'email' in category.lower() and 'domain' not in category.lower():
                match_target = email_lower
            else:
                # Default: check both email and domain
                match_target = email_lower
            
            # Check each keyword
            for keyword in keywords:
                if self._matches(match_target, keyword, match_type):
                    self.logger.debug(f"Filter matched: {category} - {keyword} -> {action}")
                    return action
        
        return None  # No match
    
    def _matches(self, text: str, pattern: str, match_type: str) -> bool:
        """Check if text matches pattern based on match_type"""
        try:
            if match_type == 'exact':
                return text == pattern.lower()
            elif match_type == 'contains':
                return pattern.lower() in text
            elif match_type == 'regex':
                return bool(re.search(pattern, text, re.IGNORECASE))
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error matching pattern '{pattern}': {str(e)}")
            return False
    
    def get_keyword_lists(self) -> Dict[str, List[str]]:
        """Get keyword lists organized by category for backward compatibility"""
        filters = self.get_filters()
        result = {}
        
        for filter_item in filters:
            category = filter_item.get('category', '')
            keywords_str = filter_item.get('keywords', '')
            
            if keywords_str:
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                result[category] = keywords
        
        return result


# Singleton instance
_filter_repository = None

def get_filter_repository() -> FilterRepository:
    """Get global filter repository instance"""
    global _filter_repository
    if _filter_repository is None:
        _filter_repository = FilterRepository()
        _filter_repository.load_filters()
    return _filter_repository
