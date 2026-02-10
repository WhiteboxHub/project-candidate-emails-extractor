import logging
import re
import csv
from pathlib import Path
from typing import List, Dict, Optional
from ..connectors.http_api import get_api_client

logger = logging.getLogger(__name__)

class FilterRepository:
    """Repository for loading and caching email filters from database"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._filters = None
        self._filters_by_priority = None
        
    def load_filters(self) -> bool:
        """Load filters from CSV first, fallback to API if CSV not available"""
        # Try CSV first
        csv_path = Path(__file__).parent.parent.parent / "keywords.csv"
        
        if csv_path.exists():
            try:
                return self._load_from_csv(csv_path)
            except Exception as e:
                self.logger.warning(f"Failed to load from CSV: {str(e)}, falling back to database")
        else:
            self.logger.info(f"CSV file not found at {csv_path}, loading from database")
        
        # Fallback to database API
        return self._load_from_database()
    
    def _load_from_csv(self, csv_path: Path) -> bool:
        """Load filters from CSV file"""
        try:
            filters = []
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert is_active to int
                    is_active = int(row.get('is_active', 1))
                    
                    # Only load active filters
                    if is_active == 1:
                        filters.append({
                            'id': int(row.get('id', 0)),
                            'category': row.get('category', ''),
                            'source': row.get('source', ''),
                            'keywords': row.get('keywords', ''),
                            'match_type': row.get('match_type', 'contains'),
                            'action': row.get('action', 'block'),
                            'priority': int(row.get('priority', 999)),
                            'context': row.get('context', ''),
                            'is_active': is_active,
                            'created_at': row.get('created_at', ''),
                            'updated_at': row.get('updated_at', '')
                        })
            
            self._filters = filters
            
            # Sort by priority (lower number = higher priority)
            self._filters.sort(key=lambda x: x.get('priority', 999))
            
            # Group by priority for efficient processing
            self._filters_by_priority = {}
            for filter_item in self._filters:
                priority = filter_item.get('priority', 999)
                if priority not in self._filters_by_priority:
                    self._filters_by_priority[priority] = []
                self._filters_by_priority[priority].append(filter_item)
            
            self.logger.info(f"✓ Loaded {len(self._filters)} active filters from CSV: {csv_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load filters from CSV: {str(e)}")
            raise
    
    def _load_from_database(self) -> bool:
        """Load filters from database API (fallback)"""
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
            
            self.logger.info(f"✓ Loaded {len(self._filters)} active filters from database")
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
            
            # ... (filtering logic for allowed_/blocked_ categories)
            if not (category.startswith('allowed_') or category.startswith('blocked_')):
                continue

            if not keywords_str:
                continue
            
            # ... (target selection)
            cat_lower = category.lower()
            if any(k in cat_lower for k in ['localpart', 'prefix', 'density', 'random']):
                match_target = local_part
            elif 'domain' in cat_lower:
                match_target = domain
            elif 'email' in cat_lower:
                match_target = email_lower
            else:
                match_target = email_lower
            
            # Check each keyword
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
            for keyword in keywords:
                if self._matches(match_target, keyword, match_type):
                    self.logger.debug(f"Filter matched: {category} - {keyword} -> {action}")
                    return action
        
        # Finally, run dynamic heuristic checks for auto-generated/marketing bots
        if self._is_dynamic_junk(local_part, domain):
            self.logger.info(f"Dynamic junk detected: {email}")
            return 'block'

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
        except Exception:
            return False

    def _is_dynamic_junk(self, local_part: str, domain: str) -> bool:
        """Heuristic check for autogenerated/bot emails"""
        
        # NEW HEURISTICS - Added 2026-02-03 to catch junk emails bypassing CSV filters
        
        # 1. UUID pattern (8-4-4-4-12 hex with dashes)
        # Example: d45493db-1629-4a02-affb-11f17d2500f6@reply.linkedin.com
        if re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', local_part):
            self.logger.info(f"Dynamic junk: UUID pattern detected in {local_part}")
            return True
        
        # 2. Pure MD5/SHA hash (32+ consecutive hex chars)
        # Example: bb2137b38d8f4e81beb7fecf9d1785a6@integrisit.com
        if re.match(r'^[a-f0-9]{32,}$', local_part):
            self.logger.info(f"Dynamic junk: MD5/SHA hash detected in {local_part}")
            return True
        
        # 3. Reply tracking patterns (reply-, reply+)
        # Example: reply-z7q4xsbuyvcu7efu4ax2erfhjy.100217@hello.email.hays.com
        # Example: reply+2v4hf0&78gig8&&aa1a5...@mg1.substack.com
        if re.match(r'^reply[-+]', local_part):
            self.logger.info(f"Dynamic junk: Reply tracking pattern detected in {local_part}")
            return True
        
        # 4. Plus-based tracking (email+tracking+hash)
        # Example: publicisgroupe+email+10pv7-14b808175a@talent.icims.eu
        if re.search(r'\+[a-z0-9]+\+[a-f0-9]{10,}', local_part):
            self.logger.info(f"Dynamic junk: Plus-based tracking detected in {local_part}")
            return True
        
        # 5. Desk patterns (screeningdesk.company, hiringdesk.company)
        # Example: screeningdesk.intuit@outlook.com
        if re.match(r'^(screening|hiring|recruiting|talent)desk\.', local_part):
            self.logger.info(f"Dynamic junk: Desk pattern detected in {local_part}")
            return True
        
        # 6. Dots with workday keyword (no.reply.mhfi.workday)
        # Example: no.reply.mhfi.workday@spglobal.com
        if 'workday' in local_part and local_part.count('.') >= 2:
            self.logger.info(f"Dynamic junk: Workday pattern with dots detected in {local_part}")
            return True
        
        # EXISTING HEURISTICS BELOW
        
        # 7. High digit density (5+ digits in localpart)
        digits = sum(c.isdigit() for c in local_part)
        if digits >= 5:
            # Check if it's a mix of letters and digits that looks random
            # (e.g. soseca2658, user19283)
            if any(c.isalpha() for c in local_part):
                return True
        
        # 8. Random hex/hash strings (8+ chars of 0-9a-f)
        # Often used by tracking/reply systems
        if re.search(r'[a-f0-9]{12,}', local_part):
            return True
            
        # 9. Known bot prefixes followed by numbers
        # v-123..., tr-123..., id-123...
        if re.match(r'^(?:v|tr|ref|id|cid|bounce|u|s|ext)-[0-9]+', local_part):
            return True
            
        # 10. Excessive subdomains (bot-like domain nesting)
        if domain.count('.') >= 4:
            return True
            
        # 11. Generic bot pattern: letters + numbers at the end
        if re.match(r'^[a-z]{3,}[0-9]{3,}$', local_part):
            return True

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
                if category not in result:
                    result[category] = []
                result[category].extend(keywords)
        
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
