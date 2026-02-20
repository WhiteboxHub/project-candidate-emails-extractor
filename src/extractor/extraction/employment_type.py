"""
Employment Type Extractor - Extract employment types from recruiter emails

This module extracts employment/contract types such as:
- W2, C2C, 1099
- Full-time, Part-time, Contract
- Permanent, Temporary
"""

import re
import logging
from typing import Optional, List, Set

logger = logging.getLogger(__name__)


class EmploymentTypeExtractor:
    """Extract employment types from email text"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from src.extractor.filtering.repository import get_filter_repository
        self.filter_repo = get_filter_repository()
        
        # Load patterns from CSV
        self.employment_patterns = self._load_employment_filters()
        
        # Compile all patterns for efficiency
        self.compiled_patterns = {}
        for emp_type, patterns in self.employment_patterns.items():
            self.compiled_patterns[emp_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
            
    def _load_employment_filters(self) -> dict:
        """Load employment patterns from filter repository (CSV)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            patterns_dict = {}
            
            if 'employment_patterns' in keyword_lists:
                patterns_list = keyword_lists['employment_patterns']
                for item in patterns_list:
                    if '|' in item:
                        emp_type, patterns_str = item.split('|', 1)
                        # Patterns within a group are separated by ; in CSV
                        patterns = [p.strip() for p in patterns_str.split(';') if p.strip()]
                        patterns_dict[emp_type.strip()] = patterns
                
                self.logger.info(f"✓ Loaded {len(patterns_dict)} employment types from CSV")
                return patterns_dict
            else:
                self.logger.warning("⚠ employment_patterns not found in CSV - using empty dict")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to load employment filters from CSV: {str(e)}")
            return {}
    
    def extract_employment_types(self, text: str, subject: str = None) -> List[str]:
        """
        Extract all employment types from text
        
        Args:
            text: Email body text
            subject: Email subject line (checked first, most reliable)
            
        Returns:
            List of normalized employment types (e.g., ['W2', 'C2C'])
        """
        try:
            found_types = set()
            
            # 1. Check subject line first (most reliable)
            if subject:
                found_types.update(self._extract_from_text(subject))
            
            # 2. Check first 1000 chars of body (employment type usually mentioned early)
            if text:
                body_preview = text[:1000]
                found_types.update(self._extract_from_text(body_preview))
            
            # Convert to sorted list for consistency
            result = sorted(list(found_types))
            
            if result:
                self.logger.debug(f"✓ Extracted employment types: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting employment types: {str(e)}")
            return []
    
    def _extract_from_text(self, text: str) -> Set[str]:
        """Extract employment types from a text snippet"""
        found = set()
        
        if not text:
            return found
        
        # Check each employment type pattern
        for emp_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    found.add(emp_type)
                    break  # Found this type, no need to check other patterns
        
        # Handle special cases and conflicts
        found = self._resolve_conflicts(found)
        
        return found
    
    def _resolve_conflicts(self, types: Set[str]) -> Set[str]:
        """
        Resolve conflicting employment types
        
        For example:
        - If both 'Full-time' and 'Contract' found, keep both (some jobs offer both)
        - If both 'Permanent' and 'Full-time' found, they're synonyms, keep 'Full-time'
        """
        # No conflicts to resolve for now - keep all found types
        # Jobs can legitimately be "W2/C2C" or "Full-time/Contract"
        return types
    
    def extract_employment_type_string(self, text: str, subject: str = None) -> Optional[str]:
        """
        Extract employment types as a formatted string
        
        Args:
            text: Email body text
            subject: Email subject line
            
        Returns:
            Formatted string like "W2, C2C" or None if no types found
        """
        types = self.extract_employment_types(text, subject)
        
        if types:
            return ', '.join(types)
        
        return None
    
    def has_employment_type(self, text: str, subject: str = None, 
                           target_type: str = None) -> bool:
        """
        Check if text contains a specific employment type
        
        Args:
            text: Email body text
            subject: Email subject line
            target_type: Employment type to check for (e.g., 'W2', 'C2C')
            
        Returns:
            True if target_type is found
        """
        types = self.extract_employment_types(text, subject)
        
        if target_type:
            return target_type in types
        
        return len(types) > 0
