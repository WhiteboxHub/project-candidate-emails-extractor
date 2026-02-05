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
from utils.filters.filter_repository import get_filter_repository

logger = logging.getLogger(__name__)


class EmploymentTypeExtractor:
    """Extract employment types from email text"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.filter_repo = get_filter_repository()
        
        # Employment type patterns with their normalized forms
        self.employment_patterns = {}
        self.compiled_patterns = {}
        
        self._load_employment_filters()
        
    def _load_employment_filters(self):
        """Load employment type patterns from CSV"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            
            if 'employment_patterns' in keyword_lists:
                patterns_raw = keyword_lists['employment_patterns']
                
                # Input format (format: Type|Pattern1;Pattern2)
                # Note: patterns_raw is a list because get_keyword_lists splits segments by comma
                for mapping in patterns_raw:
                    if '|' in mapping:
                        emp_type, regexes_str = mapping.split('|', 1)
                        # Patterns are separated by semicolon inside the mapping
                        regexes = [r.strip() for r in regexes_str.split(';') if r.strip()]
                        
                        self.employment_patterns[emp_type] = regexes
                        self.compiled_patterns[emp_type] = [
                            re.compile(r, re.IGNORECASE) for r in regexes
                        ]
                
                self.logger.info(f"✓ Loaded {len(self.employment_patterns)} employment types from CSV")
            else:
                self.logger.warning("⚠ employment_patterns not found in CSV")
                # Fallback to empty if not found (though CSV update just added it)
                self.employment_patterns = {}
                self.compiled_patterns = {}
                
        except Exception as e:
            self.logger.error(f"Failed to load employment filters from CSV: {str(e)}")
            self.employment_patterns = {}
            self.compiled_patterns = {}
    
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
