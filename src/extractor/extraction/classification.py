from typing import Tuple, List, Optional
import logging
import re
from ..filtering.repository import get_filter_repository

logger = logging.getLogger(__name__)

class RecruiterClassifier:
    """
    Classifies whether a contact is likely a recruiter/talent acquisition professional
    based on job title, email context, and other signals using CONFIGURABLE keywords.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.filter_repo = get_filter_repository()
        
        # Load keywords from CSV via repository
        self.strong_indicators = self._load_keywords('recruiter_title_strong')
        self.moderate_indicators = self._load_keywords('recruiter_title_moderate')
        self.weak_indicators = self._load_keywords('recruiter_title_weak')
        self.negative_indicators = self._load_keywords('recruiter_title_negative')
        self.context_indicators = self._load_keywords_list('recruiter_context_positive') # List for regex/patterns? No, contains.
        
        self.logger.info(f"RecruiterClassifier initialized with {len(self.strong_indicators)} strong, "
                         f"{len(self.moderate_indicators)} moderate, {len(self.negative_indicators)} negative indicators.")

    def _load_keywords(self, category_key: str) -> set:
        """Load keywords for a category from the repository as a set"""
        try:
            lists = self.filter_repo.get_keyword_lists()
            if category_key in lists:
                return {kw.lower().strip() for kw in lists[category_key]}
            self.logger.warning(f"Keyword category '{category_key}' not found in CSV")
            return set()
        except Exception as e:
            self.logger.error(f"Failed to load keywords for {category_key}: {e}")
            return set()

    def _load_keywords_list(self, category_key: str) -> list:
        """Load keywords for a category from the repository as a list (for order preservation if needed)"""
        try:
            lists = self.filter_repo.get_keyword_lists()
            if category_key in lists:
                return [kw.lower().strip() for kw in lists[category_key]]
            return []
        except Exception as e:
            self.logger.error(f"Failed to load keywords list for {category_key}: {e}")
            return []

    def is_recruiter(self, title: Optional[str], context: Optional[str] = None) -> Tuple[bool, float, str]:
        """
        Determine if the contact is a recruiter.
        
        Args:
            title: Job title string (from signature or other source)
            context: Additional context path (e.g. email body snippet)
            
        Returns:
            (is_recruiter, score, reason)
        """
        score = 0.0
        reason = "No title found"
        
        if not title:
            # Fallback to context if no title
            if context:
                return self._analyze_context(context)
            return False, 0.0, reason
            
        title_lower = title.lower()
        
        # Check negative indicators first
        for indicator in self.negative_indicators:
            if indicator in title_lower:
                return False, 0.0, f"Negative indicator found: {indicator}"
        
        # Check strong indicators
        for indicator in self.strong_indicators:
            if indicator in title_lower:
                return True, 1.0, f"Strong indicator found: {indicator}"
                
        # Check moderate indicators
        for indicator in self.moderate_indicators:
            if indicator in title_lower:
                score += 0.6
                reason = f"Moderate indicator found: {indicator}"
                
        # Check weak indicators
        if score == 0.0:
            for indicator in self.weak_indicators:
                if indicator in title_lower:
                    score += 0.3
                    reason = f"Weak indicator found: {indicator}"
        
        # Threshold for "True"
        is_recruiter = score >= 0.5
        return is_recruiter, score, reason

    def _analyze_context(self, context: str) -> Tuple[bool, float, str]:
        """Analyze email body/context for recruiter signals if no title exists."""
        if not context:
            return False, 0.0, "No context"
            
        context_lower = context.lower()
        
        # Check context indicators
        # Note: CSV provides flat list, so we treat them all as strong positive for now or default weight
        for indicator in self.context_indicators:
             if indicator in context_lower:
                 return True, 0.8, f"Context matches: {indicator}"
                
        return False, 0.0, "No context signals"
