import re
import joblib
import os
from typing import Dict, List
import logging
from ..filtering.repository import get_filter_repository

logger = logging.getLogger(__name__)

class EmailFilter:
    """Filter and classify emails (recruiter vs junk)"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load filter repository
        self.filter_repo = get_filter_repository()
        
        # Get keyword lists from database
        keyword_lists = self.filter_repo.get_keyword_lists()
        
        # Extract recruiter and anti-recruiter keywords
        self.recruiter_keywords = keyword_lists.get('recruiter_keywords', [])
        self.anti_recruiter_keywords = keyword_lists.get('anti_recruiter_keywords', [])
        
        # Load ML classifier if enabled
        self.use_ml = config.get('filters', {}).get('use_ml_classifier', False)
        if self.use_ml:
            self._load_ml_model()
    
    def _load_ml_model(self):
        """Load pre-trained ML classifier"""
        try:
            model_dir = self.config.get('filters', {}).get('ml_model_dir', '../models')
            classifier_path = os.path.join(model_dir, 'classifier.pkl')
            vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
            
            if os.path.exists(classifier_path) and os.path.exists(vectorizer_path):
                self.classifier = joblib.load(classifier_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.logger.info("ML classifier loaded successfully")
            else:
                self.logger.warning(f"ML model files not found in {model_dir}. ML filtering disabled.")
                self.use_ml = False
        except Exception as e:
            self.logger.error(f"Error loading ML model: {str(e)}")
            self.use_ml = False
    
    def _extract_clean_email(self, from_header: str) -> str:
        """Extract email address from From header"""
        if not from_header:
            return ""
        
        email_match = re.search(
            r'(?:<|\(|^)([\w\.-]+@[\w\.-]+)(?:>|\)|$)',
            from_header,
            re.IGNORECASE
        )
        return email_match.group(1).lower() if email_match else ""
    
    def is_junk_email(self, from_header: str) -> bool:
        """Check if email is junk/automated/system using database filters"""
        email = self._extract_clean_email(from_header)
        
        if not email or '@' not in email:
            return True
        
        # Check against database filters
        action = self.filter_repo.check_email(email)
        
        if action == 'block':
            self.logger.debug(f"Blocked by filter: {email}")
            return True
        elif action == 'allow':
            self.logger.debug(f"Allowed by filter: {email}")
            return False
        
        # No match - default to not junk
        return False
    
    def is_recruiter_email(self, subject: str, body: str, from_email: str) -> bool:
        """Classify if email is from a recruiter using smart keyword matching"""
        # First check if it's junk
        if self.is_junk_email(from_email):
            return False
        
        # If ML classifier available, use it
        if self.use_ml and self.classifier and self.vectorizer:
            try:
                features = self.vectorizer.transform([f"{subject} {body} {from_email}"])
                prediction = self.classifier.predict(features)[0]
                return prediction == 1
            except Exception as e:
                self.logger.error(f"ML classification error: {str(e)}")
                return False
        
        # Smart keyword matching with separate subject/body analysis
        subject_lower = subject.lower()
        body_lower = body.lower()
        text = f"{subject} {body}".lower()
        
        # Check anti-keywords (marketing indicators) - raised threshold
        anti_keyword_count = sum(1 for kw in self.anti_recruiter_keywords if kw in text)
        if anti_keyword_count >= 4:  # Raised from 2 to 4
            self.logger.debug(f"Filtered: Marketing/Newsletter detected ({anti_keyword_count} anti-keywords)")
            return False
        
        # Count recruiter keywords in subject (more reliable)
        subject_keyword_count = sum(1 for kw in self.recruiter_keywords if kw in subject_lower)
        
        # Count recruiter keywords in body
        body_keyword_count = sum(1 for kw in self.recruiter_keywords if kw in body_lower)
        
        # Smart matching rules:
        # 1. Subject has 1+ keyword = likely recruiter
        if subject_keyword_count >= 1:
            return True
        
        # 2. Body has 2+ keywords = likely recruiter
        if body_keyword_count >= 2:
            return True
        
        # 3. At least 1 keyword total (lowered from 2)
        if subject_keyword_count + body_keyword_count >= 1:
            return True
        
        return False
    
    def is_calendar_invite(self, email_message) -> bool:
        """Check if email is a calendar invite"""
        try:
            for part in email_message.walk():
                if part.get_content_type() == "text/calendar":
                    return True
            return False
        except:
            return False
    
    def filter_emails(self, emails: List[Dict], cleaner) -> tuple:
        """
        Filter email list to keep only recruiter/calendar emails
        
        Args:
            emails: List of email dictionaries
            cleaner: EmailCleaner instance for body extraction
            
        Returns:
            Tuple of (filtered_emails, filter_stats)
            - filtered_emails: List of emails that passed filtering
            - filter_stats: Dict with filtering statistics
        """
        filtered = []
        junk_count = 0
        not_recruiter_count = 0
        calendar_count = 0
        
        for email_data in emails:
            try:
                msg = email_data['message']
                from_header = msg.get('From', '')
                subject = msg.get('Subject', '')
                
                # Always include calendar invites
                if self.config.get('processing', {}).get('calendar_invites', {}).get('process', True):
                    if self.is_calendar_invite(msg):
                        self.logger.debug(f"Including calendar invite from {from_header}")
                        calendar_count += 1
                        filtered.append(email_data)
                        continue
                
                # Skip junk emails
                if self.is_junk_email(from_header):
                    junk_count += 1
                    continue
                
                # Extract and clean body
                body = cleaner.extract_body(msg)
                
                # Check if recruiter email
                if self.is_recruiter_email(subject, body, from_header):
                    email_data['clean_body'] = body
                    filtered.append(email_data)
                else:
                    not_recruiter_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error filtering email: {str(e)}")
                continue
        
        # Build filter statistics
        filter_stats = {
            'total': len(emails),
            'passed': len(filtered),
            'junk': junk_count,
            'not_recruiter': not_recruiter_count,
            'calendar_invites': calendar_count
        }
        
        self.logger.info(f"Filtered {len(filtered)} emails from {len(emails)} total (Junk: {junk_count}, Not recruiter: {not_recruiter_count}, Calendar: {calendar_count})")
        return filtered, filter_stats
