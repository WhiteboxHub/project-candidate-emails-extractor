import re
import joblib
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class EmailFilter:
    """Filter and classify emails (recruiter vs junk)"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize filter lists
        self._initialize_filter_lists()
        self.personal_domains = set(config.get('filters', {}).get('personal_domains', []))
        self.service_domains = set(config.get('filters', {}).get('service_domains', []))
        
        # Exact email blacklist
        self.exact_email_blacklist = {
            "teamzoom@zoom.us", "ops@cluso.com", "recruiter@softquip.com",
            "requirements@gainamerica.net", "assistant@glider.ai",
            "echosign@echosign.com", "aggregated@lensa.com",
            "e.linkedin.com", "hello@v3.idibu.com"
        }
        
        # Load ML classifier if enabled
        self.use_ml = config.get('filters', {}).get('use_ml_classifier', False)
        if self.use_ml:
            self._load_ml_model()
    
    def _initialize_filter_lists(self):
        """Initialize comprehensive blacklist from rules"""
        # Personal email domains
        self.domain_blacklist_patterns = [
            r'gmail\.com$', r'yahoo\.com$', r'outlook\.com$', r'icloud\.com$',
            r'protonmail\.com$', r'zoho\.com$', r'mail\.com$', r'aol\.com$',
            r'yandex\.com$', r'rambler\.ru$', r'hotmail\.com$', r'live\.com$',
            r'msn\.com$', r'student\..*', r'alumni\..*', r'academy\..*',
            r'school\..*', r'college\..*', r'university\..*', r'edu\..*',
            r'test\.com$', r'example\.com$', r'demo\.com$', r'sample\.com$',
            r'fake\.com$', r'jobboard\.com$', r'newsletter\.com$',
            r'alerts\.company\.com$',
            # Marketing/Newsletter domains
            r'neo4j\.com$', r'oreilly\.com$', r'medium\.com$', r'canva\.com$',
            r'google\.com$', r'anthropic\.com$', r'replit\.com$', r'udemy\.com$',
            # Job boards/aggregators (only spam ones, allow real job boards)
            r'jobleads\.com$', r'lensa\.com$', r'jobcase\.com$',
            r'postjobfree\.com$', r'ihire\.com$',
            # Allow: dice.com, ziprecruiter.com, monster.com, careerbuilder.com, glassdoor.com
            # LinkedIn
            r'linkedin\.com$', r'e\.linkedin\.com$', r'em\.linkedin\.com$',
            # ATS/Job platforms (only spam/automated ones)
            r'aiapply\.co$', r'directlyapply\.com$', r'jobs2web\.com$',
            # Allow: jobvite, smartrecruiters, workday, lever, greenhouse (real recruiter ATS)
            # SaaS/Product companies
            r'fireflies\.ai$', r'zapier\.com$', r'doordash\.com$',
            r'lyrahealth\.com$', r'brighthire\.ai$', r'trucksmarter\.com$',
            r'labelbox\.com$', r'mywisely\.com$', r'fedex\.com$',
            # Google Voice SMS
            r'txt\.voice\.google\.com$'
        ]
        
        # Always blacklist (automated/system emails)
        self.always_blacklist_patterns = [
            # Spam/temp mail
            r'spam\.com$', r'trashmail\.com$', r'temp-mail\.org$', r'mailinator\.com$',
            # Your own company (internal)
            r'innova-path\.com$', r'whitebox-learning\.com$',
            # LinkedIn specific
            r'jobs-listings@linkedin', r'newsletters-noreply@linkedin',
            r'inmail-hit-reply@linkedin', r'hit-reply@linkedin',
            r'messages-noreply@linkedin', r'jobalerts-noreply@linkedin',
            r'editors-noreply@linkedin', r'groups-noreply@linkedin',
            r'invitations@linkedin',
            # Indeed
            r'indeedapply@indeed', r'noreply@indeed',
            # Automated prefixes
            r'^donotreply@', r'^no-reply@', r'^noreply@', r'^do_not_reply@',
            r'^do-not-reply@', r'^noreplies@', r'^notification@',
            r'^autoresponder@', r'^tracking@', r'^calendar-notification@',
            r'^echosign@', r'^mailer@', r'^aggregated@',
            # Generic automated patterns
            r'no-reply@.*', r'do-not-reply@.*', r'notifications@.*',
            r'jobs@.*', r'info@.*', r'noreply@.*', r'newsletter@.*',
            r'alerts@.*', r'update@.*', r'donotreply@.*', r'support@.*',
            r'admin@.*', r'system@.*', r'bounce@.*', r'postmaster@.*',
            r'auto@.*', r'digest@.*', r'bulk@.*', r'mail@.*', r'email@.*',
            r'news@.*', r'press@.*', r'media@.*', r'marketing@.*',
            # Microsoft Exchange
            r'microsoftexchange',
            # Test/Invalid
            r'test@.*', r'demo@.*', r'@test\.', r'@demo\.'
        ]
        
        # Compile patterns
        self.domain_blacklist_regex = [re.compile(p, re.IGNORECASE) for p in self.domain_blacklist_patterns]
        self.always_blacklist_regex = [re.compile(p, re.IGNORECASE) for p in self.always_blacklist_patterns]
        
        # Recruiter keywords for content analysis (must have at least 2)
        self.recruiter_keywords = [
            'recruit', 'hiring', 'opportunity', 'position', 'talent',
            'career', 'role', 'interview', 'vendor', 'staffing',
            'consultant', 'placement', 'headhunter', 'agency',
            'candidate', 'resume', 'cv', 'portfolio',
            'contract', 'full time', 'full-time', 'contract to hire',
            'w2', 'corp to corp', 'c2c', '1099'
        ]
        
        # Anti-keywords (if present, likely NOT a recruiter)
        self.anti_recruiter_keywords = [
            'unsubscribe', 'webinar', 'newsletter', 'subscription',
            'courses', 'training', 'certification', 'learn more',
            'free trial', 'sign up', 'register now', 'limited time',
            'download', 'ebook', 'whitepaper', 'case study'
        ]
        
        self.junk_pattern = re.compile(
            r'^(no-?reply|auto(responder|bot)|.*alert.*|.*noreply.*|.*notifications?)@',
            re.IGNORECASE
        )
    
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
        """Check if email is junk/automated/system using rules"""
        email = self._extract_clean_email(from_header)
        
        if not email or '@' not in email:
            return True
        
        # Check always blacklist patterns (highest priority)
        for pattern in self.always_blacklist_regex:
            if pattern.search(email):
                self.logger.debug(f"Blacklisted (always): {email}")
                return True
        
        # Extract domain
        try:
            domain = email.split('@', 1)[1]
        except:
            return True
        
        # Check domain blacklist patterns
        for pattern in self.domain_blacklist_regex:
            if pattern.search(domain):
                self.logger.debug(f"Blacklisted domain: {domain}")
                return True
        
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
    
    def filter_emails(self, emails: List[Dict], cleaner) -> List[Dict]:
        """
        Filter email list to keep only recruiter/calendar emails
        
        Args:
            emails: List of email dictionaries
            cleaner: EmailCleaner instance for body extraction
            
        Returns:
            Filtered list of emails
        """
        filtered = []
        
        for email_data in emails:
            try:
                msg = email_data['message']
                from_header = msg.get('From', '')
                subject = msg.get('Subject', '')
                
                # Always include calendar invites
                if self.config.get('processing', {}).get('calendar_invites', {}).get('process', True):
                    if self.is_calendar_invite(msg):
                        self.logger.debug(f"Including calendar invite from {from_header}")
                        filtered.append(email_data)
                        continue
                
                # Skip junk emails
                if self.is_junk_email(from_header):
                    continue
                
                # Extract and clean body
                body = cleaner.extract_body(msg)
                
                # Check if recruiter email
                if self.is_recruiter_email(subject, body, from_header):
                    email_data['clean_body'] = body
                    filtered.append(email_data)
                    
            except Exception as e:
                self.logger.error(f"Error filtering email: {str(e)}")
                continue
        
        self.logger.info(f"Filtered {len(filtered)} emails from {len(emails)} total")
        return filtered
