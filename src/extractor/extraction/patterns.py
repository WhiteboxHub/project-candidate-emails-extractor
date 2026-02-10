import re
import phonenumbers
from typing import Optional
import logging
from ..filtering.repository import get_filter_repository

logger = logging.getLogger(__name__)

class RegexExtractor:
    """Extract contact information using regex patterns"""
    
    def __init__(self):
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
            re.IGNORECASE
        )
        
        self.linkedin_pattern = re.compile(
            r'https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/([a-zA-Z0-9\-_]+)',
            re.IGNORECASE
        )
        
        self.logger = logging.getLogger(__name__)
        self.filter_repo = get_filter_repository()
        
        # Load email blacklist prefixes from CSV (fallback to hardcoded)
        self.blacklist_prefixes = self._load_blacklist_prefixes()
        
        # Load file extensions for CID filtering from CSV (fallback to hardcoded)
        self.file_extensions = self._load_file_extensions()
    
    def _load_blacklist_prefixes(self) -> list:
        """Load email blacklist prefixes from filter repository (CSV only - no fallback)"""
        try:
            # Get blocked_automated_prefix and blocked_generic_prefix from CSV
            keyword_lists = self.filter_repo.get_keyword_lists()
            
            prefixes = []
            for category in ['blocked_automated_prefix', 'blocked_generic_prefix']:
                if category in keyword_lists:
                    keywords = keyword_lists[category]
                    # Extract keywords, handling regex patterns
                    for kw in keywords:
                        # Remove regex anchors (^) if present
                        clean_kw = kw.replace('^', '').replace('@', '')
                        if clean_kw:  
                            prefixes.append(clean_kw.lower())
            
            if prefixes:
                self.logger.info(f"✓ Loaded {len(prefixes)} email blacklist prefixes from CSV")
            else:
                self.logger.error("⚠ No email blacklist prefixes found in CSV - using empty list")
            
            return prefixes
                
        except Exception as e:
            self.logger.error(f"Failed to load blacklist prefixes from CSV: {str(e)} - using empty list")
            return []  # No hardcoded fallback - return empty list
    
    def _load_file_extensions(self) -> list:
        """Load file extensions for CID filtering from filter repository (CSV only - no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            
            if 'blocked_file_extension' in keyword_lists:
                extensions = keyword_lists['blocked_file_extension']
                self.logger.info(f"✓ Loaded {len(extensions)} file extensions from CSV")
                return extensions
            else:
                self.logger.error("⚠ blocked_file_extension not found in CSV - using empty list")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to load file extensions from CSV: {str(e)} - using empty list")
            return []  # No hardcoded fallback - return empty list
    
    def _is_personal_email(self, email: str) -> bool:
        """Check if email is from a personal/consumer domain using database filters"""
        if not email or '@' not in email:
            return True
        
        action = self.filter_repo.check_email(email)
        return action == 'block'
    
    def _is_valid_email_format(self, email: str) -> bool:
        """Validate email format and filter out CID references and fake emails
        
        Filters out:
        - Image CIDs: image001.png@01dc6e1f.089ef930
        - File references: document.pdf@server.com
        - Invalid formats with numbers/hex in domain
        """
        if not email or '@' not in email:
            return False
        
        try:
            local_part, domain = email.split('@', 1)
            
            # Filter out file extensions in local part (loaded from CSV)
            if any(local_part.lower().endswith(ext) for ext in self.file_extensions):
                self.logger.debug(f"Filtered out image/file CID: {email}")
                return False
            
            # Filter out hex-like domains (CID references like @01dc6e1f.089ef930)
            # These typically have only numbers and hex characters
            domain_parts = domain.split('.')
            if all(all(c in '0123456789abcdef' for c in part.lower()) for part in domain_parts):
                self.logger.debug(f"Filtered out CID reference: {email}")
                return False
            
            # Domain should have at least one alphabetic character
            if not any(c.isalpha() for c in domain):
                self.logger.debug(f"Filtered out invalid domain: {email}")
                return False
            
            # Domain should not be too short (minimum realistic: x.co = 4 chars)
            if len(domain) < 4:
                return False
            
            return True
            
        except:
            return False
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text, excluding personal emails (Gmail, Yahoo, etc.)"""
        try:
            # Get all emails
            emails = self.email_pattern.findall(text)
            if not emails:
                return None
            
            
            valid_emails = []
            for email in emails:
                email_lower = email.lower()
                
                # FIRST: Validate email format (filter out CID references)
                if not self._is_valid_email_format(email_lower):
                    continue
                
                # Skip personal email domains (Gmail, Yahoo, etc.)
                if self._is_personal_email(email_lower):
                    self.logger.debug(f"Skipped personal email: {email_lower}")
                    continue
                
                # Skip blacklisted prefixes (loaded from CSV)
                if any(email_lower.startswith(prefix) for prefix in self.blacklist_prefixes):
                    continue
                    
                valid_emails.append(email_lower)
            
            # Return first valid email
            if valid_emails:
                self.logger.debug(f"Extracted business email: {valid_emails[0]}")
                return valid_emails[0]
            
            self.logger.debug("No valid business emails found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting email: {str(e)}")
            return None
    
    def extract_all_emails(self, text: str) -> list:
        """Extract all email addresses from text"""
        try:
            return [email.lower() for email in self.email_pattern.findall(text)]
        except Exception as e:
            self.logger.error(f"Error extracting emails: {str(e)}")
            return []
    
    def extract_phone(self, text: str, region: str = 'US') -> Optional[str]:
        """Extract and format phone number with fallback regions"""
        try:
            # Try US first
            for match in phonenumbers.PhoneNumberMatcher(text, region):
                phone_number = match.number
                # Validate it's a reasonable phone number
                if phonenumbers.is_valid_number(phone_number):
                    return phonenumbers.format_number(
                        phone_number, 
                        phonenumbers.PhoneNumberFormat.E164
                    )
            
            # Fallback: Try without region
            for match in phonenumbers.PhoneNumberMatcher(text, None):
                phone_number = match.number
                if phonenumbers.is_valid_number(phone_number):
                    return phonenumbers.format_number(
                        phone_number, 
                        phonenumbers.PhoneNumberFormat.E164
                    )
                    
        except Exception as e:
            self.logger.error(f"Error extracting phone: {str(e)}")
        return None
    
    def extract_linkedin_id(self, text: str) -> Optional[str]:
        """Extract LinkedIn profile ID from URL"""
        try:
            match = self.linkedin_pattern.search(text)
            return match.group(1) if match else None
        except Exception as e:
            self.logger.error(f"Error extracting LinkedIn: {str(e)}")
            return None
    
    def extract_linkedin_url(self, text: str) -> Optional[str]:
        """Extract full LinkedIn URL"""
        try:
            match = self.linkedin_pattern.search(text)
            return match.group(0) if match else None
        except Exception as e:
            self.logger.error(f"Error extracting LinkedIn URL: {str(e)}")
            return None
