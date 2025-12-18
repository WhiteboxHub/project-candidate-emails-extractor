import re
import phonenumbers
from typing import Optional
import logging

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
        
        # Personal/consumer email domains to block
        self.personal_domains = {
            'gmail.com', 'googlemail.com', 'yahoo.com', 'yahoo.co.uk', 'yahoo.in',
            'outlook.com', 'hotmail.com', 'live.com', 'msn.com',
            'icloud.com', 'me.com', 'mac.com',
            'aol.com', 'protonmail.com', 'proton.me', 'pm.me',
            'mail.com', 'zoho.com', 'yandex.com', 'yandex.ru',
            'gmx.com', 'gmx.net', 'tutanota.com', 'tutanota.de',
            'fastmail.com', 'hushmail.com', 'mailfence.com'
        }
    
    def _is_personal_email(self, email: str) -> bool:
        """Check if email is from a personal/consumer domain (Gmail, Yahoo, etc.)"""
        if not email or '@' not in email:
            return True
        
        try:
            domain = email.split('@')[1].lower()
            return domain in self.personal_domains
        except:
            return True
    
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
            
            # Filter out file extensions in local part (image001.png, document.pdf, etc.)
            file_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.pdf', '.doc', 
                             '.docx', '.xls', '.xlsx', '.zip', '.rar', '.txt', '.csv']
            if any(local_part.lower().endswith(ext) for ext in file_extensions):
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
            
            # Filter out common automated emails
            blacklist_prefixes = ['noreply', 'no-reply', 'donotreply', 'info', 'support', 
                                 'admin', 'notifications', 'newsletter', 'mailer']
            
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
                
                # Skip blacklisted prefixes
                if any(email_lower.startswith(prefix) for prefix in blacklist_prefixes):
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
