from typing import Dict, Optional, List
import logging
from email.utils import parseaddr
import re

from .regex_util import RegexExtractor
from .ner_util import SpacyNERExtractor
from .gliner_util import GLiNERExtractor

logger = logging.getLogger(__name__)

class ContactExtractor: 
    """
    Unified contact extraction with config-driven fallback chain
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize extractors based on config
        enabled_methods = config.get('extraction', {}).get('enabled_methods', ['regex', 'spacy'])
        
        self.regex_extractor = RegexExtractor()
        self.spacy_extractor = None
        self.gliner_extractor = None
        
        if 'spacy' in enabled_methods:
            try:
                spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
                self.spacy_extractor = SpacyNERExtractor(model=spacy_model)
                self.logger.info("Spacy NER extractor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to load Spacy: {str(e)}")
        
        if 'gliner' in enabled_methods:
            try:
                self.gliner_extractor = GLiNERExtractor(config)
                self.logger.info("GLiNER extractor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to load GLiNER: {str(e)}")
    
    def extract_contacts(self, email_message, clean_body: str, source_email: str) -> List[Dict]:
        """
        Extract contact information with fallback chain - returns LIST of contacts
        
        Args:
            email_message: Email message object
            clean_body: Cleaned email body text
            source_email: Source candidate email
            
        Returns:
            List of dictionaries with extracted contact fields (can be multiple contacts per email)
        """
        contacts = []
        seen_emails = set()  # Track to avoid duplicates
        
        try:
            # Get configuration settings
            extract_multiple = self.config.get('extraction', {}).get('extract_multiple_contacts', True)
            block_gmail = self.config.get('extraction', {}).get('block_gmail', True)
            
            # Get raw HTML body for vendor span extraction
            raw_html = self._get_html_body(email_message)
            
            # PRIORITY 1: Extract vendor info from HTML span tags (e.g., <span>Name - Company</span>)
            vendor_info = {'name': None, 'company': None}
            if raw_html and self.spacy_extractor:
                vendor_info = self.spacy_extractor.extract_vendor_from_span(raw_html)
            
            # Extract all potential email addresses based on priority
            # Priority order: Reply-To > Sender > From > CC/BCC > Calendar > Body
            all_emails = []
            
            # 1. Reply-To (highest priority - direct contact)
            reply_to_email = self._extract_reply_to_email(email_message)
            if reply_to_email and not self._is_gmail_address(reply_to_email, block_gmail):
                all_emails.append(('reply-to', reply_to_email))
            
            # 2. Sender header (explicit sender)
            sender_email = self._extract_sender_email(email_message)
            if sender_email and not self._is_gmail_address(sender_email, block_gmail):
                all_emails.append(('sender', sender_email))
            
            # 3. From header (message originator)
            from_email = self._extract_from_header(email_message)
            if from_email and not self._is_gmail_address(from_email, block_gmail):
                all_emails.append(('from', from_email))
            
            # 4. CC/BCC headers (additional contacts)
            cc_emails = self._extract_cc_bcc_emails(email_message)
            for cc_email in cc_emails:
                if not self._is_gmail_address(cc_email, block_gmail):
                    all_emails.append(('cc', cc_email))
            
            # 5. Calendar invite emails
            calendar_emails = self._extract_calendar_email(email_message)
            if calendar_emails:
                for cal_email in calendar_emails:
                    if not self._is_gmail_address(cal_email, block_gmail):
                        all_emails.append(('calendar', cal_email))
            
            # 6. Body extraction (lowest priority)
            body_email = self._extract_field('email', clean_body, email_message)
            if body_email and not self._is_gmail_address(body_email, block_gmail):
                all_emails.append(('body', body_email))
            
            # If extract_multiple is False, only use the highest priority email
            if not extract_multiple and all_emails:
                all_emails = [all_emails[0]]
            
            # Create a contact for each unique email
            for source, email_addr in all_emails:
                if email_addr in seen_emails:
                    continue  # Skip duplicates
                seen_emails.add(email_addr)
                
                contact = {
                    'name': None,
                    'email': email_addr,
                    'phone': None,
                    'company': None,
                    'linkedin_id': None,
                    'location': None,
                    'source': source_email,
                    'extraction_source': source  # Track where email came from
                }
                
                # Use vendor info from span if available
                if vendor_info.get('name'):
                    contact['name'] = vendor_info['name']
                if vendor_info.get('company'):
                    contact['company'] = vendor_info['company']
                
                # Extract name if not found from span
                if not contact['name']:
                    # PRIORITY 1: Extract name from the specific header that contained this email
                    header_name = self._extract_name_from_header_for_email(email_message, contact['email'])
                    if header_name and not self._is_candidate_name(header_name, source_email):
                        contact['name'] = header_name
                    
                    # PRIORITY 2: Try signature extraction (but validate it's not candidate)
                    if not contact['name'] and self.spacy_extractor:
                        signature_name = self.spacy_extractor.extract_name_from_signature(clean_body)
                        if signature_name and not self._is_candidate_name(signature_name, source_email):
                            contact['name'] = signature_name
                    
                    # PRIORITY 3: Fallback to name from email address
                    if not contact['name']:
                        contact['name'] = self._extract_name_from_email(contact['email'])
                
                # Extract phone
                contact['phone'] = self._extract_field('phone', clean_body, email_message)
                
                # Extract LinkedIn (validate it's actually a LinkedIn ID, not a name)
                linkedin_raw = self._extract_field('linkedin_id', clean_body, email_message)
                if linkedin_raw and self._is_valid_linkedin_id(linkedin_raw):
                    contact['linkedin_id'] = linkedin_raw
                
                # Extract company if not from span
                if not contact['company']:
                    contact['company'] = self._extract_field('company', clean_body, email_message, 
                                                             email=contact['email'])
                
                # Extract location
                contact['location'] = self._extract_field('location', clean_body, email_message)
                
                # Final validation and cleanup
                contact = self._validate_and_clean_contact(contact)
                
                # Only add if we have email or linkedin
                if contact.get('email') or contact.get('linkedin_id'):
                    contacts.append(contact)
            
            return contacts
            
        except Exception as e:
            self.logger.error(f"Error extracting contacts: {str(e)}", exc_info=True)
            return contacts
    
    def _validate_and_clean_contact(self, contact: Dict) -> Dict:
        """Final validation and cleanup of extracted contact"""
        try:
            # Clean up empty strings to None
            for key in contact:
                if isinstance(contact[key], str):
                    cleaned = contact[key].strip()
                    contact[key] = cleaned if cleaned else None
            
            # Validate email format
            if contact['email']:
                if '@' not in contact['email'] or '.' not in contact['email']:
                    self.logger.debug(f"Invalid email format: {contact['email']}")
                    contact['email'] = None
            
            # Validate phone format (should start with +)
            if contact['phone']:
                if not contact['phone'].startswith('+'):
                    self.logger.debug(f"Invalid phone format: {contact['phone']}")
                    contact['phone'] = None
            
            # Ensure we have at least email OR linkedin
            if not contact['email'] and not contact['linkedin_id']:
                self.logger.debug("No email or LinkedIn found - invalid contact")
                return contact
            
        except Exception as e:
            self.logger.error(f"Error validating contact: {str(e)}")
        
        return contact
    
    def _extract_field(self, field: str, text: str, email_message=None, **kwargs) -> Optional[str]:
        """
        Extract a specific field using configured method chain
        
        Args:
            field: Field name (name, email, phone, company, linkedin_id, location)
            text: Text to extract from
            email_message: Optional email message object
            **kwargs: Additional context (e.g., email for company extraction)
            
        Returns:
            Extracted value or None
        """
        # Define extraction methods per field
        field_methods = {
            'email': ['regex'],
            'phone': ['regex'],
            'linkedin_id': ['regex'],
            'name': ['spacy', 'gliner'],
            'company': ['spacy', 'gliner'],
            'location': ['gliner']
        }
        
        methods = field_methods.get(field, ['regex'])
        
        for method in methods:
            try:
                if method == 'regex':
                    value = self._extract_regex(field, text, email_message, **kwargs)
                elif method == 'spacy' and self.spacy_extractor:
                    value = self._extract_spacy(field, text, email_message, **kwargs)
                elif method == 'gliner' and self.gliner_extractor:
                    value = self._extract_gliner(field, text, **kwargs)
                else:
                    continue
                
                if value:
                    self.logger.debug(f"Extracted {field} using {method}: {value}")
                    return value
                    
            except Exception as e:
                self.logger.error(f"Error in {method} extraction for {field}: {str(e)}")
                continue
        
        return None
    
    def _extract_regex(self, field: str, text: str, email_message=None, **kwargs) -> Optional[str]:
        """Extract field using regex patterns"""
        if field == 'email':
            return self.regex_extractor.extract_email(text)
        elif field == 'phone':
            return self.regex_extractor.extract_phone(text)
        elif field == 'linkedin_id':
            return self.regex_extractor.extract_linkedin_id(text)
        return None
    
    def _extract_spacy(self, field: str, text: str, email_message=None, **kwargs) -> Optional[str]:
        """Extract field using Spacy NER with PRIORITY system"""
        if field == 'name':
            # PRIORITY 1: Try email header (From field) - most reliable
            if email_message:
                name = self.spacy_extractor.extract_name_from_header(email_message)
                if name and len(name.split()) >= 2:  # Full name with at least 2 words
                    return name
            
            # PRIORITY 2: Try signature (bottom of email)
            name = self.spacy_extractor.extract_name_from_signature(text)
            if name and len(name.split()) >= 2:
                return name
            
            # PRIORITY 3: Fallback to NER (less reliable)
            entities = self.spacy_extractor.extract_entities(text)
            return entities.get('name')
        
        elif field == 'company':
            # PRIORITY 1: HTML span extraction (already done in extract_contacts before this)
            # This is the fallback priority system
            
            # PRIORITY 2: Signature extraction (look for company after job title)
            signature_company = self.spacy_extractor.extract_company_from_signature(text)
            if signature_company:
                self.logger.debug(f"✓ Extracted company from signature: {signature_company}")
                return signature_company
            
            # PRIORITY 3: Domain extraction from email
            email = kwargs.get('email')
            if email:
                company = self.spacy_extractor.extract_company_from_domain(email)
                if company:
                    self.logger.debug(f"✓ Extracted company from domain: {company}")
                    return company
            
            # PRIORITY 4: NER extraction (lowest confidence, filtered for job titles)
            entities = self.spacy_extractor.extract_entities(text)
            return entities.get('company')
        
        elif field == 'location':
            entities = self.spacy_extractor.extract_entities(text)
            return entities.get('location')
        
        return None
    
    def _extract_gliner(self, field: str, text: str, **kwargs) -> Optional[str]:
        """Extract field using GLiNER"""
        entities = self.gliner_extractor.extract_entities(text)
        
        if field == 'name':
            return entities.get('name')
        elif field == 'company':
            return entities.get('company')
        elif field == 'location':
            return entities.get('location')
        elif field == 'job_title':
            return entities.get('job_title')
        
        return None
    
    def _extract_name_from_email(self, email: str) -> Optional[str]:
        """Extract and format name from email address (local part before @)"""
        try:
            # Get part before @
            local_part = email.split('@')[0]
            
            # Replace common separators with space
            name = local_part.replace('.', ' ').replace('_', ' ').replace('-', ' ')
            
            # Remove numbers and special chars
            name = ''.join(char if char.isalpha() or char.isspace() else ' ' for char in name)
            
            # Title case each word
            name = ' '.join(word.capitalize() for word in name.split() if len(word) > 1)
            
            # Only return if we got a reasonable name (2+ words or 1 word with 3+ chars)
            if len(name.split()) >= 2 or len(name) >= 3:
                self.logger.debug(f"Extracted name from email: {name}")
                return name
            
        except Exception as e:
            self.logger.error(f"Error extracting name from email: {str(e)}")
        
        return None
    
    def _is_candidate_name(self, name: str, source_email: str) -> bool:
        """Check if extracted name is the candidate's own name (not recruiter)
        
        CRITICAL: Must reject candidate names to avoid inserting receiver as vendor
        """
        if not name or not source_email:
            return False
        
        name_lower = name.lower().strip()
        
        # Filter common greetings and invalid patterns
        greeting_patterns = [
            'dear', 'hi ', 'hello', 'hey', 'greetings',
            'team', 'sir', 'madam', 'folks', 'all',
            'recipient', 'candidate', 'applicant'
        ]
        
        for pattern in greeting_patterns:
            if pattern in name_lower or name_lower.startswith(pattern):
                self.logger.info(f"✗ Rejected greeting/generic name: {name}")
                return True
        
        # Reject if name looks like a company/team name (all caps, contains common company words)
        company_indicators = ['team', 'group', 'department', 'inc', 'llc', 'corp', 'ltd']
        if any(indicator in name_lower for indicator in company_indicators):
            self.logger.info(f"✗ Rejected company/team name: {name}")
            return True
        
        # Check against email local part
        email_local = source_email.split('@')[0].lower()
        
        # Clean and split name parts
        name_parts = [part.strip() for part in name_lower.replace('.', ' ').replace('_', ' ').replace('-', ' ').split() if len(part) > 1]
        
        # If no valid name parts, can't validate
        if not name_parts:
            return False
        
        # Check if name parts appear in email local part
        matches = 0
        for part in name_parts:
            if len(part) >= 3 and part in email_local:  # Only match meaningful parts (3+ chars)
                matches += 1
        
        # If 2+ name parts match email, it's the candidate
        if matches >= 2:
            self.logger.info(f"✗ Rejected candidate's own name: {name} (from {source_email})")
            return True
        
        # Additional check: If single name part matches and email is short
        if len(name_parts) == 2 and matches == 1:
            # Check if the matched part is significant portion of email
            for part in name_parts:
                if part in email_local and len(part) >= 4:
                    # If part is >50% of email local, likely candidate
                    if len(part) / len(email_local) > 0.5:
                        self.logger.info(f"✗ Rejected candidate's own name: {name} (from {source_email})")
                        return True
        
        return False
    
    def _is_valid_linkedin_id(self, value: str) -> bool:
        """Validate LinkedIn ID - should be username, not a full name"""
        if not value:
            return False
        
        # LinkedIn IDs are typically:
        # - Single word or hyphenated (john-smith-123)
        # - No spaces
        # - Max 100 chars
        # - Alphanumeric with hyphens/underscores
        
        # If has multiple spaces, it's likely a name, not an ID
        if value.count(' ') >= 2:
            return False
        
        # If has common name patterns (Mr., Mrs., Dr., Jr., etc.)
        if any(title in value.lower() for title in ['mr.', 'mrs.', 'ms.', 'dr.', 'jr.', 'sr.', 'phd']):
            return False
        
        # If too long (LinkedIn IDs are usually <50 chars)
        if len(value) > 50:
            return False
        
        # If has @ symbol, it's likely an email, not LinkedIn ID
        if '@' in value:
            return False
        
        return True
    
    def _extract_from_cc_headers(self, email_message) -> Optional[str]:
        """Extract recruiter email from TO/CC/Reply-To headers"""
        try:
            all_emails = set()
            
            # Check TO header (multiple recipients)
            to_header = email_message.get('To', '')
            if to_header:
                for addr in to_header.split(','):
                    _, email_addr = parseaddr(addr.strip())
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            all_emails.add(email_lower)
            
            # Check CC header
            cc_header = email_message.get('Cc', '')
            if cc_header:
                for addr in cc_header.split(','):
                    _, email_addr = parseaddr(addr.strip())
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            all_emails.add(email_lower)
            
            # Check Reply-To (often recruiter's direct email - HIGHEST PRIORITY)
            reply_to = email_message.get('Reply-To', '')
            if reply_to:
                _, email_addr = parseaddr(reply_to)
                if email_addr and '@' in email_addr:
                    email_lower = email_addr.lower()
                    if self._is_valid_header_email(email_lower):
                        # Prioritize Reply-To
                        return email_lower
            
            # Return first valid email
            return list(all_emails)[0] if all_emails else None
            
        except Exception as e:
            self.logger.error(f"Error extracting header emails: {str(e)}")
            return None
    
    
    def _is_gmail_address(self, email: str, block_gmail: bool = True) -> bool:
        """Check if email is from Gmail or other personal domains"""
        if not block_gmail:
            return False
        
        if not email or '@' not in email:
            return False
        
        personal_domains = {
            'gmail.com', 'googlemail.com', 'yahoo.com', 'yahoo.co.uk', 'yahoo.in',
            'outlook.com', 'hotmail.com', 'live.com', 'msn.com',
            'icloud.com', 'me.com', 'mac.com',
            'aol.com', 'protonmail.com', 'proton.me', 'pm.me'
        }
        
        try:
            domain = email.split('@')[1].lower()
            if domain in personal_domains:
                self.logger.debug(f"✗ Blocked personal email domain: {email}")
                return True
        except:
            pass
        
        return False
    
    def _extract_reply_to_email(self, email_message) -> Optional[str]:
        """Extract email from Reply-To header (highest priority)"""
        try:
            reply_to = email_message.get('Reply-To', '')
            if reply_to:
                _, email_addr = parseaddr(reply_to)
                if email_addr and '@' in email_addr:
                    email_lower = email_addr.lower()
                    if self._is_valid_header_email(email_lower):
                        self.logger.debug(f"✓ Extracted Reply-To: {email_lower}")
                        return email_lower
        except Exception as e:
            self.logger.error(f"Error extracting Reply-To: {str(e)}")
        return None
    
    def _extract_sender_email(self, email_message) -> Optional[str]:
        """Extract email from Sender header"""
        try:
            sender = email_message.get('Sender', '')
            if sender:
                _, email_addr = parseaddr(sender)
                if email_addr and '@' in email_addr:
                    email_lower = email_addr.lower()
                    if self._is_valid_header_email(email_lower):
                        self.logger.debug(f"✓ Extracted Sender: {email_lower}")
                        return email_lower
        except Exception as e:
            self.logger.error(f"Error extracting Sender: {str(e)}")
        return None
    
    def _extract_from_header(self, email_message) -> Optional[str]:
        """Extract email from From header"""
        try:
            from_header = email_message.get('From', '')
            if from_header:
                _, email_addr = parseaddr(from_header)
                if email_addr and '@' in email_addr:
                    email_lower = email_addr.lower()
                    if self._is_valid_header_email(email_lower):
                        self.logger.debug(f"✓ Extracted From: {email_lower}")
                        return email_lower
        except Exception as e:
            self.logger.error(f"Error extracting From: {str(e)}")
        return None
    
    def _extract_cc_bcc_emails(self, email_message) -> List[str]:
        """Extract emails from CC and BCC headers"""
        emails = []
        try:
            # Check CC header
            cc_header = email_message.get('Cc', '')
            if cc_header:
                for addr in cc_header.split(','):
                    _, email_addr = parseaddr(addr.strip())
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted CC: {email_lower}")
            
            # Check BCC header (rarely present in received emails, but check anyway)
            bcc_header = email_message.get('Bcc', '')
            if bcc_header:
                for addr in bcc_header.split(','):
                    _, email_addr = parseaddr(addr.strip())
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted BCC: {email_lower}")
        
        except Exception as e:
            self.logger.error(f"Error extracting CC/BCC: {str(e)}")
        
        return emails
    
    def _is_valid_header_email(self, email: str) -> bool:
        """Check if header email is valid recruiter email (not automated/system)"""
        # Skip automated/system emails
        skip_keywords = ['noreply', 'no-reply', 'donotreply', 'notifications', 
                        'alerts', 'info', 'support', 'admin', 'system', 'mailer']
        
        if any(kw in email for kw in skip_keywords):
            return False
        
        # Note: Personal domains (Gmail, Yahoo, etc.) are filtered by _is_gmail_address
        # This method only checks for automated/system emails
        return True
    
    def _extract_name_from_header_for_email(self, email_message, email_addr: str) -> Optional[str]:
        """
        Extract name from the email header that contains the specified email address.
        This ensures we get the vendor's name from their header, not from body text.
        
        Args:
            email_message: Email message object
            email_addr: Email address to find in headers
            
        Returns:
            Name from the matching header, or None
        """
        if not email_addr:
            return None
        
        email_lower = email_addr.lower()
        
        try:
            # Check Reply-To header
            reply_to = email_message.get('Reply-To', '')
            if reply_to and email_lower in reply_to.lower():
                name, addr = parseaddr(reply_to)
                if name and name != addr:  # Has a display name
                    self.logger.debug(f"✓ Extracted name from Reply-To: {name}")
                    return name.strip()
            
            # Check Sender header
            sender = email_message.get('Sender', '')
            if sender and email_lower in sender.lower():
                name, addr = parseaddr(sender)
                if name and name != addr:
                    self.logger.debug(f"✓ Extracted name from Sender: {name}")
                    return name.strip()
            
            # Check From header
            from_header = email_message.get('From', '')
            if from_header and email_lower in from_header.lower():
                name, addr = parseaddr(from_header)
                if name and name != addr:
                    self.logger.debug(f"✓ Extracted name from From: {name}")
                    return name.strip()
            
            # Check CC header
            cc = email_message.get('Cc', '')
            if cc:
                for addr_str in cc.split(','):
                    addr_str = addr_str.strip()
                    if email_lower in addr_str.lower():
                        name, addr = parseaddr(addr_str)
                        if name and name != addr:
                            self.logger.debug(f"✓ Extracted name from CC: {name}")
                            return name.strip()
            
            # Check BCC header (rarely present)
            bcc = email_message.get('Bcc', '')
            if bcc:
                for addr_str in bcc.split(','):
                    addr_str = addr_str.strip()
                    if email_lower in addr_str.lower():
                        name, addr = parseaddr(addr_str)
                        if name and name != addr:
                            self.logger.debug(f"✓ Extracted name from BCC: {name}")
                            return name.strip()
        
        except Exception as e:
            self.logger.error(f"Error extracting name from header for {email_addr}: {str(e)}")
        
        return None
    
    
    def _get_html_body(self, email_message) -> Optional[str]:
        """Extract HTML body from email message"""
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/html':
                        payload = part.get_payload(decode=True)
                        if payload:
                            return payload.decode('utf-8', errors='ignore')
            else:
                if email_message.get_content_type() == 'text/html':
                    payload = email_message.get_payload(decode=True)
                    if payload:
                        return payload.decode('utf-8', errors='ignore')
            return None
        except Exception as e:
            self.logger.error(f"Error getting HTML body: {str(e)}")
            return None
    
    def _extract_calendar_email(self, email_message) -> Optional[List[str]]:
        """Extract emails from calendar invites"""
        emails = set()
        
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/calendar":
                        payload = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        
                        # Extract ORGANIZER
                        for match in re.findall(r"ORGANIZER.*mailto:([^ \r\n]+)", payload, re.IGNORECASE):
                            emails.add(match.lower())
                        
                        # Extract ATTENDEE
                        for match in re.findall(r"ATTENDEE.*mailto:([^ \r\n]+)", payload, re.IGNORECASE):
                            emails.add(match.lower())
            
            # Fallback to headers
            if not emails:
                for header in ["Sender", "Reply-To", "From"]:
                    if header in email_message:
                        _, addr = parseaddr(email_message.get(header))
                        if addr and "noreply" not in addr.lower():
                            emails.add(addr.lower())
            
            return list(emails) if emails else None
            
        except Exception as e:
            self.logger.error(f"Error extracting calendar emails: {str(e)}")
            return None
