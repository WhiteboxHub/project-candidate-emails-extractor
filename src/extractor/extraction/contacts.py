from typing import Dict, Optional, List
import logging
import re
from email.utils import parseaddr, getaddresses
from .patterns import RegexExtractor
from .nlp_spacy import SpacyNERExtractor
from .nlp_gliner import GLiNERExtractor
from .positions import PositionExtractor
from .location import LocationExtractor
from .employment_type import EmploymentTypeExtractor
from ..filtering.repository import get_filter_repository

logger = logging.getLogger(__name__)

class ContactExtractor: 
    """
    Unified contact extraction with config-driven fallback chain
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize filter repository
        self.filter_repo = get_filter_repository()
        
        # Load filter lists from CSV (no hardcoded fallbacks)
        self.greeting_patterns = self._load_greeting_patterns()
        self.company_indicators = self._load_company_indicators()
        self.skip_keywords = self._load_skip_keywords()
        
        # Initialize extractors based on config
        enabled_methods = config.get('extraction', {}).get('enabled_methods', ['regex', 'spacy'])
        
        self.regex_extractor = RegexExtractor()
        self.spacy_extractor = None
        self.gliner_extractor = None
        self.position_extractor = None
        self.location_extractor = None
        self.employment_type_extractor = None
        
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
        
        # Initialize custom extractors
        try:
            # Position extractor (uses spacy model if available)
            spacy_nlp = self.spacy_extractor.nlp if self.spacy_extractor else None
            self.position_extractor = PositionExtractor(spacy_model=spacy_nlp)
            self.location_extractor = LocationExtractor()
            self.employment_type_extractor = EmploymentTypeExtractor()
            self.logger.info("Position, Location, and Employment Type extractors initialized")
        except Exception as e:
            self.logger.warning(f"Failed to load custom extractors: {str(e)}")
    
    def _load_greeting_patterns(self) -> list:
        """Load greeting patterns from CSV (no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            if 'greeting_patterns' in keyword_lists:
                patterns = keyword_lists['greeting_patterns']
                self.logger.info(f"✓ Loaded {len(patterns)} greeting patterns from CSV")
                return patterns
            else:
                self.logger.error("⚠ greeting_patterns not found in CSV - using empty list")
                return []
        except Exception as e:
            self.logger.error(f"Failed to load greeting patterns from CSV: {str(e)} - using empty list")
            return []
    
    def _load_company_indicators(self) -> list:
        """Load company indicators from CSV (no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            if 'company_indicators' in keyword_lists:
                indicators = keyword_lists['company_indicators']
                self.logger.info(f"✓ Loaded {len(indicators)} company indicators from CSV")
                return indicators
            else:
                self.logger.error("⚠ company_indicators not found in CSV - using empty list")
                return []
        except Exception as e:
            self.logger.error(f"Failed to load company indicators from CSV: {str(e)} - using empty list")
            return []
    
    def _load_skip_keywords(self) -> list:
        """Load skip keywords from CSV (no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            if 'skip_header_keywords' in keyword_lists:
                keywords = keyword_lists['skip_header_keywords']
                self.logger.info(f"✓ Loaded {len(keywords)} skip keywords from CSV")
                return keywords
            else:
                self.logger.error("⚠ skip_header_keywords not found in CSV - using empty list")
                return []
        except Exception as e:
            self.logger.error(f"Failed to load skip keywords from CSV: {str(e)} - using empty list")
            return []
    
    def extract_contacts(self, email_message, clean_body: str, source_email: str, subject: str = None) -> List[Dict]:
        """
        Extract contact information with fallback chain - returns LIST of contacts
        
        Args:
            email_message: Email message object
            clean_body: Cleaned email body text
            source_email: Source candidate email
            subject: Email subject line (optional, for better job position extraction)
            
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
            # Priority order for SOURCE: Reply-To > Sender > From
            # ALWAYS extract: To > CC/BCC > Calendar > Body
            all_emails = []
            
            # 1. Fallback chain for primary "source" email
            source_candidates = []
            
            # Try Reply-To first (highest priority)
            reply_to_emails = self._extract_reply_to_email(email_message)
            if reply_to_emails:
                source_candidates = [('reply-to', email) for email in reply_to_emails]
            
            # If no Reply-To, try Sender
            if not source_candidates:
                sender_emails = self._extract_sender_email(email_message)
                if sender_emails:
                    source_candidates = [('sender', email) for email in sender_emails]
            
            # If no Sender, try From
            if not source_candidates:
                from_emails = self._extract_from_header(email_message)
                if from_emails:
                    source_candidates = [('from', email) for email in from_emails]
            
            # Add filtered source candidates to all_emails
            for source, email_addr in source_candidates:
                if not self._is_gmail_address(email_addr, block_gmail):
                    all_emails.append((source, email_addr))
            
            # 2. To header (recipients - always extract as separate contacts)
            to_emails = self._extract_to_recipients(email_message)
            for to_email in to_emails:
                if not self._is_gmail_address(to_email, block_gmail):
                    all_emails.append(('to', to_email))
            
            # 3. CC/BCC headers (additional contacts - always extract)
            cc_emails = self._extract_cc_bcc_emails(email_message)
            for cc_email in cc_emails:
                if not self._is_gmail_address(cc_email, block_gmail):
                    all_emails.append(('cc', cc_email))
            
            # 4. Calendar invite emails
            calendar_emails = self._extract_calendar_email(email_message)
            if calendar_emails:
                for cal_email in calendar_emails:
                    if not self._is_gmail_address(cal_email, block_gmail):
                        all_emails.append(('calendar', cal_email))
            
            # 5. Body extraction
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
                    'job_position': None,
                    'zip_code': None,
                    'employment_type': None,  # NEW: W2, C2C, Contract, etc.
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
                
                # Fallback: Extract company from email domain if still null
                if not contact['company'] and contact['email']:
                    contact['company'] = self._extract_company_from_email(contact['email'])
                
                # CRITICAL FIX: Extract location from SUBJECT FIRST, then body
                # This ensures we capture locations like "Charlotte, NC" from subject lines
                location_from_subject = None
                if subject and self.location_extractor:
                    # Normalize acronyms in subject
                    normalized_subject = self.position_extractor._normalize_acronyms_in_text(subject) if self.position_extractor else subject
                    location_from_subject = self.location_extractor.extract_location_with_zip(normalized_subject)
                
                # If we got location from subject, use it (HIGHEST PRIORITY)
                if location_from_subject and (location_from_subject.get('location') or location_from_subject.get('zip_code')):
                    contact['location'] = location_from_subject.get('location')
                    contact['zip_code'] = location_from_subject.get('zip_code')
                    self.logger.debug(f"✓ Extracted location from SUBJECT: {contact['location']}")
                else:
                    # Fallback to body extraction
                    # Normalize acronyms in body
                    normalized_body = self.position_extractor._normalize_acronyms_in_text(clean_body) if self.position_extractor else clean_body
                    location_data = self._extract_field('location_with_zip', normalized_body, email_message)
                    if location_data and isinstance(location_data, dict):
                        contact['location'] = location_data.get('location')
                        contact['zip_code'] = location_data.get('zip_code')
                    else:
                        # Fallback to basic location extraction
                        contact['location'] = self._extract_field('location', normalized_body, email_message)
                
                
                # Check if body is encrypted or very short/junk
                subject_upper = subject.upper() if subject else ""
                is_encrypted = any(marker in subject_upper for marker in ["ENCRYPTED", "SECURE MESSAGE", "CONFIDENTIAL"])
                is_junk_body = len(clean_body.strip()) < 50
                
                # For encrypted or junk bodies, extract job position from subject
                if (is_encrypted or is_junk_body) and subject:
                    self.logger.debug(f"⚠ Body is encrypted/junk - extracting from subject: {subject[:100]}")
                    
                    # Extract position from subject if not already extracted
                    if not contact['job_position']:
                        contact['job_position'] = self._extract_field('job_position', subject, email_message=email_message, subject=subject)
                    
                    # CRITICAL FIX: Extract employment_type from subject for encrypted emails
                    if not contact['employment_type'] and self.employment_type_extractor:
                        # Normalize acronyms in subject before extraction
                        normalized_subject = self.position_extractor._normalize_acronyms_in_text(subject) if self.position_extractor else subject
                        contact['employment_type'] = self.employment_type_extractor.extract_employment_type_string(normalized_subject)
                        if contact['employment_type']:
                            self.logger.debug(f"✓ Extracted employment_type from SUBJECT: {contact['employment_type']}")
                    
                    # CRITICAL FIX: Extract location from subject for encrypted emails (if not already extracted)
                    if not contact['location'] and self.location_extractor:
                        # Normalize acronyms in subject
                        normalized_subject = self.position_extractor._normalize_acronyms_in_text(subject) if self.position_extractor else subject
                        location_data = self.location_extractor.extract_location_with_zip(normalized_subject)
                        if location_data and isinstance(location_data, dict):
                            contact['location'] = location_data.get('location')
                            contact['zip_code'] = location_data.get('zip_code')
                            self.logger.debug(f"✓ Extracted location from SUBJECT (encrypted): {contact['location']}")
                
                # Validate and clean up extracted data
                if contact['job_position']:
                    contact['job_position'] = contact['job_position'].strip()
                if contact['location']:
                    contact['location'] = contact['location'].strip()
                if contact['company']:
                    contact['company'] = contact['company'].strip()
                
                # Extract job position
                # Pass subject line for better position extraction
                if not contact['job_position']: # Only extract if not already extracted from subject
                    contact['job_position'] = self._extract_field('job_position', clean_body, email_message, subject=subject)
                
                # If still no position and it's a junk/encrypted body, try a last-ditch attempt on subject ONLY
                # This block is now mostly redundant due to the new encrypted/junk body handling above,
                # but kept for any edge cases where the above might not catch it.
                if not contact['job_position'] and is_junk_body:
                    contact['job_position'] = self._extract_field('job_position', subject, email_message)
                
                # If body is encrypted, many body-based extractions will fail. 
                # Use subject for location/company/position as much as possible.
                # This block is now mostly handled by the new encrypted/junk body handling above.
                # The location part is specifically addressed.
                if is_encrypted:
                    if not contact['location']: # Only try if location wasn't extracted from subject already
                        # Try extracting location from subject
                        loc_from_sub = self._extract_field('location_with_zip', subject, email_message)
                        if loc_from_sub and isinstance(loc_from_sub, dict):
                            contact['location'] = loc_from_sub.get('location')
                            contact['zip_code'] = loc_from_sub.get('zip_code')
                    
                    if not contact['employment_type'] and self.employment_type_extractor:
                        contact['employment_type'] = ', '.join(self.employment_type_extractor.extract_employment_types("", subject))

                # Extract employment type (W2, C2C, Contract, etc.) if not already set
                if not contact['employment_type'] and self.employment_type_extractor:
                    employment_types = self.employment_type_extractor.extract_employment_types(clean_body, subject)
                    if employment_types:
                        contact['employment_type'] = ', '.join(employment_types)
                
                # CRITICAL: Cross-validate company and location to prevent conflicts
                # STRATEGY: When conflict occurs, REJECT LOCATION (company extraction is more reliable)
                if contact['company'] and contact['location']:
                    # If company and location are the same or very similar, location is wrong
                    company_lower = contact['company'].lower().strip()
                    location_lower = contact['location'].lower().strip()
                    
                    # Check if they're identical or one contains the other
                    if company_lower == location_lower:
                        self.logger.warning(f"❌ Location matches company - rejecting location: {contact['location']} (keeping company: {contact['company']})")
                        contact['location'] = None
                    elif company_lower in location_lower or location_lower in company_lower:
                        # If one is contained in the other, reject location (keep company)
                        self.logger.warning(f"❌ Location overlaps with company - rejecting location: {contact['location']} (keeping company: {contact['company']})")
                        contact['location'] = None
                
                # Additional check: If company looks like a location, reject it
                if contact['company'] and self.spacy_extractor:
                    if hasattr(self.spacy_extractor, '_is_location') and self.spacy_extractor._is_location(contact['company']):
                        self.logger.warning(f"⚠️  Company looks like a location - rejecting: {contact['company']}")
                        contact['company'] = None
                
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
            'location': ['gliner', 'spacy'],
            'location_with_zip': ['custom'],  # Use custom location extractor
            'job_position': ['custom'],  # Use custom position extractor
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
                elif method == 'custom':
                    # Handle custom extractors
                    if field == 'job_position':
                        value = self._extract_job_position(text, **kwargs)
                    elif field == 'location_with_zip':
                        value = self._extract_location_with_zip(text, **kwargs)
                    else:
                        continue
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
            # Use NEW scoring-based extraction system (CSV-driven, no hardcoded values)
            # This replaces the old priority system with candidate scoring
            email = kwargs.get('email')
            
            # Get raw HTML if available for span extraction
            html = None
            if email_message:
                html = self._get_html_body(email_message)
            
            # Use scoring system to pick best candidate
            company = self.spacy_extractor.extract_company_with_scoring(
                text=text,
                email=email,
                html=html
            )
            
            if company:
                self.logger.debug(f"✓ Extracted company using scoring: {company}")
            
            return company
        
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
    
    def _extract_job_position(self, text: str, **kwargs) -> Optional[str]:
        """Extract job position using PositionExtractor"""
        if not self.position_extractor:
            return None
        
        try:
            subject = kwargs.get('subject', '')
            
            # CRITICAL: Normalize acronyms in BOTH text and subject BEFORE any extraction
            # This fixes "I/ML Engineer" → "AI/ML Engineer" for ALL extraction methods
            if text:
                text = self.position_extractor._normalize_acronyms_in_text(text)
            if subject:
                subject = self.position_extractor._normalize_acronyms_in_text(subject)
            
            # Try regex first (fast and accurate for common patterns)
            position = self.position_extractor.extract_job_position_regex(text)
            if position:
                self.logger.debug(f"✓ Extracted position (regex): {position}")
                return position
            
            # Try subject line if available
            if subject:
                position = self.position_extractor.extract_job_position_regex(subject)
                if position:
                    self.logger.debug(f"✓ Extracted position from subject (regex): {position}")
                    return position
            
            # Try spacy noun phrase extraction
            position = self.position_extractor.extract_job_position_spacy(text)
            if position:
                self.logger.debug(f"✓ Extracted position (spacy): {position}")
                return position
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting job position: {str(e)}")
            return None
    
    def _extract_location_with_zip(self, text: str, **kwargs) -> Optional[Dict]:
        """Extract location with zip code using LocationExtractor"""
        if not self.location_extractor:
            return None
        
        try:
            location_data = self.location_extractor.extract_location_with_zip(text)
            
            # Only return if we got at least location or zip
            if location_data.get('location') or location_data.get('zip_code'):
                return location_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting location with zip: {str(e)}")
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
    
    def _extract_company_from_email(self, email: str) -> Optional[str]:
        """Extract company name from email domain as fallback
        
        Args:
            email: Email address (e.g., prakash.k@nityo.com)
            
        Returns:
            Company name (e.g., "Nityo") or None if blocked domain
        """
        if not email or '@' not in email:
            return None
        
        try:
            # Extract domain
            domain = email.split('@')[1].lower()
            
            # Check if domain is blocked (gmail, yahoo, etc.) using filter_repo
            if self.filter_repo.check_email(email) == 'block':
                self.logger.debug(f"✗ Blocked personal domain for company extraction: {domain}")
                return None
            
            # Remove TLD (.com, .co, .org, .net, etc.)
            company_name = domain.split('.')[0]
            
            # Capitalize first letter
            company_name = company_name.capitalize()
            
            self.logger.debug(f"✓ Extracted company from email domain: {company_name} (from {email})")
            return company_name
            
        except Exception as e:
            self.logger.error(f"Error extracting company from email: {str(e)}")
            return None
    
    def _is_candidate_name(self, name: str, source_email: str) -> bool:
        """Check if extracted name is the candidate's own name (not recruiter)
        
        CRITICAL: Must reject candidate names to avoid inserting receiver as vendor
        """
        if not name or not source_email:
            return False
        
        name_lower = name.lower().strip()
        
        # Filter common greetings and invalid patterns (loaded from CSV)
        for pattern in self.greeting_patterns:
            if pattern in name_lower or name_lower.startswith(pattern):
                self.logger.info(f"✗ Rejected greeting/generic name: {name}")
                return True
        
        # Reject if name looks like a company/team name (loaded from CSV)
        if any(indicator in name_lower for indicator in self.company_indicators):
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
        """Check if email is from Gmail or other personal domains using database filters"""
        if not block_gmail:
            return False
        
        if not email or '@' not in email:
            return False
        
        action = self.filter_repo.check_email(email)
        if action == 'block':
            self.logger.debug(f"✗ Blocked personal email domain: {email}")
            return True
        
        return False
    
    def _extract_reply_to_email(self, email_message) -> List[str]:
        """Extract emails from Reply-To header"""
        emails = []
        try:
            reply_to = email_message.get('Reply-To', '')
            if reply_to:
                for name, email_addr in getaddresses([reply_to]):
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted Reply-To: {email_lower}")
        except Exception as e:
            self.logger.error(f"Error extracting Reply-To: {str(e)}")
        return emails
    
    def _extract_sender_email(self, email_message) -> List[str]:
        """Extract emails from Sender header"""
        emails = []
        try:
            sender = email_message.get('Sender', '')
            if sender:
                for name, email_addr in getaddresses([sender]):
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted Sender: {email_lower}")
        except Exception as e:
            self.logger.error(f"Error extracting Sender: {str(e)}")
        return emails
    
    def _extract_from_header(self, email_message) -> List[str]:
        """Extract emails from From header"""
        emails = []
        try:
            from_header = email_message.get('From', '')
            if from_header:
                for name, email_addr in getaddresses([from_header]):
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted From: {email_lower}")
        except Exception as e:
            self.logger.error(f"Error extracting From: {str(e)}")
        return emails
    
    def _extract_to_recipients(self, email_message) -> List[str]:
        """Extract emails from To header"""
        emails = []
        try:
            to_header = email_message.get('To', '')
            if to_header:
                for name, email_addr in getaddresses([to_header]):
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted To: {email_lower}")
        
        except Exception as e:
            self.logger.error(f"Error extracting To: {str(e)}")
        
        return emails
    
    def _extract_cc_bcc_emails(self, email_message) -> List[str]:
        """Extract emails from CC and BCC headers"""
        emails = []
        try:
            # Check CC header
            cc_header = email_message.get('Cc', '')
            if cc_header:
                for name, email_addr in getaddresses([cc_header]):
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted CC: {email_lower}")
            
            # Check BCC header (rarely present in received emails, but check anyway)
            bcc_header = email_message.get('Bcc', '')
            if bcc_header:
                for name, email_addr in getaddresses([bcc_header]):
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
        # Skip automated/system emails (loaded from CSV)
        if any(kw in email for kw in self.skip_keywords):
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
            # Headers to check in order
            headers_to_check = ['Reply-To', 'Sender', 'From', 'To', 'Cc', 'Bcc']
            
            for header_name in headers_to_check:
                header_value = email_message.get(header_name, '')
                if not header_value:
                    continue
                    
                # Parse all addresses in this header
                for name, addr in getaddresses([header_value]):
                    if addr and addr.lower() == email_lower:
                        if name and name != addr:
                            self.logger.debug(f"✓ Extracted name from {header_name}: {name}")
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
            
            
            return list(emails) if emails else None
            
        except Exception as e:
            self.logger.error(f"Error extracting calendar emails: {str(e)}")
            return None
