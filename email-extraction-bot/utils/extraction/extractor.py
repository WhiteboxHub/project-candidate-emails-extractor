from typing import Dict, Optional, List
import logging
from email.utils import parseaddr, getaddresses
import re

from .regex_util import RegexExtractor
from .ner_util import SpacyNERExtractor
from .gliner_util import GLiNERExtractor
from utils.filters.email_filters import EmailFilter

logger = logging.getLogger(__name__)

class ContactExtractor:
    """
    Unified contact extraction with DB-driven filtering
    """
    def __init__(self, config: dict, email_filter: EmailFilter = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.email_filter = email_filter  # DB-driven filter

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
        Extract contact information with DB-driven filtering
        """
        contacts = []
        seen_emails = set()
        try:
            extract_multiple = self.config.get('extraction', {}).get('extract_multiple_contacts', True)
            raw_html = self._get_html_body(email_message)
            vendor_info = {'name': None, 'company': None}
            if raw_html and self.spacy_extractor:
                vendor_info = self.spacy_extractor.extract_vendor_from_span(raw_html)
            all_emails = []

            # 1. Reply-To (highest priority)
            reply_to_email = self._extract_reply_to_email(email_message)
            if reply_to_email and self._is_valid_email(reply_to_email):
                all_emails.append(('reply-to', reply_to_email))

            # 2. Sender header
            sender_email = self._extract_sender_email(email_message)
            if sender_email and self._is_valid_email(sender_email):
                all_emails.append(('sender', sender_email))

            # 3. From header
            from_email = self._extract_from_header(email_message)
            if from_email and self._is_valid_email(from_email):
                all_emails.append(('from', from_email))

            # 4. To header (extract all To recipients)
            to_emails = self._extract_to_emails(email_message)
            for to_email in to_emails:
                if self._is_valid_email(to_email):
                    all_emails.append(('to', to_email))

            # 5. CC/BCC headers
            cc_emails = self._extract_cc_bcc_emails(email_message)
            for cc_email in cc_emails:
                if self._is_valid_email(cc_email):
                    all_emails.append(('cc', cc_email))

            # 6. Calendar invite emails
            calendar_emails = self._extract_calendar_email(email_message)
            if calendar_emails:
                for cal_email in calendar_emails:
                    if self._is_valid_email(cal_email):
                        all_emails.append(('calendar', cal_email))

            # 7. Body extraction (lowest priority)
            body_email = self._extract_field('email', clean_body, email_message)
            if body_email and self._is_valid_email(body_email):
                all_emails.append(('body', body_email))

            # Only use top-priority contact if extract_multiple is False
            if not extract_multiple and all_emails:
                all_emails = [all_emails[0]]

            # Build unique contacts
            for source, email_addr in all_emails:
                if email_addr in seen_emails:
                    continue
                seen_emails.add(email_addr)
                contact = {
                    'name': None,
                    'email': email_addr,
                    'phone': None,
                    'company': None,
                    'linkedin_id': None,
                    'location': None,
                    'source': source_email,
                    'extraction_source': source
                }
                # Vendor info
                if vendor_info.get('name'):
                    contact['name'] = vendor_info['name']
                if vendor_info.get('company'):
                    contact['company'] = vendor_info['company']
                # Name extraction
                if not contact['name']:
                    header_name = self._extract_name_from_header_for_email(email_message, contact['email'])
                    if header_name and self._is_valid_name(header_name, source_email):
                        contact['name'] = header_name
                    elif not contact['name'] and self.spacy_extractor:
                        signature_name = self.spacy_extractor.extract_name_from_signature(clean_body)
                        if signature_name and self._is_valid_name(signature_name, source_email):
                            contact['name'] = signature_name
                    if not contact['name']:
                        contact['name'] = self._extract_name_from_email(contact['email'])
                # Phone
                phone_raw = self._extract_field('phone', clean_body, email_message)
                if phone_raw and self._is_valid_phone(phone_raw):
                    contact['phone'] = phone_raw
                # LinkedIn
                linkedin_raw = self._extract_field('linkedin_id', clean_body, email_message)
                if linkedin_raw and self._is_valid_linkedin_id(linkedin_raw):
                    contact['linkedin_id'] = linkedin_raw
                # Company fallback
                if not contact['company']:
                    company = self._extract_field('company', clean_body, email_message, email=contact['email'])
                    if company and self._is_valid_company(company):
                        contact['company'] = company
                # Location
                location = self._extract_field('location', clean_body, email_message)
                if location and self._is_valid_location_db(location):
                    contact['location'] = location
                contact = self._validate_and_clean_contact(contact)
                if contact.get('email') or contact.get('linkedin_id'):
                    contacts.append(contact)
            return contacts

        except Exception as e:
            self.logger.error(f"Error extracting contacts: {str(e)}", exc_info=True)
            return contacts

    def _validate_and_clean_contact(self, contact: Dict) -> Dict:
        """Final validation and cleanup of extracted contact"""
        try:
            # Clean up strings
            for key in contact:
                if isinstance(contact[key], str):
                    cleaned = contact[key].strip()
                    contact[key] = cleaned if cleaned else None
            # Email
            if contact['email']:
                if '@' not in contact['email'] or '.' not in contact['email']:
                    self.logger.debug(f"Invalid email format: {contact['email']}")
                    contact['email'] = None
            # Phone - already validated during extraction
            # Requires email or LinkedIn
            if not contact['email'] and not contact['linkedin_id']:
                self.logger.debug("No email or LinkedIn found - invalid contact")
                return contact
        except Exception as e:
            self.logger.error(f"Error validating contact: {str(e)}")
        return contact

    def _extract_field(self, field: str, text: str, email_message=None, **kwargs) -> Optional[str]:
        """
        Extract a specific field using configured method chain
        """
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
                value = None
                if method == 'regex':
                    value = self._extract_regex(field, text, email_message, **kwargs)
                elif method == 'spacy' and self.spacy_extractor:
                    value = self._extract_spacy(field, text, email_message, **kwargs)
                elif method == 'gliner' and self.gliner_extractor:
                    value = self._extract_gliner(field, text, **kwargs)
                if value:
                    self.logger.debug(f"Extracted {field} using {method}: {value}")
                    return value
            except Exception as e:
                self.logger.error(f"Error in {method} extraction for {field}: {str(e)}")
        return None

    def _extract_regex(self, field: str, text: str, email_message=None, **kwargs) -> Optional[str]:
        if field == 'email':
            return self.regex_extractor.extract_email(text)
        elif field == 'phone':
            return self.regex_extractor.extract_phone(text)
        elif field == 'linkedin_id':
            return self.regex_extractor.extract_linkedin_id(text)
        return None

    def _extract_spacy(self, field: str, text: str, email_message=None, **kwargs) -> Optional[str]:
        if field == 'name':
            if email_message:
                name = self.spacy_extractor.extract_name_from_header(email_message)
                if name and len(name.split()) >= 2:
                    return name
            name = self.spacy_extractor.extract_name_from_signature(text)
            if name and len(name.split()) >= 2:
                return name
            entities = self.spacy_extractor.extract_entities(text)
            return entities.get('name')
        elif field == 'company':
            signature_company = self.spacy_extractor.extract_company_from_signature(text)
            if signature_company:
                self.logger.debug(f"✓ Extracted company from signature: {signature_company}")
                return signature_company
            email = kwargs.get('email')
            if email:
                company = self.spacy_extractor.extract_company_from_domain(email)
                if company:
                    self.logger.debug(f"✓ Extracted company from domain: {company}")
                    return company
            entities = self.spacy_extractor.extract_entities(text)
            return entities.get('company')
        elif field == 'location':
            entities = self.spacy_extractor.extract_entities(text)
            return entities.get('location')
        return None

    def _extract_gliner(self, field: str, text: str, **kwargs) -> Optional[str]:
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
        try:
            local_part = email.split('@')[0]
            name = local_part.replace('.', ' ').replace('_', ' ').replace('-', ' ')
            name = ''.join(char if char.isalpha() or char.isspace() else ' ' for char in name)
            name = ' '.join(word.capitalize() for word in name.split() if len(word) > 1)
            if len(name.split()) >= 2 or len(name) >= 3:
                self.logger.debug(f"Extracted name from email: {name}")
                return name
        except Exception as e:
            self.logger.error(f"Error extracting name from email: {str(e)}")
        return None

    def _is_valid_name(self, name: str, source_email: str) -> bool:
        """
        Validate name using DB rules from job_automation_keywords table.
        100% DB-driven - NO hardcoded fallbacks.
        """
        if not name or len(name.strip()) < 2:
            return False
        
        name_lower = name.lower().strip()
        words = name.split()
        
        # Basic format validation (minimal hardcoded rules)
        if len(words) == 1:
            if len(name) < 3:
                return False
        if len(words) > 4:
            return False
        if any(c.isdigit() for c in name):
            return False
        if name.isupper() and len(name) > 5:
            return False
        
        # Must have proper capitalization (at least first letter of each word)
        if not any(word[0].isupper() for word in words if word):
            return False
        
        # Use DB rules via email_filter (100% DB-driven)
        if not self.email_filter or not hasattr(self.email_filter, '_check_name_against_db_rules'):
            self.logger.error("No email_filter available for name validation - rejecting")
            return False
        
        if not self.email_filter._check_name_against_db_rules(name_lower):
            return False
        
        # Check if name matches source email (likely not a real name)
        if source_email:
            email_local = source_email.split('@')[0].lower()
            name_parts = [w.lower() for w in words if len(w) >= 3]
            matches = sum(1 for part in name_parts if part in email_local)
            if matches >= 2:
                return False
        
        return True

    def _is_valid_company(self, company: str) -> bool:
        """
        Validate company using DB rules from job_automation_keywords table.
        100% DB-driven - NO hardcoded fallbacks.
        """
        if not company or len(company.strip()) < 2:
            return False
        
        company_lower = company.lower().strip()
        
        # Basic format validation (minimal hardcoded rules)
        if len(company) > 100:
            return False
        
        # Reject if it looks like a sentence (too many words)
        if len(company.split()) > 8:
            return False
        
        # Use DB rules via email_filter (100% DB-driven)
        if not self.email_filter or not hasattr(self.email_filter, '_check_company_against_db_rules'):
            self.logger.error("No email_filter available for company validation - rejecting")
            return False
        
        if not self.email_filter._check_company_against_db_rules(company_lower):
            return False
        
        return True

    def _is_valid_location_db(self, location: str) -> bool:
        """
        Validate location using DB rules from job_automation_keywords table.
        100% DB-driven - NO hardcoded fallbacks.
        """
        if not location or len(location.strip()) < 2:
            return False
        
        # Basic format validation (minimal hardcoded rules)
        if len(location) > 100:
            return False
        
        if len(location.split()) > 6:
            return False
        
        # Use DB rules via email_filter (100% DB-driven)
        if not self.email_filter or not hasattr(self.email_filter, '_is_valid_location'):
            self.logger.error("No email_filter available for location validation - rejecting")
            return False
        
        return self.email_filter._is_valid_location(location)

    def _is_valid_email(self, email: str) -> bool:
        """
        Validate email using DB rules from job_automation_keywords table.
        100% DB-driven - NO hardcoded fallbacks.
        """
        if not email or '@' not in email:
            return False
        
        if not self.email_filter:
            self.logger.error("No email_filter available for email validation - rejecting")
            return False
        
        return self.email_filter.is_email_allowed(email)

    def _is_valid_linkedin_id(self, value: str) -> bool:
        """
        Validate LinkedIn ID using DB rules from job_automation_keywords table.
        100% DB-driven - NO hardcoded fallbacks.
        """
        if not value:
            return False
        
        # Use DB rules via email_filter (100% DB-driven)
        if not self.email_filter or not hasattr(self.email_filter, '_is_valid_linkedin'):
            self.logger.error("No email_filter available for LinkedIn validation - rejecting")
            return False
        
        return self.email_filter._is_valid_linkedin(value)

    def _extract_from_cc_headers(self, email_message) -> Optional[str]:
        try:
            all_emails = set()
            to_header = email_message.get('To', '')
            if to_header:
                for addr in to_header.split(','):
                    _, email_addr = parseaddr(addr.strip())
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            all_emails.add(email_lower)
            cc_header = email_message.get('Cc', '')
            if cc_header:
                for addr in cc_header.split(','):
                    _, email_addr = parseaddr(addr.strip())
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower()
                        if self._is_valid_header_email(email_lower):
                            all_emails.add(email_lower)
            reply_to = email_message.get('Reply-To', '')
            if reply_to:
                _, email_addr = parseaddr(reply_to)
                if email_addr and '@' in email_addr:
                    email_lower = email_addr.lower()
                    if self._is_valid_header_email(email_lower):
                        return email_lower
            return list(all_emails)[0] if all_emails else None
        except Exception as e:
            self.logger.error(f"Error extracting header emails: {str(e)}")
            return None

    def _is_valid_phone(self, phone: str) -> bool:
        """
        Validate phone number using DB rules from job_automation_keywords table.
        100% DB-driven - NO hardcoded fallbacks.
        """
        if not phone:
            return False
        
        if not self.email_filter or not hasattr(self.email_filter, '_is_valid_phone'):
            self.logger.error("No email_filter available for phone validation - rejecting")
            return False
        
        return self.email_filter._is_valid_phone(phone)

    def _is_valid_header_email(self, email: str) -> bool:
        return self._is_valid_email(email)

    def _extract_reply_to_email(self, email_message) -> Optional[str]:
        """
        Extract Reply-To email using RFC-compliant parsing.
        Returns the first valid email address from Reply-To header.
        """
        try:
            reply_to = email_message.get('Reply-To', '')
            if reply_to:
                # Use getaddresses for proper RFC parsing
                addresses = getaddresses([reply_to])
                for name, email_addr in addresses:
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower().strip()
                        if self._is_valid_header_email(email_lower):
                            self.logger.debug(f"✓ Extracted Reply-To: {email_lower}")
                            return email_lower
        except Exception as e:
            self.logger.error(f"Error extracting Reply-To: {str(e)}", exc_info=True)
        return None

    def _extract_sender_email(self, email_message) -> Optional[str]:
        """
        Extract Sender email using RFC-compliant parsing.
        Returns the first valid email address from Sender header.
        """
        try:
            sender = email_message.get('Sender', '')
            if sender:
                # Use getaddresses for proper RFC parsing
                addresses = getaddresses([sender])
                for name, email_addr in addresses:
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower().strip()
                        if self._is_valid_header_email(email_lower):
                            self.logger.debug(f"✓ Extracted Sender: {email_lower}")
                            return email_lower
        except Exception as e:
            self.logger.error(f"Error extracting Sender: {str(e)}", exc_info=True)
        return None

    def _extract_from_header(self, email_message) -> Optional[str]:
        """
        Extract From email using RFC-compliant parsing.
        Returns the first valid email address from From header.
        """
        try:
            from_header = email_message.get('From', '')
            if from_header:
                # Use getaddresses for proper RFC parsing
                addresses = getaddresses([from_header])
                for name, email_addr in addresses:
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower().strip()
                        if self._is_valid_header_email(email_lower):
                            self.logger.debug(f"✓ Extracted From: {email_lower}")
                            return email_lower
        except Exception as e:
            self.logger.error(f"Error extracting From: {str(e)}", exc_info=True)
        return None

    def _extract_to_emails(self, email_message) -> List[str]:
        """
        Extract To emails using RFC-compliant parsing.
        Uses getaddresses() to properly handle multi-address headers.
        """
        emails = []
        try:
            to_header = email_message.get('To', '')
            if to_header:
                # getaddresses handles RFC-compliant formats with commas in names
                to_addresses = getaddresses([to_header])
                for name, email_addr in to_addresses:
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower().strip()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted To: {email_lower}")
        except Exception as e:
            self.logger.error(f"Error extracting To: {str(e)}", exc_info=True)
        return emails

    def _extract_cc_bcc_emails(self, email_message) -> List[str]:
        """
        Extract CC and BCC emails using RFC-compliant parsing.
        Uses getaddresses() to properly handle multi-address headers.
        """
        emails = []
        try:
            # Extract CC emails - use getaddresses for proper RFC parsing
            cc_header = email_message.get('Cc', '')
            if cc_header:
                # getaddresses handles RFC-compliant formats with commas in names
                cc_addresses = getaddresses([cc_header])
                for name, email_addr in cc_addresses:
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower().strip()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted CC: {email_lower}")
            
            # Extract BCC emails
            bcc_header = email_message.get('Bcc', '')
            if bcc_header:
                bcc_addresses = getaddresses([bcc_header])
                for name, email_addr in bcc_addresses:
                    if email_addr and '@' in email_addr:
                        email_lower = email_addr.lower().strip()
                        if self._is_valid_header_email(email_lower):
                            emails.append(email_lower)
                            self.logger.debug(f"✓ Extracted BCC: {email_lower}")
        except Exception as e:
            self.logger.error(f"Error extracting CC/BCC: {str(e)}", exc_info=True)
        return emails

    def _extract_name_from_header_for_email(self, email_message, email_addr: str) -> Optional[str]:
        """
        Extract name from email headers for a specific email address.
        Uses getaddresses() for proper RFC-compliant parsing of CC/BCC.
        """
        if not email_addr:
            return None
        email_lower = email_addr.lower()
        try:
            # Check Reply-To
            reply_to = email_message.get('Reply-To', '')
            if reply_to:
                addresses = getaddresses([reply_to])
                for name, addr in addresses:
                    if addr and addr.lower() == email_lower:
                        if name and name != addr:
                            self.logger.debug(f"✓ Extracted name from Reply-To: {name}")
                            return name.strip()
            
            # Check Sender
            sender = email_message.get('Sender', '')
            if sender:
                addresses = getaddresses([sender])
                for name, addr in addresses:
                    if addr and addr.lower() == email_lower:
                        if name and name != addr:
                            self.logger.debug(f"✓ Extracted name from Sender: {name}")
                            return name.strip()
            
            # Check From
            from_header = email_message.get('From', '')
            if from_header:
                addresses = getaddresses([from_header])
                for name, addr in addresses:
                    if addr and addr.lower() == email_lower:
                        if name and name != addr:
                            self.logger.debug(f"✓ Extracted name from From: {name}")
                            return name.strip()
            
            # Check CC - use getaddresses for proper parsing
            cc = email_message.get('Cc', '')
            if cc:
                addresses = getaddresses([cc])
                for name, addr in addresses:
                    if addr and addr.lower() == email_lower:
                        if name and name != addr:
                            self.logger.debug(f"✓ Extracted name from CC: {name}")
                            return name.strip()
            
            # Check BCC - use getaddresses for proper parsing
            bcc = email_message.get('Bcc', '')
            if bcc:
                addresses = getaddresses([bcc])
                for name, addr in addresses:
                    if addr and addr.lower() == email_lower:
                        if name and name != addr:
                            self.logger.debug(f"✓ Extracted name from BCC: {name}")
                            return name.strip()
        except Exception as e:
            self.logger.error(f"Error extracting name from header for {email_addr}: {str(e)}", exc_info=True)
        return None

    def _get_html_body(self, email_message) -> Optional[str]:
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
        """
        Extract emails from calendar invites.
        Extracts ORGANIZER and ATTENDEE emails from calendar parts,
        and falls back to headers if no calendar part found.
        """
        emails = set()
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/calendar":
                        payload = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        # ORGANIZER
                        for match in re.findall(r"ORGANIZER.*mailto:([^ \r\n]+)", payload, re.IGNORECASE):
                            email_lower = match.lower().strip()
                            if email_lower and '@' in email_lower:
                                emails.add(email_lower)
                        # ATTENDEE
                        for match in re.findall(r"ATTENDEE.*mailto:([^ \r\n]+)", payload, re.IGNORECASE):
                            email_lower = match.lower().strip()
                            if email_lower and '@' in email_lower:
                                emails.add(email_lower)
            
            # Fallback: extract from headers if no calendar part found
            if not emails:
                for header in ["Sender", "Reply-To", "From"]:
                    header_value = email_message.get(header, '')
                    if header_value:
                        addresses = getaddresses([header_value])
                        for name, addr in addresses:
                            if addr and '@' in addr and "noreply" not in addr.lower():
                                email_lower = addr.lower().strip()
                                emails.add(email_lower)
            
            return list(emails) if emails else None
        except Exception as e:
            self.logger.error(f"Error extracting calendar emails: {str(e)}", exc_info=True)
            return None
