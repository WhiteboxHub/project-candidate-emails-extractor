from typing import List, Dict
import logging
from utils.api_client import APIClient

logger = logging.getLogger(__name__)

class VendorUtil:
    """
    Manage vendor_contact_extracts via API
    
    Note: This requires a vendor contacts API endpoint to be added to the backend.
    Expected endpoints:
    - POST /vendor/contacts - Create/update vendor contacts
    - GET /vendor/contacts - List vendor contacts
    - GET /vendor/contacts/{id} - Get specific contact
    """
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
    
    def save_contacts(self, contacts: List[Dict]) -> int:
        """
        Save extracted contacts via API with advanced deduplication
        
        Args:
            contacts: List of contact dictionaries
            
        Returns:
            Number of new contacts inserted
        """
        if not contacts:
            self.logger.info("No contacts to save")
            return 0
        
        # Pre-filter contacts for quality
        valid_contacts = []
        for contact in contacts:
            if self._is_valid_contact(contact):
                valid_contacts.append(contact)
            else:
                self.logger.debug(f"Skipped invalid contact: {contact.get('email', 'N/A')}")
        
        if not valid_contacts:
            self.logger.info("No valid contacts after filtering")
            return 0
        
        try:
            # POST /api/vendor_contact (individual contact creation)
            # The API expects individual contacts, so we'll save them one by one
            saved_count = 0
            
            for contact in valid_contacts:
                try:
                    # Ensure we have a full_name (required field)
                    full_name = contact.get('name', '')
                    if not full_name or len(full_name.strip()) == 0:
                        # Skip contacts without a name (required field)
                        self.logger.debug(f"Skipping contact without name: {contact.get('email')}")
                        continue
                    
                    # Map contact fields to API format (VendorContactExtractCreate schema)
                    contact_data = {
                        "full_name": full_name,  # Required field
                        "source_email": contact.get('source'),
                        "email": contact.get('email'),
                        "phone": contact.get('phone'),
                        "linkedin_id": contact.get('linkedin_id'),
                        "company_name": contact.get('company'),
                        "location": contact.get('location')
                        # Note: moved_to_vendor is NOT in Create schema, only in Update
                        # Note: extraction_date is auto-set by the API
                    }
                    
                    # Remove None values to avoid sending null fields
                    contact_data = {k: v for k, v in contact_data.items() if v is not None and v != ''}
                    
                    # Ensure full_name is still present after filtering
                    if 'full_name' not in contact_data:
                        self.logger.debug(f"Skipping contact - full_name removed after filtering: {contact.get('email')}")
                        continue
                    
                    # Log the data being sent for debugging
                    self.logger.debug(f"Sending contact data: {contact_data}")
                    
                    response = self.api_client.post('/api/vendor_contact', contact_data)
                    saved_count += 1
                    self.logger.debug(f"Saved contact: {contact.get('email', contact.get('linkedin_id'))}")
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check if it's a duplicate error from the API
                    if "duplicate entry" in error_msg or "integrity error" in error_msg:
                        self.logger.warning(f"Skipping duplicate contact: {contact.get('email', contact.get('linkedin_id'))}")
                    else:
                        self.logger.error(f"Failed to save contact {contact.get('email')}: {str(e)}")
                        # Log the full contact data that failed
                        self.logger.error(f"Failed contact data: {contact}")
                    continue
            
            self.logger.info(f"Saved {saved_count} new contacts via API")
            return saved_count
            
        except Exception as e:
            self.logger.error(f"API error saving contacts: {str(e)}")
            return 0
    
    def _is_valid_contact(self, contact: Dict) -> bool:
        """Validate contact has minimum required quality"""
        try:
            email = contact.get('email', '')
            linkedin = contact.get('linkedin_id', '')
            name = contact.get('name', '')
            
            # Must have email OR linkedin
            if not email and not linkedin:
                return False
            
            # If has email, validate format
            if email:
                if '@' not in email or '.' not in email:
                    return False
                # Skip generic/automated emails
                email_lower = email.lower()
                if any(x in email_lower for x in ['noreply', 'no-reply', 'info@', 'support@', 'admin@']):
                    return False
            
            # If has linkedin, validate it's not a name
            if linkedin:
                if ' ' in linkedin or len(linkedin) > 50:
                    return False
            
            # If has name, basic validation
            if name:
                # Skip single-word names or too long
                words = name.split()
                if len(words) < 2 or len(words) > 4:
                    return False
                # Skip names with numbers
                if any(c.isdigit() for c in name):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating contact: {str(e)}")
            return False
