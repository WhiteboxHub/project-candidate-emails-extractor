from typing import List, Dict
import logging
import json
from ..connectors.http_api import APIClient

logger = logging.getLogger(__name__)


class VendorUtil:
    """
    Manage vendor_contact_extracts via API

    Uses SAME API endpoint:
    POST /api/vendor_contact

    But sends BULK payload (list of contacts)
    """

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)

    def save_contacts(self, contacts: List[Dict], candidate_id: int = None) -> int:
        """
        Save extracted contacts via API using BULK insert
        (same API signature, single request)

        Args:
            contacts: List of contact dictionaries

        Returns:
            Number of new contacts inserted
        """
        if not contacts:
            self.logger.info("No contacts to save")
            return 0

        # --------------------------------------------------
        # Step 1: Pre-filter contacts for quality
        # --------------------------------------------------
        valid_contacts = []
        for contact in contacts:
            if self._is_valid_contact(contact):
                valid_contacts.append(contact)
            else:
                self.logger.debug(
                    f"Skipped invalid contact: {contact.get('email', 'N/A')}"
                )

        if not valid_contacts:
            self.logger.info("No valid contacts after filtering")
            return 0

        # --------------------------------------------------
        # Step 2: Prepare BULK payload
        # --------------------------------------------------
        bulk_contacts = []

        for contact in valid_contacts:
            full_name = (contact.get("name") or "").strip()
            if not full_name:
                self.logger.debug(
                    f"Skipping contact without name: {contact.get('email')}"
                )
                continue

            contact_data = {
                "full_name": full_name,  # Required field
                "source_email": contact.get("source"),
                "email": contact.get("email"),
                "phone": contact.get("phone"),
                "linkedin_id": contact.get("linkedin_id"),
                "company_name": contact.get("company"),
                "location": contact.get("location"),
            }

            # Remove None / empty values
            contact_data = {
                k: v for k, v in contact_data.items()
                if v is not None and v != ""
            }

            # Ensure required field still exists
            if "full_name" not in contact_data:
                self.logger.debug(
                    f"Skipping contact - full_name missing after filtering: {contact.get('email')}"
                )
                continue

            bulk_contacts.append(contact_data)

        if not bulk_contacts:
            self.logger.info("No contacts prepared for bulk insert")
            return 0

        # --------------------------------------------------
        # Step 3: SINGLE API CALL (same endpoint)
        # --------------------------------------------------
        try:
            self.logger.info(
                f"Sending {len(bulk_contacts)} contacts to /api/vendor_contact"
            )

            # IMPORTANT: sending Wrapped Dict, matching VendorContactBulkCreate
            response = self.api_client.post(
                "/api/vendor_contact/bulk",
                {"contacts": bulk_contacts}
            )

            # Flexible response handling
            if isinstance(response, dict):
                inserted = response.get("inserted", len(bulk_contacts))
                skipped = response.get("skipped", 0)
            else:
                inserted = len(bulk_contacts)
                skipped = 0

            self.logger.info(
                f"Bulk insert completed | Inserted: {inserted}, Skipped: {skipped}"
            )
            
            # --------------------------------------------------
            # Step 4: Save to raw_position table (SEPARATE API CALL)
            # --------------------------------------------------
            if candidate_id:
                try:
                    bulk_raw_positions = []
                    for contact in valid_contacts:
                        # Construct raw_contact_info as JSON string
                        contact_info = {
                            "name": contact.get("name"),
                            "email": contact.get("email"),
                            "phone": contact.get("phone"),
                            "linkedin": contact.get("linkedin_id")
                        }
                        
                        # Build raw_position payload matching the API schema
                        raw_payload = {
                            "candidate_id": candidate_id,
                            "source": "email",
                            "source_uid": contact.get("extracted_from_uid"),
                            "extractor_version": "v2.0",
                            "raw_title": contact.get("job_position"),
                            "raw_company": contact.get("company"),
                            "raw_location": contact.get("location"),
                            "raw_zip": contact.get("zip_code"),
                            "raw_description": contact.get("raw_body"),
                            "raw_contact_info": json.dumps(contact_info),
                            "raw_notes": f"Extracted from {contact.get('extraction_source')}",
                            "raw_payload": contact,  # Send as dict, schema handles it
                            "processing_status": "new"
                        }
                        bulk_raw_positions.append(raw_payload)
                    
                    if bulk_raw_positions:
                        self.logger.info(f"Sending {len(bulk_raw_positions)} raw positions to /api/raw_position")
                        
                        # POST to /api/raw-positions/bulk
                        response_raw = self.api_client.post(
                            "/api/raw-positions/bulk",
                            {"positions": bulk_raw_positions}
                        )
                        
                        if isinstance(response_raw, dict):
                            inserted_positions = response_raw.get("inserted", 0)
                            skipped_positions = response_raw.get("skipped", 0)
                            self.logger.info(f"Raw positions saved: {inserted_positions} inserted, {skipped_positions} skipped")
                        else:
                            self.logger.info("Raw positions saved successfully")
                        
                except Exception as e:
                    self.logger.error(f"Error saving raw positions: {str(e)}")
                    # Do not fail the whole batch if raw position save fails
                    # Vendor contacts are already saved at this point

            return inserted

        except Exception as e:
            self.logger.error(f"API error saving contacts: {str(e)}")
            return 0

    def _is_valid_contact(self, contact: Dict) -> bool:
        """Validate contact has minimum required quality"""
        try:
            email = contact.get("email", "")
            linkedin = contact.get("linkedin_id", "")
            name = contact.get("name", "")

            # Must have email OR linkedin
            if not email and not linkedin:
                return False

            # Email validation
            if email:
                if "@" not in email or "." not in email:
                    return False
                email_lower = email.lower()
                if any(x in email_lower for x in [
                    "noreply", "no-reply", "info@", "support@", "admin@"
                ]):
                    return False

            # LinkedIn validation
            if linkedin:
                if " " in linkedin or len(linkedin) > 50:
                    return False

            # Name validation
            if name:
                words = name.split()
                if len(words) < 2 or len(words) > 4:
                    return False
                if any(c.isdigit() for c in name):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating contact: {str(e)}")
            return False
