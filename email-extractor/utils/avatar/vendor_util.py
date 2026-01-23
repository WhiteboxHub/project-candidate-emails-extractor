from typing import List, Dict
import logging
from utils.api_client import APIClient

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

    def save_contacts(self, contacts: List[Dict]) -> int:
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
            full_name = contact.get("name", "").strip()
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

            # IMPORTANT: sending LIST, not dict
            response = self.api_client.post(
                "/api/vendor_contact",
                bulk_contacts
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
