from typing import Dict, List, Optional
import logging
import json

from ..connectors.http_api import APIClient
from ..filtering.repository import get_filter_repository
from ..core.database import get_db_client
from datetime import datetime

logger = logging.getLogger(__name__)


class VendorUtil:
    """Persist extracted vendor/recruiter contacts and related raw positions in bulk."""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        self.filter_repo = get_filter_repository()
        self.db_client = get_db_client()

    def get_existing_emails(self, source_email: str) -> set:
        """
        Fetch existing vendor emails for this candidate from DB to avoid re-inserting.
        """
        if not source_email:
            return set()
            
        try:
            query = "SELECT email FROM vendor_contact_extracts WHERE source_email = %s"
            results = self.db_client.execute_query(query, (source_email,))
            
            existing = set()
            for row in results:
                if row.get('email'):
                    existing.add(row['email'].strip().lower())
            
            self.logger.info(f"Loaded {len(existing)} existing contacts for deduplication")
            return existing
        except Exception as e:
            self.logger.error(f"Failed to fetch existing contacts: {e}")
            return set()


    def save_contacts(self, contacts: List[Dict], candidate_id: Optional[int] = None) -> Dict[str, int]:
        """
        Persist filtered contacts and raw positions in bulk.

        Returns:
            Dict with keys:
            - contacts_inserted
            - contacts_skipped
            - positions_inserted
            - positions_skipped
        """
        result = {
            "contacts_inserted": 0,
            "contacts_skipped": 0,
            "positions_inserted": 0,
            "positions_skipped": 0,
        }
        if not contacts:
            self.logger.info("No contacts to save")
            return result

        filtered_contacts = []
        seen_keys = set()
        for contact in contacts:
            if not self._is_valid_contact(contact):
                result["contacts_skipped"] += 1
                continue
            if not self._is_vendor_recruiter_contact(contact):
                result["contacts_skipped"] += 1
                continue

            email_key = (contact.get("email") or "").strip().lower()
            linkedin_key = (contact.get("linkedin_id") or "").strip().lower()
            dedupe_key = email_key or f"li:{linkedin_key}"
            if dedupe_key and dedupe_key in seen_keys:
                result["contacts_skipped"] += 1
                continue
            if dedupe_key:
                seen_keys.add(dedupe_key)

            filtered_contacts.append(contact)

        if not filtered_contacts:
            self.logger.info("No vendor/recruiter contacts after validation")
            return result

        bulk_contacts = self._build_vendor_contacts_payload(filtered_contacts)
        if not bulk_contacts:
            self.logger.info("No contacts prepared for vendor_contact bulk insert")
            return result

        try:
            self.logger.info("Sending %s contacts to /api/vendor_contact/bulk", len(bulk_contacts))
            response = self.api_client.post("/api/vendor_contact/bulk", {"contacts": bulk_contacts})
            inserted, skipped = self._extract_insert_skip_counts(response, default_inserted=len(bulk_contacts))
            result["contacts_inserted"] = inserted
            result["contacts_skipped"] += skipped
        except Exception as error:
            self.logger.error("API error saving vendor contacts: %s", error)
            return result

        if not candidate_id:
            return result

        raw_positions = self._build_raw_positions_payload(filtered_contacts, candidate_id)
        if not raw_positions:
            self.logger.info("No raw positions produced from filtered vendor contacts")
            return result

        try:
            self.logger.info("Sending %s raw positions to /api/raw-positions/bulk", len(raw_positions))
            response = self.api_client.post("/api/raw-positions/bulk", {"positions": raw_positions})
            inserted, skipped = self._extract_insert_skip_counts(response, default_inserted=len(raw_positions))
            result["positions_inserted"] = inserted
            result["positions_skipped"] = skipped
        except Exception as error:
            self.logger.error("Error saving raw positions: %s", error)

        return result

    def _build_vendor_contacts_payload(self, contacts: List[Dict]) -> List[Dict]:
        payload = []
        for contact in contacts:
            full_name = (contact.get("name") or "").strip()
            if not full_name:
                continue

            item = {
                "full_name": full_name,
                "source_email": contact.get("source"),
                "email": contact.get("email"),
                "phone": contact.get("phone"),
                "linkedin_id": contact.get("linkedin_id"),
                "company_name": contact.get("company"),
                "location": contact.get("location"),
                "extraction_date": datetime.now().date().isoformat(),
                "job_source": "Bot Candidate Email Extractor"
            }
            item = {key: value for key, value in item.items() if value not in (None, "")}
            if "full_name" in item:
                payload.append(item)
        return payload

    def _build_raw_positions_payload(self, contacts: List[Dict], candidate_id: int) -> List[Dict]:
        payload = []
        for contact in contacts:
            # Only generate a raw position for contacts that carry job content.
            has_position_signal = any(
                contact.get(field) for field in ("job_position", "raw_body", "location", "company")
            )
            if not has_position_signal:
                continue

            contact_info = {
                "name": contact.get("name"),
                "email": contact.get("email"),
                "phone": contact.get("phone"),
                "linkedin": contact.get("linkedin_id"),
            }
            payload.append(
                {
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
                    "raw_payload": contact,
                    "processing_status": "new",
                }
            )
        return payload

    def _extract_insert_skip_counts(self, response: Dict, default_inserted: int) -> tuple:
        if isinstance(response, dict):
            inserted = int(response.get("inserted", response.get("saved", default_inserted)) or 0)
            skipped = int(response.get("skipped", 0) or 0)
            return inserted, skipped
        return default_inserted, 0

    def _is_vendor_recruiter_contact(self, contact: Dict) -> bool:
        source_email = (contact.get("source") or "").strip().lower()
        contact_email = (contact.get("email") or "").strip().lower()
        if source_email and contact_email and source_email == contact_email:
            return False

        if contact_email:
            action = self.filter_repo.check_email(contact_email)
            if action == "block":
                return False
        return True

    def _is_valid_contact(self, contact: Dict) -> bool:
        """Validate contact has minimum required quality."""
        try:
            email = (contact.get("email") or "").strip()
            linkedin = (contact.get("linkedin_id") or "").strip()
            name = (contact.get("name") or "").strip()

            if not email and not linkedin:
                return False

            if email:
                if "@" not in email or "." not in email:
                    return False
                email_lower = email.lower()
                blocked_local_parts = ["noreply", "no-reply", "info@", "support@", "admin@"]
                if any(token in email_lower for token in blocked_local_parts):
                    return False

            if linkedin and (" " in linkedin or len(linkedin) > 80):
                return False

            if name:
                words = name.split()
                if len(words) < 2 or len(words) > 6:
                    return False
                if any(char.isdigit() for char in name):
                    return False
            return True
        except Exception as error:
            self.logger.error("Error validating contact: %s", error)
            return False
