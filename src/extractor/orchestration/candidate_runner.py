import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..connectors.imap_gmail import GmailIMAPConnector
from ..email.reader import EmailReader


@dataclass
class CandidateRunResult:
    candidate_id: Optional[int]
    email: str
    status: str
    candidate_name: Optional[str] = None
    contacts_saved: int = 0
    positions_saved: int = 0
    contacts_deduplicated: int = 0  # Track efficiency
    emails_fetched: int = 0
    last_uid: Optional[str] = None
    duplicates_skipped: int = 0
    non_vendor_filtered: int = 0
    filter_stats: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    def to_metadata(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "candidate_email": self.email,
            "status": self.status,
            "emails_fetched": self.emails_fetched,
            "last_uid": self.last_uid,
            "duplicates_skipped": self.duplicates_skipped,
            "non_vendor_filtered": self.non_vendor_filtered,
            "emails_inserted": self.contacts_saved,  # Alias for clarity in report
            "contacts_saved": self.contacts_saved,
            "positions_saved": self.positions_saved,
            "filter_stats": self.filter_stats,
            "error": self.error,
        }


class CandidateRunner:
    """Process a single candidate inbox end-to-end."""

    def __init__(
        self,
        config,
        cleaner,
        extractor,
        email_filter,
        uid_tracker,
        vendor_util,
        connector_cls=GmailIMAPConnector,
        reader_cls=EmailReader,
        deduplication_cache=None,
    ):
        self.config = config
        self.cleaner = cleaner
        self.extractor = extractor
        self.email_filter = email_filter
        self.uid_tracker = uid_tracker
        self.vendor_util = vendor_util
        self.connector_cls = connector_cls
        self.reader_cls = reader_cls
        self.deduplication_cache = deduplication_cache
        self.logger = logging.getLogger(__name__)

    def run(self, candidate: Dict) -> CandidateRunResult:
        email = (candidate.get("email") or "").strip()
        password = candidate.get("imap_password")
        candidate_id = candidate.get("candidate_id") or candidate.get("id")
        candidate_name = candidate.get("name") or candidate.get("full_name")  # Extract name from candidate dict

        if not email:
            return CandidateRunResult(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                email="",
                status="failed",
                error="Missing candidate email",
            )
        if not password:
            return CandidateRunResult(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                email=email,
                status="failed",
                error="Missing candidate IMAP password",
            )

        filter_stats = {
            "total": 0,
            "passed": 0,
            "junk": 0,
            "not_recruiter": 0,
            "calendar_invites": 0,
        }

        connector = self.connector_cls(email=email, password=password)
        if not connector.connect():
            return CandidateRunResult(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                email=email,
                status="failed",
                error="Authentication failed - unable to connect to IMAP server",
                filter_stats=filter_stats,
            )

        # Smart Deduplication: Pre-fetch existing contacts
        seen_emails = set()
        if self.vendor_util:
            self.logger.info(f"Fetching existing contacts for {email} to enable smart deduplication...")
            seen_emails = self.vendor_util.get_existing_emails(email)
            
        emails_fetched = 0
        deduplicated_count = 0
        extracted_contacts: List[Dict] = []
        last_processed_uid = None  # Track the last UID processed

        try:
            reader = self.reader_cls(connector)
            batch_size = int(self.config.get("email", {}).get("batch_size", 100))
            last_uid = self.uid_tracker.get_last_uid(email)
            start_index = 0

            # Track the highest UID seen so far to prevent overwriting with lower UIDs from older batches
            high_water_mark_uid = 0
            if last_uid and str(last_uid).isdigit():
                high_water_mark_uid = int(last_uid)

            while True:
                emails, next_start_index = reader.fetch_emails(
                    since_uid=last_uid,
                    batch_size=batch_size,
                    start_index=start_index,
                )
                if not emails:
                    break

                emails_fetched += len(emails)
                filtered_emails, batch_stats = self.email_filter.filter_emails(emails, self.cleaner)
                for key in filter_stats:
                    filter_stats[key] += int(batch_stats.get(key, 0))

                for email_data in filtered_emails:
                    try:
                        message = email_data["message"]
                        clean_body = email_data.get("clean_body") or self.cleaner.extract_body(message)
                        contacts = self.extractor.extract_contacts(
                            message,
                            clean_body,
                            source_email=email,
                            subject=message.get("Subject", ""),
                        )
                        for contact in contacts:
                            if not (contact.get("email") or contact.get("linkedin_id")):
                                continue
                            
                            # Deduplicate against DB cache
                            contact_email = (contact.get("email") or "").strip().lower()
                            if contact_email and contact_email in seen_emails:
                                self.logger.debug(f"Skipping duplicate contact found in DB: {contact_email}")
                                deduplicated_count += 1
                                continue
                            
                            # Add to local cache to prevent duplicates within this run too
                            if contact_email:
                                seen_emails.add(contact_email)

                            contact["raw_body"] = clean_body
                            contact["extracted_from_uid"] = email_data.get("uid")
                            
                            # Check global run cache (Intra-run deduplication)
                            if self.deduplication_cache and self.deduplication_cache.is_seen_in_run(contact_email):
                                deduplicated_count += 1
                                self.logger.info(f"Skipping intra-run duplicate: {contact_email}")
                                continue

                            extracted_contacts.append(contact)
                            
                            # Mark as seen in run cache
                            if self.deduplication_cache:
                                self.deduplication_cache.mark_seen_in_run(contact_email)
                    except Exception as extraction_error:
                        self.logger.error(
                            "Error extracting candidate_id=%s email=%s uid=%s: %s",
                            candidate_id,
                            email,
                            email_data.get("uid"),
                            extraction_error,
                        )

                if emails:
                    batch_max_uid = max(int(item["uid"]) for item in emails)
                    # Only update if we found a strictly higher UID than we've ever seen/stored
                    if batch_max_uid > high_water_mark_uid:
                        high_water_mark_uid = batch_max_uid
                        self.uid_tracker.update_last_uid(email, str(high_water_mark_uid))
                        last_processed_uid = str(high_water_mark_uid)

                if next_start_index is None:
                    break
                start_index = next_start_index

            # Detailed Efficiency Log - "Great Logging"
            self.logger.info("=" * 60)
            self.logger.info(f"PROCESSING SUMMARY for {email}:")
            self.logger.info(f"  - Emails Fetched:       {emails_fetched}")
            self.logger.info(f"  - Existing Contacts:    {len(seen_emails)} (Cached from DB)")
            self.logger.info(f"  - New Deduplicates:     {deduplicated_count} (Skipped locally)")
            self.logger.info(f"  - Unique to Save:       {len(extracted_contacts)}")
            self.logger.info("=" * 60)

            # Calculate non-vendor filtered count
            non_vendor_count = filter_stats.get("junk", 0) + filter_stats.get("not_recruiter", 0)
            
            if not self.vendor_util:
                return CandidateRunResult(
                    candidate_id=candidate_id,
                    candidate_name=candidate_name,
                    email=email,
                    status="success",
                    contacts_saved=len(extracted_contacts),
                    positions_saved=0,
                    contacts_deduplicated=deduplicated_count,
                    emails_fetched=emails_fetched,
                    last_uid=last_processed_uid,
                    duplicates_skipped=deduplicated_count,
                    non_vendor_filtered=non_vendor_count,
                    filter_stats=filter_stats,
                )

            save_result = self.vendor_util.save_contacts(extracted_contacts, candidate_id=candidate_id)
            if not isinstance(save_result, dict):
                save_result = {"contacts_inserted": int(save_result), "positions_inserted": 0}
            return CandidateRunResult(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                email=email,
                status="success",
                contacts_saved=save_result.get("contacts_inserted", 0),
                positions_saved=save_result.get("positions_inserted", 0),
                contacts_deduplicated=deduplicated_count,
                emails_fetched=emails_fetched,
                last_uid=last_processed_uid,
                duplicates_skipped=deduplicated_count,
                non_vendor_filtered=non_vendor_count,
                filter_stats=filter_stats,
            )
        except Exception as processing_error:
            self.logger.error(
                "Candidate processing failed candidate_id=%s email=%s: %s",
                candidate_id,
                email,
                processing_error,
                exc_info=True,
            )
            non_vendor_count = filter_stats.get("junk", 0) + filter_stats.get("not_recruiter", 0)
            return CandidateRunResult(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                email=email,
                status="failed",
                error=str(processing_error),
                emails_fetched=emails_fetched,
                last_uid=last_processed_uid,
                duplicates_skipped=deduplicated_count,
                non_vendor_filtered=non_vendor_count,
                filter_stats=filter_stats,
            )
        finally:
            connector.disconnect()
