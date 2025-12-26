#!/usr/bin/env python3
"""
Email Contact Extractor Service

Main orchestrator for extracting vendor contacts from candidate email inboxes.
DB-driven filtering + scoring + extraction pipeline.
"""

import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config
from utils.logging.logger import get_logger
from utils.email.connectors import GmailIMAPConnector
from utils.email.reader import EmailReader
from utils.preprocessor.cleaner import EmailCleaner
from utils.extraction.extractor import ContactExtractor
from utils.filters.email_filters import EmailFilter
from utils.avatar.candidate_util import CandidateUtil
from utils.avatar.vendor_util import VendorUtil
from utils.avatar.job_activity_log_util import JobActivityLogUtil
from utils.uid_tracker import get_uid_tracker


class EmailExtractionService:
    """Main service for email contact extraction"""

    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        self.config_loader = get_config()
        self.config_loader.config_path = Path(config_path)
        self.config = self.config_loader.load()

        # Setup logging
        self.logger = get_logger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("Email Contact Extractor Service Started")
        self.logger.info("=" * 80)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all service components"""
        db_config = {
            "host": self.config["database"]["host"],
            "port": self.config["database"]["port"],
            "user": self.config["database"]["user"],
            "password": self.config["database"]["password"],
            "database": self.config["database"]["database"],
        }

        self.candidate_util = CandidateUtil(db_config)
        self.vendor_util = VendorUtil(db_config)

        # Fixed employee/job IDs for audit trail
        self.job_activity_log_util = JobActivityLogUtil(
            db_config,
            employee_id=353,
            job_id=25
        )

        self.cleaner = EmailCleaner()
        
        # EmailFilter MUST be DB-driven and initialized first
        self.email_filter = EmailFilter(self.config, db_config)
        
        # Pass EmailFilter to ContactExtractor
        self.extractor = ContactExtractor(self.config, email_filter=self.email_filter)

        # UID tracker
        self.uid_tracker = get_uid_tracker("last_run.json")

        self.logger.info("All components initialized successfully")
        
        # Display cache statistics banner
        self._display_cache_stats()
    
    def _display_cache_stats(self):
        """Display cache statistics banner at startup"""
        try:
            stats = self.email_filter.cache_manager.get_cache_stats()
            
            self.logger.info("=" * 80)
            self.logger.info("DATABASE CACHE STATISTICS")
            self.logger.info("=" * 80)
            self.logger.info(f"Cache Status: {'FRESH' if stats['is_fresh'] else 'STALE'}")
            self.logger.info(f"Cache Age: {stats['cache_age_seconds']:.1f}s / {stats['cache_ttl_seconds']}s TTL")
            self.logger.info(f"Load Count: {stats['load_count']} | Hit Rate: {stats['hit_rate']:.1%}")
            self.logger.info("-" * 80)
            self.logger.info("LOADED RULES:")
            self.logger.info(f"  • Sender/Email Rules: {stats['rule_counts']['sender_rules']}")
            self.logger.info(f"  • Content Scoring Rules: {stats['rule_counts']['content_rules']}")
            self.logger.info(f"  • Name Validation Rules: {stats['rule_counts']['name_rules']}")
            self.logger.info(f"  • Company Validation Rules: {stats['rule_counts']['company_rules']}")
            self.logger.info(f"  • TOTAL RULES: {stats['rule_counts']['total_rules']}")
            self.logger.info("=" * 80)
            self.logger.info("✓ All filtering is 100% DATABASE-DRIVEN (no hardcoded rules)")
            self.logger.info("=" * 80)
        except Exception as e:
            self.logger.warning(f"Could not display cache stats: {e}")


    def run(self):
        """Main execution method"""
        try:
            candidates = self.candidate_util.get_active_candidates()

            if not candidates:
                self.logger.warning("No candidates found with email credentials")
                return

            self.logger.info(f"Processing {len(candidates)} candidate inboxes")

            total_contacts = 0
            for candidate in candidates:
                try:
                    total_contacts += self.process_candidate(candidate)
                except Exception as e:
                    self.logger.error(
                        f"Failed processing candidate {candidate.get('email')}: {e}",
                        exc_info=True,
                    )

            self.logger.info("=" * 80)
            self.logger.info(f"Extraction Complete — Total Contacts: {total_contacts}")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error("Service execution failed", exc_info=True)

    def process_candidate(self, candidate: dict) -> int:
        """
        Process a single candidate inbox using:

        Phase 1: Sender allow/block (DB rules)
        Phase 2: Cleaning
        Phase 3: Content scoring (DB keywords)
        Phase 4: Extraction
        Phase 5: Save + audit
        """

        email = candidate["email"]
        password = candidate["imap_password"]

        self.logger.info(f"\n{'─' * 80}")
        self.logger.info(f"Processing inbox: {email}")
        self.logger.info(f"{'─' * 80}")

        connector = GmailIMAPConnector(email=email, password=password)

        if not connector.connect():
            self.logger.error(f"Failed to connect to {email}")
            return 0

        try:
            reader = EmailReader(connector)
            batch_size = self.config["email"]["batch_size"]

            last_uid = self.uid_tracker.get_last_uid(email)
            start_index = 0

            total_contacts = []
            processed_emails = 0

            while True:
                emails, next_start_index = reader.fetch_emails(
                    since_uid=last_uid,
                    batch_size=batch_size,
                    start_index=start_index,
                )

                if not emails:
                    break

                self.logger.info(f"Fetched {len(emails)} emails")

                for email_data in emails:
                    uid = email_data.get("uid")
                    msg = email_data["message"]

                    try:
                        # =====================================================
                        # PHASE 1: SENDER FILTER (DB PRIORITY RULES)
                        # =====================================================
                        sender_status = self.email_filter.check_sender(msg)
                        if sender_status != "allow":
                            continue

                        # =====================================================
                        # PHASE 2: CLEAN EMAIL CONTENT
                        # =====================================================
                        raw_subject = msg.get("Subject", "")
                        raw_body = self.cleaner.extract_body(msg)

                        clean_subject = self.cleaner.clean_text(raw_subject)
                        clean_body = self.cleaner.clean_text(raw_body)

                        if not clean_body or len(clean_body) < 50:
                            continue

                        # =====================================================
                        # PHASE 3: CONTACT EXTRACTION
                        # Extract ALL contacts from emails that passed sender filter
                        # =====================================================
                        contacts = self.extractor.extract_contacts(
                            msg,
                            clean_body,
                            source_email=email,
                        )

                        if not contacts:
                            continue

                        for contact in contacts:
                            if not (contact.get("email") or contact.get("linkedin_id")):
                                continue

                            # Validate extracted email against DB rules
                            contact_email = contact.get("email")
                            if contact_email:
                                if not self.email_filter.is_email_allowed(contact_email):
                                    self.logger.debug(
                                        f"Blocked extracted email: {contact_email} "
                                        f"(source: {contact.get('extraction_source', 'unknown')})"
                                    )
                                    continue

                            total_contacts.append(contact)

                        processed_emails += 1

                    except Exception as e:
                        self.logger.error(
                            f"Error processing UID {uid}: {e}",
                            exc_info=True,
                        )

                # Update UID tracker
                max_uid = max(int(e["uid"]) for e in emails)
                self.uid_tracker.update_last_uid(email, str(max_uid))

                if not next_start_index:
                    break
                start_index = next_start_index

            # =====================================================
            # SAVE CONTACTS + AUDIT
            # =====================================================
            contacts_saved = 0
            if total_contacts:
                contacts_saved = self.vendor_util.save_contacts(total_contacts)

            if contacts_saved > 0:
                self.job_activity_log_util.log_activity(
                    candidate_id=candidate["candidate_id"],
                    contacts_extracted=contacts_saved,
                )

            self.logger.info(
                f"✓ Completed {email}: {contacts_saved} contacts "
                f"from {processed_emails} recruiter emails"
            )

            return contacts_saved

        finally:
            connector.disconnect()


def main():
    """Main entry point"""
    try:
        service = EmailExtractionService()
        service.run()
    except KeyboardInterrupt:
        print("\nService interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error("Fatal error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
