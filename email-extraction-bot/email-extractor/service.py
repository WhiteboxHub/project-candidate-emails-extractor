#!/usr/bin/env python3
"""
Email Contact Extractor Service

Main orchestrator for extracting vendor contacts from candidate email inboxes.
Config-driven extraction pipeline with multiple fallback methods.
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
    
    def __init__(self, config_path='config/config.yaml'):
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
            'host': self.config['database']['host'],
            'port': self.config['database']['port'],
            'user': self.config['database']['user'],
            'password': self.config['database']['password'],
            'database': self.config['database']['database']
        }
        
        self.candidate_util = CandidateUtil(db_config)
        self.vendor_util = VendorUtil(db_config)
        # Employee ID 353 for job_id 25 ('Bot Candidate Email Extractor')
        self.job_activity_log_util = JobActivityLogUtil(db_config, employee_id=353, job_id=25)
        
        self.cleaner = EmailCleaner()
        self.extractor = ContactExtractor(self.config)
        self.email_filter = EmailFilter(self.config)
        
        # Initialize UID tracker
        self.uid_tracker = get_uid_tracker('last_run.json')
        
        self.logger.info("All components initialized successfully")
    
    def run(self):
        """Main execution method"""
        try:
            # Get candidates with email credentials
            candidates = self.candidate_util.get_active_candidates()
            
            if not candidates:
                self.logger.warning("No candidates found with email credentials")
                return
            
            self.logger.info(f"Processing {len(candidates)} candidate accounts")
            
            # Process each candidate
            total_contacts = 0
            for candidate in candidates:
                try:
                    contacts_extracted = self.process_candidate(candidate)
                    total_contacts += contacts_extracted
                except Exception as e:
                    self.logger.error(f"Failed to process candidate {candidate['email']}: {str(e)}")
                    continue
            
            self.logger.info("=" * 80)
            self.logger.info(f"Extraction Complete - Total Contacts: {total_contacts}")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Service execution failed: {str(e)}", exc_info=True)
    
    def process_candidate(self, candidate: dict) -> int:
        """
        Process a single candidate's inbox
        
        Args:
            candidate: Candidate dictionary from database
            
        Returns:
            Number of contacts extracted
        """
        email = candidate['email']
        password = candidate['imap_password']
        candidate_name = candidate.get('name', email)
        self.logger.info(f"\n{'─' * 80}")
        self.logger.info(f"Processing: {email}")
        self.logger.info(f"{'─' * 80}")
        
        # Connect to email account (server/port are hardcoded in GmailIMAPConnector)
        connector = GmailIMAPConnector(
            email=email,
            password=password
        )
        
        if not connector.connect():
            self.logger.error(f"Failed to connect to {email}")
            return 0
        
        try:
            reader = EmailReader(connector)
            batch_size = self.config['email']['batch_size']
            
            # Load last processed UID from tracker
            last_uid = self.uid_tracker.get_last_uid(email)
            
            start_index = 0
            total_contacts = []
            total_emails_processed = 0
            
            while True:
                # Fetch email batch
                emails, next_start_index = reader.fetch_emails(
                    since_uid=last_uid,
                    batch_size=batch_size,
                    start_index=start_index
                )
                
                if not emails:
                    break
                
                self.logger.info(f"Fetched {len(emails)} emails")
                
                # Filter emails
                filtered_emails = self.email_filter.filter_emails(emails, self.cleaner)
                self.logger.info(f"Filtered to {len(filtered_emails)} relevant emails")
                
                # Extract contacts
                for email_data in filtered_emails:
                    try:
                        clean_body = email_data.get('clean_body', 
                                                    self.cleaner.extract_body(email_data['message']))
                        
                        # extract_contacts now returns a LIST of contacts
                        contacts_list = self.extractor.extract_contacts(
                            email_data['message'],
                            clean_body,
                            source_email=email
                        )
                        
                        # Add all valid contacts to the total list
                        for contact in contacts_list:
                            # Only save if we have email or linkedin
                            if contact.get('email') or contact.get('linkedin_id'):
                                total_contacts.append(contact)
                                self.logger.debug(f"Extracted contact: {contact.get('email', 'N/A')} from {contact.get('extraction_source', 'unknown')}")
                            
                    except Exception as e:
                        self.logger.error(f"Error extracting from email UID {email_data['uid']}: {str(e)}")
                        continue
                
                # Update last processed UID in tracker
                if emails:
                    max_uid = max(int(e['uid']) for e in emails)
                    self.uid_tracker.update_last_uid(email, str(max_uid))
                
                total_emails_processed += len(filtered_emails)
                
                # Continue to next batch
                if not next_start_index:
                    break
                start_index = next_start_index
            
            # Save all contacts
            contacts_saved = 0
            if total_contacts:
                contacts_saved = self.vendor_util.save_contacts(total_contacts)
            
            # Log activity to job_activity_log 
            if contacts_saved > 0:
                candidate_id = candidate['candidate_id']
                self.job_activity_log_util.log_activity(
                    candidate_id=candidate_id,
                    contacts_extracted=contacts_saved
                )
            else:
                self.logger.warning(f"No candidate_id for {email} - skipping activity log")
            
            self.logger.info(f"✓ Completed {email}: {contacts_saved} contacts saved from {total_emails_processed} emails")
            return contacts_saved
            
        finally:
            connector.disconnect()
    
    # UID tracking now handled by uid_tracker utility


def main():
    """Main entry point"""
    try:
        service = EmailExtractionService()
        service.run()
    except KeyboardInterrupt:
        print("\n\nService interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
