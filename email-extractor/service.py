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
        # Initialize API client from environment variables
        from utils.api_client import get_api_client
        
        try:
            self.api_client = get_api_client()
            self.logger.info("API client initialized successfully")
        except ValueError as e:
            self.logger.error(f"Failed to initialize API client: {str(e)}")
            raise
        
        # Initialize utilities with API client
        self.candidate_util = CandidateUtil(self.api_client)
        self.vendor_util = VendorUtil(self.api_client)
        # Use job unique_id to identify the job type
        self.job_activity_log_util = JobActivityLogUtil(
            self.api_client, 
            job_unique_id='bot_candidate_email_extractor'
        )
        
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
        candidate_id = candidate.get('candidate_id')
        
        self.logger.info(f"\n{'─' * 80}")
        self.logger.info(f"Processing: {email}")
        self.logger.info(f"{'─' * 80}")
        
        # Initialize tracking variables
        error_message = None
        total_emails_fetched = 0
        contacts_saved = 0
        filter_stats_aggregated = {
            'total': 0,
            'passed': 0,
            'junk': 0,
            'not_recruiter': 0,
            'calendar_invites': 0
        }
        
        # Connect to email account (server/port are hardcoded in GmailIMAPConnector)
        connector = GmailIMAPConnector(
            email=email,
            password=password
        )
        
        if not connector.connect():
            error_message = "Authentication failed - unable to connect to IMAP server"
            self.logger.error(f"Failed to connect to {email}")
        else:
            try:
                reader = EmailReader(connector)
                batch_size = self.config['email']['batch_size']
                
                # Load last processed UID from tracker
                last_uid = self.uid_tracker.get_last_uid(email)
                
                start_index = 0
                total_contacts = []
                
                while True:
                    # Fetch email batch
                    emails, next_start_index = reader.fetch_emails(
                        since_uid=last_uid,
                        batch_size=batch_size,
                        start_index=start_index
                    )
                    
                    if not emails:
                        break
                    
                    total_emails_fetched += len(emails)
                    self.logger.info(f"Fetched {len(emails)} emails")
                    
                    # Filter emails and get statistics
                    filtered_emails, filter_stats = self.email_filter.filter_emails(emails, self.cleaner)
                    
                    # Aggregate filter stats
                    for key in filter_stats_aggregated:
                        filter_stats_aggregated[key] += filter_stats.get(key, 0)
                    
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
                    
                    # Continue to next batch
                    if not next_start_index:
                        break
                    start_index = next_start_index
                
                # Save all contacts
                if total_contacts:
                    contacts_saved = self.vendor_util.save_contacts(total_contacts)
                
                self.logger.info(f"✓ Completed {email}: {contacts_saved} contacts saved from {filter_stats_aggregated['passed']} filtered emails")
                
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(f"Error processing {email}: {str(e)}", exc_info=True)
            finally:
                connector.disconnect()
        
        # Build comprehensive notes
        notes_parts = []
        
        # Show insertion count prominently
        if contacts_saved > 0:
            notes_parts.append(f"{contacts_saved} inserted")
        else:
            notes_parts.append("0 inserted")
        
        # Add email fetching info
        notes_parts.append(f"Fetched: {total_emails_fetched}")
        
        # Add filtering stats if any emails were processed
        if filter_stats_aggregated['total'] > 0:
            filter_breakdown = []
            if filter_stats_aggregated['junk'] > 0:
                filter_breakdown.append(f"Junk: {filter_stats_aggregated['junk']}")
            if filter_stats_aggregated['not_recruiter'] > 0:
                filter_breakdown.append(f"Not recruiter: {filter_stats_aggregated['not_recruiter']}")
            if filter_stats_aggregated['calendar_invites'] > 0:
                filter_breakdown.append(f"Calendar: {filter_stats_aggregated['calendar_invites']}")
            
            if filter_breakdown:
                notes_parts.append(f"Filtered: {filter_stats_aggregated['passed']} ({', '.join(filter_breakdown)})")
            else:
                notes_parts.append(f"Filtered: {filter_stats_aggregated['passed']}")
        
        # Show error prominently if it occurred
        if error_message:
            notes_parts.append(f"ERROR: {error_message}")
        
        notes = " | ".join(notes_parts)
        
        # Log activity for ALL candidates (even with 0 contacts or errors)
        if candidate_id:
            self.job_activity_log_util.log_activity(
                candidate_id=candidate_id,
                contacts_extracted=contacts_saved,
                notes=notes
            )
        else:
            self.logger.warning(f"No candidate_id for {email} - skipping activity log")
        
        return contacts_saved
    
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
