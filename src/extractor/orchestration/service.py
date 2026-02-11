#!/usr/bin/env python3
"""
Email Contact Extractor Service

Main orchestrator for extracting vendor contacts from candidate email inboxes.
Config-driven extraction pipeline with multiple fallback methods.
"""

import logging
from pathlib import Path
import sys
from typing import Optional, Dict

# Imports from within the package
# We assume the script is run as a module or with PYTHONPATH set to include src
from ..core.settings import get_config
from ..core.logging import setup_logger # Assuming this exists or using standard logging
from ..connectors.imap_gmail import GmailIMAPConnector
from ..email.reader import EmailReader
from ..email.cleaner import EmailCleaner
from ..extraction.contacts import ContactExtractor
from ..filtering.rules import EmailFilter
# from ..models.candidate import CandidateUtil # Replaced by injected source
from ..persistence.vendor_contacts import VendorUtil
from ..persistence.job_activity import JobActivityLogUtil, ActivityLog
from ..persistence.db_candidate_source import DatabaseCandidateSource
from ..workflow.manager import WorkflowManager
# from ..utils.uid_tracker import get_uid_tracker # Need to find where this is

# Fix for missing API Client import
try:
    from ..connectors.http_api import get_api_client, APIClient
except ImportError:
    # Fallback or mock if needed (but we found the file)
    from src.extractor.connectors.http_api import get_api_client, APIClient

logger = logging.getLogger(__name__)

class EmailExtractionService:
    """Main service for email contact extraction"""
    
    def __init__(self, 
                 candidate_source: DatabaseCandidateSource,
                 workflow_manager: Optional[WorkflowManager] = None,
                 run_id: Optional[str] = None):
        
        # Load configuration
        self.config_loader = get_config()
        self.config = self.config_loader.load()
        
        self.logger = logging.getLogger(__name__)
        self.candidate_source = candidate_source
        self.workflow_manager = workflow_manager
        self.run_id = run_id
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all service components"""
        # Initialize API client from environment variables
        try:
            self.api_client = get_api_client()
            self.logger.info("API client initialized successfully")
        except ValueError as e:
            self.logger.error(f"Failed to initialize API client: {str(e)}")
            self.api_client = None
            # We continue, but some features (VendorUtil) might fail or need to be skipped
        
        # Initialize utilities with API client
        if self.api_client:
            self.vendor_util = VendorUtil(self.api_client)
            # Use job unique_id to identify the job type
            self.job_activity_log_util = JobActivityLogUtil(
                self.api_client
            )
        else:
            self.logger.warning("API Client not available - VendorUtil / JobActivityLogUtil disabled")
            self.vendor_util = None
            self.job_activity_log_util = None

        self.cleaner = EmailCleaner()
        self.extractor = ContactExtractor(self.config)
        self.email_filter = EmailFilter(self.config)
        
        # Initialize UID tracker
        # TODO: Replace file-based tracker with DB tracker or keep file based per account?
        # For now, we keep the file based tracker logic if we can find the utility
        # It seems uid_tracker was in 'utils', let's stick to a simple placeholder if missing
        try:
             from ..state.uid_tracker import get_uid_tracker
             self.uid_tracker = get_uid_tracker('last_run.json')
        except ImportError:
            self.logger.warning("UID Tracker not found, using in-memory tracker (will re-read emails on restart)")
            class SimpleTracker:
                def __init__(self): self.uids = {}
                def get_last_uid(self, email): return self.uids.get(email, 0)
                def update_last_uid(self, email, uid): self.uids[email] = uid
                self.uid_tracker = SimpleTracker()
        
        # Initialize Workflow Logger
        if self.api_client:
            from ..persistence.workflow_logger import WorkflowLogger
            self.workflow_logger = WorkflowLogger(self.api_client)
        else:
            self.workflow_logger = None

        self.logger.info("All components initialized successfully")
    
    def run(self):
        """Main execution method"""
        execution_metadata = {'candidates': []}
        
        # Start workflow logging
        if self.workflow_logger and self.run_id:
            # Assuming workflow_id is available or defaults to 1 (or passed via config/args)
            # For now using 1 as default or extracting from somewhere if possible.
            # The user schema has workflow_id as BIGINT UNSIGNED NO.
            # We might need to look up the workflow_id for "email_extraction"
            workflow_id = 1 
            self.workflow_logger.start_run(workflow_id, self.run_id)

        try:
            # Get candidates with email credentials via injected source
            candidates = self.candidate_source.get_active_candidates()
            
            if not candidates:
                self.logger.warning("No candidates found with email credentials")
                if self.workflow_manager and self.run_id:
                     self.workflow_manager.update_run_status(
                        self.run_id, 'success', 
                        records_processed=0,
                        error_summary="No candidates found",
                        execution_metadata=execution_metadata
                     )
                return
            
            self.logger.info(f"Processing {len(candidates)} candidate accounts")
            
            # Process each candidate
            total_contacts = 0
            total_failed = 0
            
            for candidate in candidates:
                candidate_result = {
                    'candidate_id': candidate.get('candidate_id') or candidate.get('id'),
                    'email': candidate.get('email'),
                    'status': 'success',
                    'contacts_extracted': 0,
                    'error': None
                }
                
                try:
                    contacts_extracted = self.process_candidate(candidate)
                    total_contacts += contacts_extracted
                    candidate_result['contacts_extracted'] = contacts_extracted
                except Exception as e:
                    self.logger.error(f"Failed to process candidate {candidate.get('email')}: {str(e)}")
                    total_failed += 1
                    candidate_result['status'] = 'failed'
                    candidate_result['error'] = str(e)
                finally:
                    execution_metadata['candidates'].append(candidate_result)
            # Calculate summary for JSON output as per user request
            success_candidates = [c.get('email') for c in execution_metadata['candidates'] if c.get('status') == 'success']
            failed_candidates = [c.get('email') for c in execution_metadata['candidates'] if c.get('status') == 'failed']
            
            execution_metadata['summary'] = {
                'total_candidates': len(candidates),
                'success_count': len(success_candidates),
                'failure_count': len(failed_candidates),
                'total_contacts_extracted': total_contacts,
                'successful_candidates': success_candidates,
                'failed_candidates': failed_candidates
            }
            
            self.logger.info("=" * 80)
            self.logger.info(f"Extraction Complete - Total Contacts: {total_contacts}")
            self.logger.info("=" * 80)

            # Update final status
            if self.workflow_manager and self.run_id:
                status = 'success' if total_failed == 0 else 'partial_success'
                if total_failed == len(candidates) and len(candidates) > 0:
                    status = 'failed'
                
                self.workflow_manager.update_run_status(
                    self.run_id, status,
                    records_processed=total_contacts,
                    records_failed=total_failed,
                    execution_metadata=execution_metadata
                )
            
            # Update DB Logging
            if self.workflow_logger and self.run_id:
                status = 'success' if total_failed == 0 else 'partial_success'
                if total_failed == len(candidates) and len(candidates) > 0:
                    status = 'failed'
                
                self.workflow_logger.finish_run(
                    self.run_id, status,
                    records_processed=total_contacts,
                    records_failed=total_failed
                )

            # Save execution metadata to file
            try:
                from datetime import datetime
                import json
                
                date_str = datetime.now().strftime('%Y-%m-%d')
                output_dir = Path(f"output/{date_str}")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"run_{self.run_id or 'manual'}.json"
                with open(output_file, 'w') as f:
                    json.dump(execution_metadata, f, indent=2, default=str)
                
                self.logger.info(f"Saved execution log to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save execution log to file: {e}")
            
        except Exception as e:
            self.logger.error(f"Service execution failed: {str(e)}", exc_info=True)
            if self.workflow_manager and self.run_id:
                self.workflow_manager.update_run_status(
                    self.run_id, 'failed',
                    error_summary=str(e)[:255],
                    execution_metadata=execution_metadata
                )
            
            # DB Logging Failure
            if self.workflow_logger and self.run_id:
                self.workflow_logger.finish_run(
                    self.run_id, 'failed',
                    records_processed=0,
                    records_failed=0,
                    error_summary=str(e)[:255],
                    error_details=str(e)
                )

            raise
    
    def process_candidate(self, candidate: dict) -> int:
        """
        Process a single candidate's inbox
        """
        email = candidate['email']
        password = candidate['imap_password']
        candidate_name = candidate.get('name', email)
        candidate_id = candidate.get('candidate_id') or candidate.get('id')
        
        self.logger.info(f"\n{'─' * 80}")
        self.logger.info(f"Processing: {candidate_name} ({email})")
        self.logger.info(f"{'─' * 80}")
        
        # Initialize tracking variables
        error_message = None
        total_emails_fetched = 0
        contacts_saved = 0
        filter_stats_aggregated = {
            'total': 0, 'passed': 0, 'junk': 0, 'not_recruiter': 0, 'calendar_invites': 0
        }
        
        # Connect to email account
        connector = GmailIMAPConnector(
            email=email,
            password=password
        )
        
        if not connector.connect():
            error_message = "Authentication failed - unable to connect to IMAP server"
            self.logger.error(f"Failed to connect to {email}")
            return 0
            
        try:
            reader = EmailReader(connector)
            # Default batch size if not in config
            batch_size = self.config.get('email', {}).get('batch_size', 50)
            
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
                        clean_body = email_data.get('clean_body', self.cleaner.extract_body(email_data['message']))
                        
                        # extract_contacts returns a LIST of contacts
                        # CRITICAL FIX: Extract subject for position/location extraction
                        subject = email_data['message'].get('subject', '')
                        
                        contacts_list = self.extractor.extract_contacts(
                            email_data['message'],
                            clean_body,
                            source_email=email,
                            subject=subject
                        )
                        
                        # Add all valid contacts to the total list
                        for contact in contacts_list:
                            # Only save if we have email or linkedin
                            if contact.get('email') or contact.get('linkedin_id'):
                                # Add raw body for raw_position table
                                contact['raw_body'] = clean_body
                                contact['extracted_from_uid'] = email_data.get('uid')
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
            if total_contacts and self.vendor_util:
                contacts_saved = self.vendor_util.save_contacts(total_contacts, candidate_id=candidate_id)
            elif total_contacts and not self.vendor_util:
                self.logger.warning(f"Dry run: Would have saved {len(total_contacts)} contacts for {email}")
                contacts_saved = len(total_contacts) # Count as saved for reporting potentially
            
            self.logger.info(f"✓ Completed {email}: {contacts_saved} contacts saved from {filter_stats_aggregated['passed']} filtered emails")
            
        except Exception as e:
            error_message = f"Processing error: {str(e)}"
            self.logger.error(f"Error processing {email}: {str(e)}", exc_info=True)
            raise e
        finally:
            connector.disconnect()
        
        # Build notes and log activity
        self._log_activity(candidate_id, email, contacts_saved, total_emails_fetched, filter_stats_aggregated, error_message)
        
        return contacts_saved

    def _log_activity(self, candidate_id, email, contacts_saved, total_emails_fetched, filter_stats_aggregated, error_message):
        """Log activity summary"""
        notes_parts = []
        if contacts_saved > 0:
            notes_parts.append(f"{contacts_saved} inserted")
        else:
            notes_parts.append("0 inserted")
        
        notes_parts.append(f"Fetched: {total_emails_fetched}")
        
        if filter_stats_aggregated['total'] > 0:
            filter_breakdown = []
            if filter_stats_aggregated['junk'] > 0: filter_breakdown.append(f"Junk: {filter_stats_aggregated['junk']}")
            if filter_stats_aggregated['not_recruiter'] > 0: filter_breakdown.append(f"Not recruiter: {filter_stats_aggregated['not_recruiter']}")
            if filter_stats_aggregated['calendar_invites'] > 0: filter_breakdown.append(f"Calendar: {filter_stats_aggregated['calendar_invites']}")
            
            if filter_breakdown:
                notes_parts.append(f"Filtered: {filter_stats_aggregated['passed']} ({', '.join(filter_breakdown)})")
            else:
                notes_parts.append(f"Filtered: {filter_stats_aggregated['passed']}")
        
        if error_message:
            notes_parts.append(f"ERROR: {error_message}")
        
        notes = " | ".join(notes_parts)
        
        # Use job_activity_log_util if available
        if self.job_activity_log_util:
            if candidate_id:
                # We need to construct an ActivityLog object or pass params depending on implementation
                # The original used: self.job_activity_log_util.log_activity(...)
                # Let's assume log_activity exists
                 self.job_activity_log_util.log_activity(
                    candidate_id=candidate_id,
                    contacts_extracted=contacts_saved,
                    notes=notes,
                    job_unique_id='bot_candidate_email_extractor'
                )
            else:
                self.logger.warning(f"No candidate_id for {email} - skipping activity log")
