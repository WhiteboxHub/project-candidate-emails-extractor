import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import new architecture components
from src.extractor.core.settings import get_config
from src.extractor.connectors.imap_gmail import GmailIMAPConnector
from src.extractor.email.reader import EmailReader
from src.extractor.email.cleaner import EmailCleaner
from src.extractor.extraction.contacts import ContactExtractor
from src.extractor.filtering.rules import EmailFilter
from src.extractor.state.uid_tracker import get_uid_tracker

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewArchitectureTest:
    """Test new architecture extraction pipeline locally"""
    
    def __init__(self):
        # Load environment variables from email-extractor/.env
        env_path = project_root / 'email-extractor' / '.env'
        load_dotenv(env_path)
        
        # Get credentials from .env
        self.test_email = os.getenv('TEST_EMAIL')
        self.test_app_password = os.getenv('TEST_APP_PASSWORD')
        self.batch_size = int(os.getenv('TEST_BATCH_SIZE', '100'))
        
        if not self.test_email or not self.test_app_password:
            print("[ERROR] TEST_EMAIL and TEST_APP_PASSWORD must be set in email-extractor/.env")
            print("\nAdd these lines to email-extractor/.env:")
            print("  TEST_EMAIL=your.email@gmail.com")
            print("  TEST_APP_PASSWORD=your_16_char_app_password")
            sys.exit(1)
        
        print("=" * 80)
        print("NEW ARCHITECTURE TEST - Extracting from YOUR account")
        print("=" * 80)
        
        # Load config
        try:
            config_loader = get_config()
            # Try to load from configs/config.yaml
            config_path = project_root / 'configs' / 'config.yaml'
            if config_path.exists():
                config_loader.config_path = config_path
            self.config = config_loader.load()
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}, using minimal config")
            # Minimal config fallback
            self.config = {
                'email': {
                    'imap_server': 'imap.gmail.com',
                    'imap_port': 993,
                    'batch_size': self.batch_size
                },
                'extraction': {
                    'enabled_methods': ['regex', 'spacy', 'gliner'],
                    'block_gmail': True,
                    'extract_multiple_contacts': True
                }
            }
        
        # Initialize NEW architecture components
        self.cleaner = EmailCleaner()
        self.extractor = ContactExtractor(self.config)
        self.email_filter = EmailFilter(self.config)
        
        # Initialize UID tracker for test mode
        self.uid_tracker = get_uid_tracker('last_run_new_test.json')
        
        print(f"Testing with: {self.test_email}")
        print(f"Batch size: {self.batch_size} emails")
        print(f"Using NEW architecture from src/")
    
    def run_test(self):
        """Run extraction test on your account"""
        try:
            print("\n" + "=" * 80)
            print(f"[CONNECT] Connecting to: {self.test_email}")
            print("=" * 80)
            
            # Connect to email
            connector = GmailIMAPConnector(
                email=self.test_email,
                password=self.test_app_password
            )
            
            if not connector.connect():
                print(f"[ERROR] Failed to connect to {self.test_email}")
                print("\n🔧 Troubleshooting:")
                print("  1. Enable 2FA: https://myaccount.google.com/security")
                print("  2. Create App Password: https://myaccount.google.com/apppasswords")
                print("  3. Use 16-character app password (not regular password)")
                print("  4. Enable IMAP in Gmail Settings")
                return
            
            print("[OK] Connected successfully!")
            
            try:
                reader = EmailReader(connector)
                
                # Check last UID
                last_uid = self.uid_tracker.get_last_uid(self.test_email)
                if last_uid:
                    print(f"\n[INFO] Resuming from last UID: {last_uid}")
                else:
                    print("\n[INFO] First run - processing all emails")
                
                # Fetch emails
                print(f"\n[FETCH] Fetching up to {self.batch_size} emails...")
                emails, _ = reader.fetch_emails(
                    since_uid=last_uid,
                    batch_size=self.batch_size,
                    start_index=0
                )
                
                if not emails:
                    print("[WARNING] No emails found in inbox")
                    return
                
                print(f"[OK] Fetched {len(emails)} emails")
                
                # Filter emails
                print("\n[FILTER] Filtering recruiter emails...")
                filtered_emails, filter_stats = self.email_filter.filter_emails(emails, self.cleaner)
                
                print(f"[OK] Found {len(filtered_emails)} potential recruiter emails")
                print(f"     Filter stats: {filter_stats}")
                
                if not filtered_emails:
                    print("\n[WARNING] No recruiter emails found after filtering")
                    print("   (All emails were from blacklisted domains or spam)")
                    return
                
                # Extract contacts
                print("\n[EXTRACT] Extracting vendor contacts using NEW architecture...")
                print("=" * 80)
                contacts = []
                seen_emails = set()
                seen_linkedin = set()
                
                for i, email_data in enumerate(filtered_emails, 1):
                    try:
                        clean_body = email_data.get('clean_body', 
                                                    self.cleaner.extract_body(email_data['message']))
                        
                        subject = email_data.get('subject', '')
                        
                        # NEW ARCHITECTURE: extract_contacts returns LIST
                        contacts_list = self.extractor.extract_contacts(
                            email_data['message'],
                            clean_body,
                            source_email=self.test_email,
                            subject=subject
                        )
                        
                        # Process each contact in the list
                        for contact in contacts_list:
                            # Add metadata
                            contact['raw_body'] = clean_body
                            contact['extracted_from_subject'] = email_data.get('subject', 'N/A')
                            contact['extracted_from_date'] = email_data.get('date', 'N/A')
                            contact['extracted_from_uid'] = email_data.get('uid', 'N/A')
                            
                            # Only save if we have email or linkedin
                            if contact.get('email') or contact.get('linkedin_id'):
                                # Check for duplicates
                                email_addr = contact.get('email', '').lower()
                                linkedin_id = (contact.get('linkedin_id') or '').lower()
                                
                                is_duplicate = False
                                if email_addr and email_addr in seen_emails:
                                    is_duplicate = True
                                elif linkedin_id and linkedin_id in seen_linkedin:
                                    is_duplicate = True
                                
                                if not is_duplicate:
                                    contacts.append(contact)
                                    if email_addr:
                                        seen_emails.add(email_addr)
                                    if linkedin_id:
                                        seen_linkedin.add(linkedin_id)
                                    
                                    # Enhanced display
                                    extraction_src = contact.get("extraction_source", "unknown")
                                    
                                    print(f"\n  [{i}/{len(filtered_emails)}] ✓ EXTRACTED (from: {extraction_src}):")
                                    print(f"      Name:     {contact.get('name', 'N/A')}")
                                    print(f"      Company:  {contact.get('company', 'N/A')}")
                                    print(f"      Email:    {contact.get('email', 'N/A')}")
                                    print(f"      Phone:    {contact.get('phone', 'N/A')}")
                                    print(f"      Position: {contact.get('job_position', 'N/A')}")
                                    print(f"      Emp Type: {contact.get('employment_type', 'N/A')}")
                                    print(f"      Location: {contact.get('location', 'N/A')}")
                                    print(f"      ZIP:      {contact.get('zip_code', 'N/A')}")
                                    print(f"      Subject:  {email_data.get('subject', 'N/A')[:60]}...")
                                else:
                                    print(f"\n  [{i}/{len(filtered_emails)}] ⊘ Duplicate: {contact.get('email', contact.get('linkedin_id', 'N/A'))}")
                            else:
                                print(f"\n  [{i}/{len(filtered_emails)}] ⊘ Skipped: No email or LinkedIn found")
                        
                    except Exception as e:
                        print(f"\n  [{i}/{len(filtered_emails)}] ✗ Error: {str(e)}")
                        logger.error(f"Extraction error for email {i}: {str(e)}", exc_info=True)
                        continue
                
                print("\n" + "=" * 80)
                
                # Update last UID
                if emails:
                    max_uid = max(int(e['uid']) for e in emails)
                    self.uid_tracker.update_last_uid(self.test_email, str(max_uid))
                    print(f"\n[INFO] Updated last UID to: {max_uid}")
                
                # Save to JSON
                self.save_results(contacts)
                
                # Print summary
                self.print_summary(emails, filtered_emails, contacts, filter_stats)
                
            finally:
                connector.disconnect()
                print("\n[OK] Disconnected from email")
                
        except Exception as e:
            logger.error(f"Test failed: {str(e)}", exc_info=True)
            print(f"\n[ERROR] Error: {str(e)}")
    
    def save_results(self, contacts):
        """Save contacts to JSON file"""
        if not contacts:
            print("\n[WARNING] No contacts to save")
            return
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"test_results_new_arch_{timestamp}.json"
        
        # Prepare data
        output_data = {
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'source_email': self.test_email,
                'total_contacts': len(contacts),
                'batch_size': self.batch_size,
                'architecture': 'NEW (src/)'
            },
            'contacts': contacts
        }
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] Results saved to: {output_file}")
        
        # Also save detailed contact list
        print(f"\n[LIST] DETAILED CONTACT RESULTS:")
        print("=" * 80)
        for i, contact in enumerate(contacts, 1):
            print(f"\n{i}. {contact.get('name', 'N/A')}")
            print(f"   Email:        {contact.get('email', 'N/A')}")
            print(f"   Phone:        {contact.get('phone', 'N/A')}")
            print(f"   Company:      {contact.get('company', 'N/A')}")
            print(f"   Position:     {contact.get('job_position', 'N/A')}")
            print(f"   LinkedIn:     {contact.get('linkedin_id', 'N/A')}")
            print(f"   Location:     {contact.get('location', 'N/A')}")
            print(f"   ZIP Code:     {contact.get('zip_code', 'N/A')}")
            print(f"   From Subject: {contact.get('extracted_from_subject', 'N/A')[:50]}...")
        
        print("\n" + "=" * 80)
    
    def print_summary(self, emails, filtered_emails, contacts, filter_stats):
        """Print extraction summary"""
        print("\n" + "=" * 80)
        print("[STATS] EXTRACTION SUMMARY (NEW ARCHITECTURE)")
        print("=" * 80)
        print(f"Total emails fetched:      {len(emails)}")
        print(f"Recruiter emails found:    {len(filtered_emails)}")
        print(f"Contacts extracted:        {len(contacts)}")
        print()
        print("Filter Statistics:")
        for key, value in filter_stats.items():
            print(f"  - {key}: {value}")
        print()
        print("Breakdown by field:")
        if contacts:
            names_found = sum(1 for c in contacts if c.get('name'))
            emails_found = sum(1 for c in contacts if c.get('email'))
            phones_found = sum(1 for c in contacts if c.get('phone'))
            companies_found = sum(1 for c in contacts if c.get('company'))
            linkedin_found = sum(1 for c in contacts if c.get('linkedin_id'))
            positions_found = sum(1 for c in contacts if c.get('job_position'))
            locations_found = sum(1 for c in contacts if c.get('location'))
            zips_found = sum(1 for c in contacts if c.get('zip_code'))
            emp_types_found = sum(1 for c in contacts if c.get('employment_type'))
            
            total = len(contacts)
            print(f"  - Names:      {names_found}/{total} ({names_found*100//total if total else 0}%)")
            print(f"  - Emails:     {emails_found}/{total} ({emails_found*100//total if total else 0}%)")
            print(f"  - Phones:     {phones_found}/{total} ({phones_found*100//total if total else 0}%)")
            print(f"  - Companies:  {companies_found}/{total} ({companies_found*100//total if total else 0}%)")
            print(f"  - LinkedIn:   {linkedin_found}/{total} ({linkedin_found*100//total if total else 0}%)")
            print(f"  - Positions:  {positions_found}/{total} ({positions_found*100//total if total else 0}%)")
            print(f"  - Emp Types:  {emp_types_found}/{total} ({emp_types_found*100//total if total else 0}%)")
            print(f"  - Locations:  {locations_found}/{total} ({locations_found*100//total if total else 0}%)")
            print(f"  - ZIP Codes:  {zips_found}/{total} ({zips_found*100//total if total else 0}%)")
        print("=" * 80)


def main():
    """Main entry point"""
    print("""
================================================================================
    EMAIL VENDOR EXTRACTOR - NEW ARCHITECTURE TEST
================================================================================
    
This test will:
  1. Connect to YOUR Gmail account
  2. Fetch recent emails (no database needed)
  3. Extract vendor/recruiter contacts using NEW architecture:
     - Multi-method extraction (Regex → SpaCy → GLiNER)
     - Database-driven filtering (keywords.csv)
     - Enhanced validation and deduplication
  4. Save results to JSON file
  
Setup Required:
  1. Add to email-extractor/.env:
     TEST_EMAIL=your.email@gmail.com
     TEST_APP_PASSWORD=your_16_char_app_password
     TEST_BATCH_SIZE=50  (optional, default: 100)
  
  2. Gmail App Password:
     - Enable 2FA: https://myaccount.google.com/security
     - Create App Password: https://myaccount.google.com/apppasswords
     - Use 16-character password (not regular password)
  
================================================================================
    """)
    
    try:
        tester = NewArchitectureTest()
        tester.run_test()
        
        print("\n" + "=" * 80)
        print("[OK] TEST COMPLETE - Check results above")
        print("=" * 80)
        print("\nNEW ARCHITECTURE FEATURES:")
        print("  ✓ Multi-method extraction pipeline (Regex → SpaCy → GLiNER)")
        print("  ✓ Database-driven filtering from keywords.csv")
        print("  ✓ Enhanced position extraction with 70+ patterns")
        print("  ✓ Location extraction with zip codes")
        print("  ✓ Employment type detection")
        print("  ✓ Multiple contact extraction per email")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
