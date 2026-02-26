import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)

from src.extractor.orchestration.service import EmailExtractionService

# Mock Candidate Source
class MockCandidateSource:
    def get_active_candidates(self, **kwargs):
        # Load from .env or hardcoded
        email = os.getenv('TEST_EMAIL')
        password = os.getenv('TEST_APP_PASSWORD')
        if not email or not password:
            print("Skipping test: TEST_EMAIL or TEST_APP_PASSWORD not set")
            return []
            
        return [{
            'id': 1,
            'email': email,
            'imap_password': password,
            'name': 'Test User'
        }]

def verify_json_summary():
    # Load env
    load_dotenv(project_root / '.env')
    
    # Init service
    service = EmailExtractionService(
        candidate_source=MockCandidateSource(),
        run_id='verify_summary_test'
    )
    
    # Run
    # We want to limit the run to just 1 email to be fast?
    # Service uses config for batch size.
    # We can inject a config or just let it run a small batch.
    # But `service` loads config from file.
    # We might want to patch config or just let it run.
    # IF it runs too many emails, it might be slow.
    # user has 2 emails in previous test.
    
    try:
        service.run()
    except Exception as e:
        print(f"Service run failed: {e}")
        # Even if it fails, it might save JSON?
        pass

    # Check output
    from datetime import datetime
    date_str = datetime.now().strftime('%Y-%m-%d')
    output_file = project_root / f"output/{date_str}/run_verify_summary_test.json"
    
    if output_file.exists():
        with open(output_file, 'r') as f:
            data = json.load(f)
            print("JSON Output Content:")
            # Print summary if exists
            if 'summary' in data:
                print(json.dumps(data['summary'], indent=2))
            else:
                print("SUMMARY NOT FOUND IN JSON")
    else:
        print(f"Output file not found: {output_file}")

if __name__ == "__main__":
    verify_json_summary()
