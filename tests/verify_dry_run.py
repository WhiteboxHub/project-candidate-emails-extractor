import sys
import unittest
from unittest.mock import MagicMock, patch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock mysql.connector before importing app modules
import sys
from unittest.mock import MagicMock
sys.modules['mysql'] = MagicMock()
sys.modules['mysql.connector'] = MagicMock()
sys.modules['mysql.connector.pooling'] = MagicMock()
sys.modules['yaml'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['joblib'] = MagicMock()
sys.modules['phonenumbers'] = MagicMock()
sys.modules['spacy'] = MagicMock()
sys.modules['gliner'] = MagicMock()
sys.modules['tokenizers'] = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['tldextract'] = MagicMock()
sys.modules['pdfminer'] = MagicMock()
sys.modules['docx'] = MagicMock()

from src.extractor.workflow.manager import WorkflowManager
from src.extractor.persistence.db_candidate_source import DatabaseCandidateSource
from src.extractor.orchestration.service import EmailExtractionService

# Mocking the database client
class MockDBClient:
    def __init__(self):
        self.queries = []
        
    def execute_query(self, query, params=None):
        self.queries.append((query, params))
        print(f"[MOCK DB] Executing Query: {query} | Params: {params}")
        
        # Mock responses based on query content
        if "SELECT" in query and "automation_workflows" in query:
            return [{
                'id': 1,
                'workflow_key': 'email_extractor',
                'name': 'Test Workflow',
                'status': 'active',
                'credentials_list_sql': 'SELECT * FROM candidates',
                'recipient_list_sql': None,
                'parameters_config': {}
            }]
        elif "SELECT" in query and "candidates" in query: # The mocked sql from above
            print("[MOCK DB] Returning mock candidates")
            return [
                {
                    'candidate_id': 101,
                    'email': 'test@example.com', 
                    'password': 'secret_password',
                    'imap_server': 'imap.gmail.com'
                }
            ]
        return []

    def execute_non_query(self, query, params=None):
        self.queries.append((query, params))
        print(f"[MOCK DB] Executing Non-Query: {query} | Params: {params}")
        return 1

class TestWorkflowExecution(unittest.TestCase):
    
    @patch('src.extractor.workflow.manager.get_db_client')
    @patch('src.extractor.orchestration.service.GmailIMAPConnector') # Mock email connection to avoid real network
    @patch('src.extractor.orchestration.service.EmailReader')
    @patch('src.extractor.orchestration.service.get_config')
    def test_dry_run(self, mock_get_config, mock_reader_cls, mock_connector_cls, mock_get_db):
        
        # Setup DB Mock
        mock_db = MockDBClient()
        mock_get_db.return_value = mock_db
        
        # Setup Email Mocks
        mock_connector = mock_connector_cls.return_value
        mock_connector.connect.return_value = True
        
        mock_reader = mock_reader_cls.return_value
        # Return one empty batch to finish the loop immediately
        mock_reader.fetch_emails.return_value = ([], None) 
        
        # Setup Config Mock
        mock_config_loader = mock_get_config.return_value
        mock_config_loader.load.return_value = {'some': 'config'}
        
        print("\n--- Starting Dry Run ---")
        
        # 1. Initialize Manager
        manager = WorkflowManager()
        config = manager.get_workflow_config('email_extractor')
        self.assertIsNotNone(config)
        self.assertEqual(config['workflow_key'], 'email_extractor')
        
        # 2. Start Run
        run_id = manager.start_run(config['id'])
        print(f"Run ID: {run_id}")
        self.assertIsNotNone(run_id)
        
        # 3. Source
        source = DatabaseCandidateSource(config['credentials_list_sql'])
        
        # 4. Service
        service = EmailExtractionService(
            candidate_source=source,
            workflow_manager=manager,
            run_id=run_id
        )
        
        # Disable sub-components that might fail in sandbox
        service.vendor_util = None 
        service.job_activity_log_util = None
        
        # 5. Run
        service.run()
        
        print("--- Dry Run Completed ---")
        
        # Verify DB interactions
        # Check if update_run_status was called with 'success'
        found_success = False
        for q, p in mock_db.queries:
            if "UPDATE automation_workflow_logs" in q and 'success' in p:
                found_success = True
                break
        
        if found_success:
            print("SUCCESS: Workflow logged success status to DB.")
        else:
            self.fail("Workflow did not log success status.")

if __name__ == '__main__':
    unittest.main()
