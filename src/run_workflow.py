#!/usr/bin/env python3
"""
Automation Workflow Runner

Entry point for Windows Task Scheduler config.
Executes a specific workflow by key.

Usage:
    python src/run_workflow.py --workflow-key email_extractor
"""

import sys
import logging
import argparse
from pathlib import Path
import traceback
import json

# Add src to path to allow absolute imports if running from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor.core.logging import setup_logger as setup_logging
from src.extractor.workflow.manager import WorkflowManager
from src.extractor.persistence.db_candidate_source import DatabaseCandidateSource
from src.extractor.orchestration.service import EmailExtractionService

# Configure logging to stdout/file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("workflow_runner")

def main():
    parser = argparse.ArgumentParser(description="Run an automation workflow")
    parser.add_argument(
        "--workflow-key", 
        type=str, 
        required=True,
        help="Unique key of the workflow to run (e.g., 'email_extractor')"
    )
    parser.add_argument(
        "--schedule-id",
        type=int,
        required=False,
        help="ID of the schedule that triggered this run (optional)"
    )
    parser.add_argument(
        "--params",
        type=str,
        required=False,
        help="JSON string of runtime parameters"
    )
    
    args = parser.parse_args()
    workflow_key = args.workflow_key
    schedule_id = args.schedule_id
    params_str = args.params
    
    parameters = None
    if params_str:
        try:
            parameters = json.loads(params_str)
        except json.JSONDecodeError:
            logger.error("Invalid JSON provided in --params")
            sys.exit(1)
    
    logger.info(f"Starting workflow run for key: {workflow_key}")
    
    try:
        # 1. Initialize Manager and Load Config
        manager = WorkflowManager()
        config = manager.get_workflow_config(workflow_key)
        
        if not config:
            logger.error(f"Workflow configuration not found or inactive for key: {workflow_key}")
            sys.exit(1)
            
        workflow_id = config['id']
        workflow_name = config['name']
        credentials_sql = config['credentials_list_sql']
        
        logger.info(f"Loaded workflow: {workflow_name} (ID: {workflow_id})")
        
        # 2. Start Run Tracking
        run_id = manager.start_run(workflow_id, schedule_id, parameters)
        
        try:
            # 3. Initialize Source
            if not credentials_sql:
                raise ValueError("Workflow configuration missing 'credentials_list_sql'")
                
            candidate_source = DatabaseCandidateSource(credentials_sql)
            
            # 4. Initialize and Run Service
            # Note: Service handles its own internal logging, but we pass manager/run_id for status updates
            service = EmailExtractionService(
                candidate_source=candidate_source,
                workflow_manager=manager,
                run_id=run_id
            )
            
            service.run()
            
            # 5. Update Schedule Status (if applicable)
            if schedule_id:
                manager.update_schedule_status(schedule_id)
            
            logger.info(f"Workflow run {run_id} completed successfully.")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            traceback.print_exc()
            # Status update to failed is handled within Service.run() catch block usually,
            # but if initialization fails before run(), we catch it here.
            manager.update_run_status(
                run_id, 'failed',
                error_summary=str(e)[:255],
                error_details=traceback.format_exc()
            )
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"Fatal error in workflow runner: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
