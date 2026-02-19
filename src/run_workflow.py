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
from typing import Dict
from dotenv import load_dotenv

# Load .env file BEFORE any other imports that use environment variables
load_dotenv() 

# Add src to path to allow absolute imports if running from root
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def _safe_json_load(value, default):
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _merge_parameters(defaults: Dict, runtime: Dict) -> Dict:
    merged = dict(defaults or {})
    for key, value in (runtime or {}).items():
        merged[key] = value
    return merged

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
    parser.add_argument(
        "--candidate-id",
        type=int,
        required=False,
        help="Process one candidate ID only (overrides list mode)"
    )
    parser.add_argument(
        "--candidate-email",
        type=str,
        required=False,
        help="Process one candidate email only (overrides list mode)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test SQL query and show candidates without running extraction"
    )
    
    args = parser.parse_args()
    workflow_key = args.workflow_key
    schedule_id = args.schedule_id
    params_str = args.params
    candidate_id = args.candidate_id
    candidate_email = args.candidate_email
    
    runtime_parameters = None
    if params_str:
        try:
            runtime_parameters = json.loads(params_str)
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
            
        workflow_id = config["id"]
        workflow_name = config["name"]
        credentials_sql = config["credentials_list_sql"]
        default_parameters = _safe_json_load(config.get("parameters_config"), default={})
        parameters = _merge_parameters(default_parameters, runtime_parameters or {})
        if candidate_id is not None:
            parameters["candidate_id"] = candidate_id
        if candidate_email:
            parameters["candidate_email"] = candidate_email
        
        logger.info(f"Loaded workflow: {workflow_name} (ID: {workflow_id})")
        
        # 2. Start Run Tracking
        run_id = manager.start_run(workflow_id, schedule_id, parameters)
        
        try:
            # 3. Initialize Source
            if not credentials_sql:
                raise ValueError("Workflow configuration missing 'credentials_list_sql'")
                
            candidate_source = DatabaseCandidateSource(credentials_sql)
            
            # DRY-RUN MODE: Test SQL query without running extraction
            if args.dry_run:
                logger.info("=" * 80)
                logger.info("DRY-RUN MODE - Testing SQL query without extraction")
                logger.info("=" * 80)
                logger.info(f"Workflow: {workflow_name} (ID: {workflow_id})")
                logger.info(f"SQL Query: {credentials_sql}")
                logger.info("")
                
                try:
                    candidates = candidate_source.get_active_candidates(
                        candidate_id=parameters.get("candidate_id"),
                        candidate_email=parameters.get("candidate_email"),
                    )
                    
                    if not candidates:
                        logger.warning("⚠ No candidates found with email credentials")
                        logger.info("=" * 80)
                        sys.exit(0)
                    
                    logger.info(f"✓ Found {len(candidates)} candidate(s) to process:")
                    logger.info("")
                    
                    for idx, candidate in enumerate(candidates[:10], 1):  # Show first 10
                        logger.info(f"  {idx}. Email: {candidate.get('email')}")
                        logger.info(f"     ID: {candidate.get('candidate_id')}")
                        logger.info(f"     Name: {candidate.get('name', 'N/A')}")
                        logger.info(f"     Has Password: {'✓' if candidate.get('imap_password') else '✗'}")
                        logger.info("")
                    
                    if len(candidates) > 10:
                        logger.info(f"  ... and {len(candidates) - 10} more candidates")
                        logger.info("")
                    
                    logger.info("=" * 80)
                    logger.info("DRY-RUN COMPLETE - No extraction performed")
                    logger.info("To run extraction, remove the --dry-run flag")
                    logger.info("=" * 80)
                    sys.exit(0)
                    
                except Exception as e:
                    logger.error(f"✗ SQL query failed: {e}")
                    logger.error("Please check your credentials_list_sql configuration")
                    logger.info("=" * 80)
                    sys.exit(1)
            
            
            # 4. Initialize and Run Service
            # Note: Service handles its own internal logging, but we pass manager/run_id for status updates
            service = EmailExtractionService(
                candidate_source=candidate_source,
                workflow_manager=manager,
                run_id=run_id,
                workflow_id=workflow_id,
                runtime_parameters=parameters,
            )
            
            service.run(
                candidate_id=parameters.get("candidate_id"),
                candidate_email=parameters.get("candidate_email"),
            )
            
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
