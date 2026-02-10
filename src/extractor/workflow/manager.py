import logging
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from ..core.database import get_db_client

logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    Manages the lifecycle of automation workflows:
    - Loading configuration
    - Tracking execution status in logs
    """
    
    def __init__(self):
        self.db_client = get_db_client()

    def get_workflow_config(self, workflow_key: str) -> Optional[Dict[str, Any]]:
        """
        Fetch active workflow configuration by key.
        """
        query = """
        SELECT 
            id, workflow_key, name, status, 
            credentials_list_sql, recipient_list_sql, 
            parameters_config
        FROM automation_workflows 
        WHERE workflow_key = %s AND status = 'active'
        LIMIT 1
        """
        results = self.db_client.execute_query(query, (workflow_key,))
        
        if not results:
            logger.error(f"Workflow '{workflow_key}' not found or not active.")
            return None
            
        config = results[0]
        # Parse JSON config if it's a string, otherwise it's already a dict/list
        if isinstance(config.get('parameters_config'), str):
             try:
                 config['parameters_config'] = json.loads(config['parameters_config'])
             except json.JSONDecodeError:
                 logger.warning(f"Failed to parse parameters_config for workflow {workflow_key}")

        return config

    def start_run(self, workflow_id: int, schedule_id: Optional[int] = None, parameters: Optional[Dict] = None) -> str:
        """
        Create a new log entry with status 'running'.
        Returns the run_id (UUID).
        """
        run_id = str(uuid.uuid4())
        
        # Serialize parameters if provided
        parameters_json = json.dumps(parameters) if parameters else None
        
        query = """
        INSERT INTO automation_workflow_logs 
        (workflow_id, schedule_id, run_id, status, parameters_used, started_at) 
        VALUES (%s, %s, %s, 'running', %s, NOW(6))
        """
        
        try:
            self.db_client.execute_non_query(query, (workflow_id, schedule_id, run_id, parameters_json))
            logger.info(f"Started workflow run {run_id} for workflow_id {workflow_id}")
            return run_id
        except Exception as e:
            logger.error(f"Failed to start workflow run: {e}")
            raise

    def update_run_status(self, run_id: str, status: str, 
                          records_processed: int = 0, 
                          records_failed: int = 0,
                          error_summary: Optional[str] = None,
                          error_details: Optional[str] = None,
                          execution_metadata: Optional[Dict] = None):
        """
        Update the status of a running workflow.
        """
        # Serialize metadata
        metadata_json = json.dumps(execution_metadata) if execution_metadata else None
        
        query = """
        UPDATE automation_workflow_logs 
        SET 
            status = %s,
            records_processed = %s,
            records_failed = %s,
            error_summary = %s,
            error_details = %s,
            execution_metadata = %s,
            finished_at = CASE WHEN %s IN ('success', 'failed', 'partial_success', 'timed_out') THEN NOW(6) ELSE finished_at END,
            updated_at = NOW(6)
        WHERE run_id = %s
        """
        
        # Truncate error summary if needed
        if error_summary and len(error_summary) > 255:
            error_summary = error_summary[:252] + "..."
            
        try:
            self.db_client.execute_non_query(
                query, 
                (status, records_processed, records_failed, error_summary, error_details, metadata_json, status, run_id)
            )
            logger.info(f"Updated run {run_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update run status for {run_id}: {e}")
            # Don't raise here to avoid crashing the cleanup logic in finally blocks

    def update_schedule_status(self, schedule_id: int):
        """
        Update the schedule's next run time (placeholder logic)
        Assumes automation_workflows_schedule table exists.
        """
        if not schedule_id:
            return
            
        # TODO: Implement actual next_run calculation based on cron/interval
        # For now, we just update last_run_at
        query = """
        UPDATE automation_workflows_schedule
        SET last_run_at = NOW(6)
        WHERE id = %s
        """
        try:
            self.db_client.execute_non_query(query, (schedule_id,))
            logger.info(f"Updated schedule {schedule_id} last_run_at")
        except Exception as e:
            logger.error(f"Failed to update schedule {schedule_id}: {e}")
