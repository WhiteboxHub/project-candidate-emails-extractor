import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WorkflowLogger:
    """
    Handles logging of workflow execution details to the 'automation_workflow_logs' table.
    """
    
    def __init__(self, api_client):
        self.logger = logging.getLogger(__name__)
        self.api_client = api_client
        self.table_name = 'automation_workflow_logs'

    def start_run(self, workflow_id: int, run_id: str, schedule_id: Optional[int] = None, parameters: Optional[Dict] = None) -> bool:
        """
        Creates a new log entry with status 'running' at the start of execution.
        """
        try:
            data = {
                'workflow_id': workflow_id,
                'run_id': run_id,
                'schedule_id': schedule_id,
                'status': 'running',
                'parameters_used': json.dumps(parameters) if parameters else None,
                'started_at': datetime.utcnow().isoformat(),
                'records_processed': 0,
                'records_failed': 0
            }
            
            # Using the API client to insert data
            # Assuming api_client.post works for table insertions or there's a specific endpoint
            # If standard table API: /api/automation-workflow-logs
            response = self.api_client.post(f"/api/{self.table_name.replace('_', '-')}", data)
            
            if response:
                self.logger.info(f"Started workflow run logging: {run_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to start workflow logging: {str(e)}")
            return False

    def update_status(self, run_id: str, status: str, 
                      records_processed: Optional[int] = None, 
                      records_failed: Optional[int] = None,
                      execution_metadata: Optional[Dict] = None) -> bool:
        """
        Updates the status and statistics of a running workflow.
        """
        try:
            data = {'status': status}
            if records_processed is not None:
                data['records_processed'] = records_processed
            if records_failed is not None:
                data['records_failed'] = records_failed
            if execution_metadata is not None:
                data['execution_metadata'] = json.dumps(execution_metadata)
            
            # We need to update based on run_id. 
            # Assuming API supports update by query or we need the primary key ID.
            # If the API requires ID, we might need to store it from start_run response.
            # PROPOSED STRATEGY: Use a custom endpoint or assume we can patch by run_id if supported,
            # OR fetches the ID first.
            
            # For now, let's assume we can filter by run_id or user needs to implement the specific API call.
            # Converting to standard filtering if this were a direct DB access:
            # UPDATE ... WHERE run_id = run_id
            
            # If using standard REST API generated from DB:
            # likely need to GET /api/automation-workflow-logs?run_id=X first to get ID.
            
            records = self.api_client.get(f"/api/{self.table_name.replace('_', '-')}?run_id={run_id}")
            if records and isinstance(records, list) and len(records) > 0:
                record_id = records[0]['id']
                self.api_client.put(f"/api/{self.table_name.replace('_', '-')}/{record_id}", data)
                return True
            else:
                 self.logger.warning(f"Could not find workflow log for run_id: {run_id}")
                 return False

        except Exception as e:
            self.logger.error(f"Failed to update workflow logging: {str(e)}")
            return False

    def finish_run(self, run_id: str, status: str, 
                   records_processed: int, records_failed: int,
                   error_summary: Optional[str] = None,
                   error_details: Optional[str] = None) -> bool:
        """
        Finalizes the log entry with end time and final status.
        """
        try:
            data = {
                'status': status,
                'finished_at': datetime.utcnow().isoformat(),
                'records_processed': records_processed,
                'records_failed': records_failed
            }
            if error_summary:
                data['error_summary'] = error_summary
            if error_details:
                data['error_details'] = error_details

            # Get record ID first 
            records = self.api_client.get(f"/api/{self.table_name.replace('_', '-')}?run_id={run_id}")
            if records and isinstance(records, list) and len(records) > 0:
                record_id = records[0]['id']
                self.api_client.put(f"/api/{self.table_name.replace('_', '-')}/{record_id}", data)
                self.logger.info(f"Finished workflow run logging: {run_id} ({status})")
                return True
            else:
                self.logger.warning(f"Could not find workflow log for run_id: {run_id}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to finish workflow logging: {str(e)}")
            return False