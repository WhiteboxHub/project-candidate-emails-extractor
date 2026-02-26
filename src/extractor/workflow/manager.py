import logging
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages the lifecycle of automation workflows via the wbl-backend REST API.

    All SQL queries have been replaced with API calls:
      - get_workflow_config()   → GET  /api/automation-workflow/by-key/{key}
      - start_run()             → POST /api/automation-workflow-log/
      - update_run_status()     → PATCH /api/automation-workflow-log/by-run-id/{run_id}
      - update_schedule_status()→ PUT  /api/automation-workflow-schedule/{id}
    """

    def __init__(self, api_client):
        """
        api_client: an instance of APIClient (src.extractor.connectors.http_api.APIClient).
        """
        self.api_client = api_client

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow config
    # ─────────────────────────────────────────────────────────────────────────

    def get_workflow_config(self, workflow_key: str) -> Optional[Dict[str, Any]]:
        """
        Fetch active workflow configuration by key via API.
        """
        try:
            config = self.api_client.get(f"/api/automation-workflow/by-key/{workflow_key}")
            if not config:
                logger.error("Workflow '%s' not found or not active.", workflow_key)
                return None
            # parameters_config may come back already parsed (dict) or as JSON string
            if isinstance(config.get("parameters_config"), str):
                try:
                    config["parameters_config"] = json.loads(config["parameters_config"])
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse parameters_config for workflow %s", workflow_key
                    )
            return config
        except Exception as e:
            logger.error("Failed to fetch workflow config for key '%s': %s", workflow_key, e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Run lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start_run(
        self,
        workflow_id: int,
        schedule_id: Optional[int] = None,
        parameters: Optional[Dict] = None,
    ) -> str:
        """
        Create a new workflow log entry with status 'running'.
        Returns the run_id (UUID).
        """
        run_id = str(uuid.uuid4())
        payload = {
            "workflow_id": workflow_id,
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "records_processed": 0,
            "records_failed": 0,
        }
        if schedule_id is not None:
            payload["schedule_id"] = schedule_id
        if parameters:
            payload["parameters_used"] = parameters

        try:
            self.api_client.post("/api/automation-workflow-log/", payload)
            logger.info("Started workflow run %s for workflow_id %s", run_id, workflow_id)
            return run_id
        except Exception as e:
            logger.error("Failed to start workflow run: %s", e)
            raise

    def update_run_status(
        self,
        run_id: str,
        status: str,
        records_processed: int = 0,
        records_failed: int = 0,
        error_summary: Optional[str] = None,
        error_details: Optional[str] = None,
        execution_metadata: Optional[Dict] = None,
    ):
        """
        Update the status of a running workflow log entry by run_id.
        """
        if error_summary and len(error_summary) > 255:
            error_summary = error_summary[:252] + "..."

        payload: Dict[str, Any] = {
            "status": status,
            "records_processed": records_processed,
            "records_failed": records_failed,
        }
        if error_summary:
            payload["error_summary"] = error_summary
        if error_details:
            payload["error_details"] = error_details
        if execution_metadata:
            payload["execution_metadata"] = execution_metadata
        # Set finished_at for terminal states
        if status in ("success", "failed", "partial_success", "timed_out"):
            payload["finished_at"] = datetime.utcnow().isoformat()

        try:
            self.api_client.patch(
                f"/api/automation-workflow-log/by-run-id/{run_id}", payload
            )
            logger.info("Updated run %s status to %s", run_id, status)
        except Exception as e:
            logger.error("Failed to update run status for %s: %s", run_id, e)
            # Don't raise — avoid crashing the cleanup logic in finally blocks

    def update_schedule_status(self, schedule_id: int):
        """
        Update the schedule's last_run_at via the existing schedule PUT endpoint.
        """
        if not schedule_id:
            return
        try:
            self.api_client.put(
                f"/api/automation-workflow-schedule/{schedule_id}",
                {"last_run_at": datetime.utcnow().isoformat()},
            )
            logger.info("Updated schedule %s last_run_at", schedule_id)
        except Exception as e:
            logger.error("Failed to update schedule %s: %s", schedule_id, e)
