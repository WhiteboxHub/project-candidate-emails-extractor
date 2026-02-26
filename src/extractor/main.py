from .core.settings import load_config
from .core.logging import setup_logging
from .orchestration.service import EmailExtractionService
from .workflow.manager import WorkflowManager
from .persistence.db_candidate_source import DatabaseCandidateSource
from .connectors.http_api import get_api_client
from .core.constants import DEFAULT_CONFIG_PATH, DEFAULT_LOGGING_CONFIG_PATH
import logging
import sys

logger = logging.getLogger(__name__)

def main():
    setup_logging(DEFAULT_LOGGING_CONFIG_PATH)
    # config = load_config(DEFAULT_CONFIG_PATH) # Config is loaded by Service internal logic or passed via params

    try:
        # 1. Initialize API Client and Workflow Manager
        api_client = get_api_client()
        workflow_manager = WorkflowManager(api_client)
        workflow_key = "email_extractor" # Default key, could be env var

        # 2. Fetch Workflow Configuration
        logger.info(f"Fetching configuration for workflow: {workflow_key}")
        workflow_config = workflow_manager.get_workflow_config(workflow_key)

        if not workflow_config:
            logger.error(f"Workflow '{workflow_key}' not found or inactive.")
            sys.exit(1)

        # 3. Extract SQL and Parameters
        credentials_sql = workflow_config.get("credentials_list_sql")
        workflow_id = workflow_config.get("id")
        parameters = workflow_config.get("parameters_config") or {}

        if not credentials_sql:
            logger.error("No credentials_list_sql found in workflow configuration.")
            sys.exit(1)

        # 4. Initialize Data Source
        candidate_source = DatabaseCandidateSource(credentials_sql)

        # 5. Start Execution Tracking
        run_id = workflow_manager.start_run(
            workflow_id=workflow_id,
            parameters=parameters
        )
        logger.info(f"Initialized run {run_id}")

        # 6. Initialize and Run Service
        service = EmailExtractionService(
            candidate_source=candidate_source,
            workflow_manager=workflow_manager,
            run_id=run_id,
            workflow_id=workflow_id,
            runtime_parameters=parameters
        )
        
        service.run()

    except Exception as e:
        logger.critical(f"Fatal error during initialization: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
