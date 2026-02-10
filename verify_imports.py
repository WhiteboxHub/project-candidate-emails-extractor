
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_imports")

def test_imports():
    logger.info("Starting import verification...")
    
    try:
        import utils.config
        logger.info("✅ utils.config imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import utils.config: {e}")

    try:
        import utils.logging.logger
        logger.info("✅ utils.logging.logger imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import utils.logging.logger: {e}")

    try:
        import service
        logger.info("✅ service imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import service: {e}")
        
    try:
        import sync_keywords_to_csv
        logger.info("✅ sync_keywords_to_csv imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import sync_keywords_to_csv: {e}")
        
    try:
        from utils.email.connectors import GmailIMAPConnector
        logger.info("✅ utils.email.connectors.GmailIMAPConnector imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import GmailIMAPConnector: {e}")

    logger.info("Verification complete.")

if __name__ == "__main__":
    test_imports()
