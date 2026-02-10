from .core.settings import load_config
from .core.logging import setup_logging
from .orchestration.service import Service
from .core.constants import DEFAULT_CONFIG_PATH, DEFAULT_LOGGING_CONFIG_PATH

def main():
    setup_logging(DEFAULT_LOGGING_CONFIG_PATH)
    config = load_config(DEFAULT_CONFIG_PATH)
    service = Service(config)
    service.run()

if __name__ == "__main__":
    main()
