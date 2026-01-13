import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and manage application configuration"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config_path = Path(config_path)
        self._config = None
        load_dotenv()  # Load .env file
        
    def load(self):
        """Load configuration from YAML file with environment variable substitution"""
        try:
            with open(self.config_path, 'r') as f:
                config_str = f.read()
                
            # Replace environment variables
            config_str = self._substitute_env_vars(config_str)
            
            self._config = yaml.safe_load(config_str)
            logger.info(f"Configuration loaded from {self.config_path}")
            return self._config
            
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise
    
    def _substitute_env_vars(self, config_str):
        """Replace ${VAR} with environment variable values"""
        import re
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replacer, config_str)
    
    def get(self, key_path, default=None):
        """Get config value using dot notation (e.g., 'database.host')"""
        if self._config is None:
            self.load()
        
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        
        return value if value is not None else default
    
    @property
    def config(self):
        """Get full configuration dictionary"""
        if self._config is None:
            self.load()
        return self._config


# Singleton instance
_config_loader = None

def get_config():
    """Get global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader
