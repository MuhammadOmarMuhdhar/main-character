import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ConfigError(Exception):
    message: str


class Config:
    def __init__(self, config_path: Optional[str] = None, test_mode: bool = False):
        self.test_mode = test_mode
        self._config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path is None:
            # Point to utils/config.yaml
            utils_dir = Path(__file__).parent.parent / "utils"
            config_path = utils_dir / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ConfigError(f"Error reading config file: {e}")
        
        return config
    
    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        if self.test_mode and isinstance(value, (int, float)):
            test_override = self._get_test_override(path)
            if test_override is not None:
                return test_override
        
        return value
    
    def _get_test_override(self, path: str) -> Any:
        test_config = self._config.get('test_mode', {})
        keys = path.split('.')
        value = test_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def get_collection_config(self) -> Dict[str, Any]:
        return self.get('collection', {})
    
    def get_ratio_detection_config(self) -> Dict[str, Any]:
        return self.get('ratio_detection', {})
    
    def get_engagement_config(self) -> Dict[str, Any]:
        return self.get('engagement', {})
    
    def get_ranking_config(self) -> Dict[str, Any]:
        return self.get('ranking', {})
    
    def get_sentiment_config(self) -> Dict[str, Any]:
        return self.get('sentiment', {})
    
    def get_topics_config(self) -> Dict[str, Any]:
        return self.get('topics', {})
    
    def get_api_limits_config(self) -> Dict[str, Any]:
        return self.get('api_limits', {})
    
    def get_timeouts_config(self) -> Dict[str, Any]:
        return self.get('timeouts', {})
    
    def get_metrics_config(self) -> Dict[str, Any]:
        return self.get('metrics', {})
    
    def get_rolling_window_config(self) -> Dict[str, Any]:
        return self.get('rolling_window', {})
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        return self.get('data_processing', {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        return self.get('pipeline', {})
    
    def get_cli_defaults(self, section: str) -> Dict[str, Any]:
        return self.get(f'cli_defaults.{section}', {})


_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None, test_mode: bool = False) -> Config:
    global _global_config
    
    if _global_config is None or config_path is not None:
        _global_config = Config(config_path, test_mode)
    
    if test_mode != _global_config.test_mode:
        _global_config = Config(config_path, test_mode)
    
    return _global_config


def reset_config():
    global _global_config
    _global_config = None