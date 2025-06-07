"""
Configuration Loader
Handles loading and parsing YAML configuration files for models and features.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """
    Configuration loader for YAML files.
    Provides centralized access to model and feature configurations.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._model_config = None
        self._feature_config = None
        
    def load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from models.yaml"""
        if self._model_config is None:
            config_path = self.config_dir / "models.yaml"
            self._model_config = self._load_yaml(config_path)
        return self._model_config
    
    def load_feature_config(self) -> Dict[str, Any]:
        """Load feature configuration from features.yaml"""  
        if self._feature_config is None:
            config_path = self.config_dir / "features.yaml"
            self._feature_config = self._load_yaml(config_path)
        return self._feature_config
    
    def get_model_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model engine"""
        config = self.load_model_config()
        if engine_name not in config.get('engines', {}):
            raise ValueError(f"Model engine '{engine_name}' not found in configuration")
        return config['engines'][engine_name]
    
    def get_feature_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for a specific feature engine"""
        config = self.load_feature_config()
        if engine_name not in config.get('engines', {}):
            raise ValueError(f"Feature engine '{engine_name}' not found in configuration")
        return config['engines'][engine_name]
    
    def get_model_defaults(self, problem_type: str = None) -> str:
        """Get default model engine for problem type"""
        config = self.load_model_config()
        defaults = config.get('defaults', {})
        
        if problem_type and problem_type in defaults:
            return defaults[problem_type]
        return defaults.get('regression', 'random_forest')  # Fallback
    
    def get_feature_defaults(self, data_type: str = 'mixed') -> list:
        """Get default feature engines for data type"""
        config = self.load_feature_config()
        defaults = config.get('defaults', {})
        return defaults.get(data_type, ['basic_stats'])
    
    def get_enabled_feature_engines(self) -> list:
        """Get list of enabled feature engines"""
        config = self.load_feature_config()
        engines = config.get('engines', {})
        return [name for name, engine_config in engines.items() 
                if engine_config.get('enabled', False)]
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get global training configuration"""
        config = self.load_model_config()
        return config.get('training', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        config = self.load_feature_config()
        return config.get('preprocessing', {})
    
    def list_available_engines(self) -> Dict[str, list]:
        """List all available engines"""
        model_config = self.load_model_config()
        feature_config = self.load_feature_config()
        
        return {
            'model_engines': list(model_config.get('engines', {}).keys()),
            'feature_engines': list(feature_config.get('engines', {}).keys())
        }
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling"""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
                
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {file_path}: {str(e)}")
    
    def reload_configs(self):
        """Force reload of all configurations"""
        self._model_config = None
        self._feature_config = None


# Global configuration loader instance
config_loader = ConfigLoader() 