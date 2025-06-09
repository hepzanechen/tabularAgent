"""
Model Engine Factory
Creates model engines based on configuration with hyperparameter tuning support.
"""

import importlib
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from models.base import ModelEngine
from core.config_loader import config_loader

class ModelEngineFactory:
    """
    Factory for creating model engines from configuration.
    Supports dynamic loading, configuration-driven instantiation, and hyperparameter tuning.
    """
    
    def __init__(self):
        """Initialize model engine factory."""
        self._engine_registry = {}
        self._load_builtin_engines()
    
    def create_engine(self, engine_name: str = None, problem_type: str = None, 
                     custom_config: Dict[str, Any] = None) -> ModelEngine:
        """
        Create a model engine by name or auto-select based on problem type.
        
        Args:
            engine_name: Name of the engine to create. If None, uses default for problem_type.
            problem_type: Problem type for auto-selection ('regression' or 'classification')
            custom_config: Optional custom configuration to override defaults
            
        Returns:
            Configured ModelEngine instance
        """
        # Auto-select engine if not specified
        if engine_name is None:
            engine_name = config_loader.get_model_defaults(problem_type)
        
        # Load config from YAML
        engine_config = config_loader.get_model_engine_config(engine_name)
        
        # Override with custom config if provided
        if custom_config:
            engine_config = {**engine_config, **custom_config}
        
        # Get engine class
        engine_class = self._get_engine_class(engine_config['class'])
        
        # Create engine instance
        return engine_class(config=engine_config)
    
    def create_engine_with_tuning(self, engine_name: str = None, problem_type: str = None,
                                 custom_params: Dict[str, Any] = None) -> ModelEngine:
        """
        Create a model engine configured for hyperparameter tuning.
        
        Args:
            engine_name: Name of the engine to create
            problem_type: Problem type ('regression' or 'classification')
            custom_params: Custom parameters to override tuning configuration
            
        Returns:
            ModelEngine configured for hyperparameter tuning
        """
        engine = self.create_engine(engine_name, problem_type)
        
        if not engine.supports_tuning():
            print(f"Warning: Engine '{engine.get_engine_name()}' does not support tuning")
        
        # Override tuning parameters if provided
        if custom_params:
            engine.tuning_config.update(custom_params)
        
        return engine
    
    def register_engine(self, engine_name: str, engine_class: type):
        """
        Register a custom engine class.
        
        Args:
            engine_name: Name to register the engine under
            engine_class: ModelEngine subclass
        """
        if not issubclass(engine_class, ModelEngine):
            raise ValueError(f"Engine class must inherit from ModelEngine")
        
        self._engine_registry[engine_name] = engine_class
    
    def list_available_engines(self) -> list:
        """List all available engine names."""
        config_engines = config_loader.list_available_engines()['model_engines']
        registered_engines = list(self._engine_registry.keys())
        return list(set(config_engines + registered_engines))
    
    def get_engine_info(self, engine_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an engine.
        
        Args:
            engine_name: Name of the engine
            
        Returns:
            Dictionary with engine information
        """
        try:
            config = config_loader.get_model_engine_config(engine_name)
            return {
                'name': engine_name,
                'class': config.get('class'),
                'description': config.get('description', ''),
                'default_params': config.get('default_params', {}),
                'supports_tuning': bool(config.get('tuning', {})),
                'tuning_method': config.get('tuning', {}).get('method', 'none')
            }
        except Exception as e:
            return {'name': engine_name, 'error': str(e)}
    
    def _get_engine_class(self, class_name: str) -> type:
        """
        Get engine class by name with dynamic loading.
        
        Args:
            class_name: Name of the engine class
            
        Returns:
            Engine class
        """
        # Check registry first
        if class_name in self._engine_registry:
            return self._engine_registry[class_name]
        
        # Try to import from sklearn_models for all Sklearn classes
        if class_name.startswith('Sklearn'):
            try:
                module_path = f"models.sklearn_models"
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError):
                pass
        
        # Try to import from advanced_models for advanced ML classes
        advanced_classes = ['LightGBMModel', 'XGBoostModel', 'CatBoostModel', 
                          'TabPFNModel', 'NeuralNetworkModel', 'AutoSklearnModel', 
                          'H2OModel', 'OptunaModel']
        if class_name in advanced_classes:
            try:
                module_path = f"models.advanced_models"
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError):
                pass

        # Try generic model import
        try:
            module_path = f"models"
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            pass
        
        raise ImportError(f"Could not import model engine class: {class_name}")
    
    def _load_builtin_engines(self):
        """Load built-in engine classes."""
        # This will be populated as we create concrete engine implementations
        builtin_engines = {
            # Will be populated with actual engine classes
        }
        
        self._engine_registry.update(builtin_engines)


# Global factory instance
model_engine_factory = ModelEngineFactory() 