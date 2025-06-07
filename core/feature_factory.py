"""
Feature Engine Factory
Creates feature engines based on configuration with dynamic loading.
"""

import importlib
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from features.base import FeatureEngine
from core.config_loader import config_loader

class FeatureEngineFactory:
    """
    Factory for creating feature engines from configuration.
    Supports dynamic loading and configuration-driven instantiation.
    """
    
    def __init__(self):
        """Initialize feature engine factory."""
        self._engine_registry = {}
        self._load_builtin_engines()
    
    def create_engine(self, engine_name: str, custom_config: Dict[str, Any] = None) -> FeatureEngine:
        """
        Create a feature engine by name.
        
        Args:
            engine_name: Name of the engine to create
            custom_config: Optional custom configuration to override defaults
            
        Returns:
            Configured FeatureEngine instance
        """
        # Load config from YAML
        engine_config = config_loader.get_feature_engine_config(engine_name)
        
        # Override with custom config if provided
        if custom_config:
            engine_config = {**engine_config, **custom_config}
        
        # Get engine class
        engine_class = self._get_engine_class(engine_config['class'])
        
        # Create engine instance
        return engine_class(config=engine_config)
    
    def create_engines_from_config(self, engine_names: List[str] = None, 
                                  data_type: str = 'mixed') -> List[FeatureEngine]:
        """
        Create multiple engines from configuration.
        
        Args:
            engine_names: List of engine names to create. If None, uses enabled engines.
            data_type: Data type for default engine selection
            
        Returns:
            List of configured FeatureEngine instances
        """
        if engine_names is None:
            # Use enabled engines from config
            engine_names = config_loader.get_enabled_feature_engines()
            
            # If no engines are enabled, use defaults for data type
            if not engine_names:
                engine_names = config_loader.get_feature_defaults(data_type)
        
        engines = []
        for engine_name in engine_names:
            try:
                engine = self.create_engine(engine_name)
                if engine.is_enabled():
                    engines.append(engine)
            except Exception as e:
                print(f"Warning: Failed to create engine '{engine_name}': {str(e)}")
        
        return engines
    
    def register_engine(self, engine_name: str, engine_class: type):
        """
        Register a custom engine class.
        
        Args:
            engine_name: Name to register the engine under
            engine_class: FeatureEngine subclass
        """
        if not issubclass(engine_class, FeatureEngine):
            raise ValueError(f"Engine class must inherit from FeatureEngine")
        
        self._engine_registry[engine_name] = engine_class
    
    def list_available_engines(self) -> List[str]:
        """List all available engine names."""
        config_engines = config_loader.list_available_engines()['feature_engines']
        registered_engines = list(self._engine_registry.keys())
        return list(set(config_engines + registered_engines))
    
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
        
        # Try to import from features package (specific implementations)
        try:
            if class_name == 'BasicStatsEngine':
                module_path = f"features.basic_stats"
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
        except (ImportError, AttributeError):
            pass
        
        # Try generic features import
        try:
            module_path = f"features"
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            pass
        
        raise ImportError(f"Could not import feature engine class: {class_name}")
    
    def _load_builtin_engines(self):
        """Load built-in engine classes."""
        # This will be populated as we create concrete engine implementations
        builtin_engines = {
            # Will be populated with actual engine classes
        }
        
        self._engine_registry.update(builtin_engines)


# Global factory instance
feature_engine_factory = FeatureEngineFactory() 