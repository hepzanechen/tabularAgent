"""
Feature Engine Base Class
Abstract base class for feature engineering engines with config support.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import pandas as pd

class FeatureEngine(ABC):
    """
    Abstract base class for feature engineering engines.
    All concrete feature engines should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize feature engine with configuration.
        
        Args:
            config: Engine configuration dictionary
        """
        self.config = config or {}
        self.params = self.config.get('params', {})
        self.enabled = self.config.get('enabled', True)
        self.description = self.config.get('description', '')
    
    @abstractmethod
    def create_features(self, df: pd.DataFrame, base_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create new features from existing ones.
        
        Args:
            df: pandas DataFrame
            base_features: list of base feature column names
            
        Returns:
            tuple: (modified_df, list_of_new_feature_names)
        """
        pass
    
    def get_engine_name(self) -> str:
        """Get name of the engine."""
        return self.__class__.__name__
        
    def is_enabled(self) -> bool:
        """Check if engine is enabled."""
        return self.enabled
        
    def get_description(self) -> str:
        """Get engine description."""
        return self.description or self.get_engine_name() 