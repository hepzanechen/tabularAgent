"""
Model Engine Base Class
Abstract base class for model training engines with config support.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class ModelEngine(ABC):
    """
    Abstract base class for model training engines.
    All concrete model engines should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model engine with configuration.
        
        Args:
            config: Engine configuration dictionary
        """
        self.config = config or {}
        self.default_params = self.config.get('default_params', {})
        self.tuning_config = self.config.get('tuning', {})
        self.description = self.config.get('description', '')
    
    @abstractmethod
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """
        Create a fresh model instance.
        
        Args:
            problem_type: 'regression' or 'classification'
            random_state: Random seed
            **kwargs: Additional parameters
            
        Returns:
            Configured model instance
        """
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              problem_type: str, random_state: int = 42, **custom_params):
        """
        Train and return a model.
        Common implementation for all sklearn-based engines.
        
        Args:
            X_train: Training features
            y_train: Training target
            problem_type: 'regression' or 'classification'
            random_state: Random seed
            **custom_params: Override default parameters
            
        Returns:
            Trained model instance
        """
        model = self.create_model(problem_type, random_state, **custom_params)
        model.fit(X_train, y_train)
        return model
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           problem_type: str, random_state: int = 42):
        """
        Perform hyperparameter tuning using configuration.
        
        Args:
            X_train: Training features
            y_train: Training target
            problem_type: 'regression' or 'classification'
            random_state: Random seed
            
        Returns:
            Best trained model instance
        """
        if not self.supports_tuning():
            print("Warning: This engine does not support hyperparameter tuning")
            return self.train(X_train, y_train, problem_type, random_state)
        
        from sklearn.model_selection import GridSearchCV
        
        # Create base model
        base_model = self.create_model(problem_type, random_state)
        
        # Get tuning configuration
        param_grid = self.tuning_config.get('param_grid', {})
        cv = self.tuning_config.get('cv', 5)
        
        # Set scoring metric based on problem type
        scoring = 'r2' if problem_type == 'regression' else 'accuracy'
        
        print(f"🎛️ Tuning hyperparameters with {len(param_grid)} parameters...")
        print(f"Parameter grid: {param_grid}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best score: {grid_search.best_score_:.4f}")
        print(f"✅ Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def get_engine_name(self) -> str:
        """Get name of the engine."""
        return self.__class__.__name__
        
    def get_description(self) -> str:
        """Get engine description."""
        return self.description or self.get_engine_name()
        
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this engine."""
        return self.default_params.copy()
        
    def get_tuning_config(self) -> Dict[str, Any]:
        """Get hyperparameter tuning configuration."""
        return self.tuning_config.copy()
        
    def supports_tuning(self) -> bool:
        """Check if engine supports hyperparameter tuning."""
        return bool(self.tuning_config) 