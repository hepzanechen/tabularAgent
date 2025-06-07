"""
Model Training Orchestrator
Manages model training, evaluation, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Dict, Any, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from core.config_loader import config_loader
from core.model_factory import model_engine_factory

class ModelTrainer:
    """
    Core model training orchestrator.
    Manages model training, evaluation, and hyperparameter tuning.
    """
    
    def __init__(self, problem_type='auto'):
        """
        Initialize model trainer.
        
        Args:
            problem_type: 'regression', 'classification', or 'auto'
        """
        self.problem_type = problem_type
        self.model = None
        self.model_engine = None
        self.metrics = {}
    
    def set_model_engine(self, engine_name, custom_config=None):
        """Set model engine by name."""
        try:
            self.model_engine = model_engine_factory.create_engine(
                engine_name, 
                problem_type=self.problem_type,
                custom_config=custom_config
            )
            print(f"Set model engine: {self.model_engine.get_engine_name()}")
        except Exception as e:
            print(f"Warning: Failed to set engine '{engine_name}': {str(e)}")
        return self
    
    def train(self, df, feature_cols, target_col, test_size=None):
        """
        Train the model.
        
        Args:
            df: pandas DataFrame
            feature_cols: list of feature column names
            target_col: target column name
            test_size: test set size (optional)
        """
        if self.model_engine is None:
            raise ValueError("No model engine set - call set_model_engine() first")
        
        # Prepare data
        X = df[feature_cols]
        y = df[target_col]
        
        if test_size:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # Train model
        self.model = self.model_engine.train(
            X_train, y_train,
            problem_type=self.problem_type
        )
        
        # Evaluate if test set provided
        if X_test is not None:
            self._evaluate(X_test, y_test)
        
        return self
    
    def tune_and_train(self, df, feature_cols, target_col, test_size=None):
        """
        Train the model with hyperparameter tuning.
        
        Args:
            df: pandas DataFrame
            feature_cols: list of feature column names
            target_col: target column name
            test_size: test set size (optional)
        """
        if self.model_engine is None:
            raise ValueError("No model engine set - call set_model_engine() first")
        
        print("🎛️ Starting hyperparameter tuning...")
        
        # Prepare data
        X = df[feature_cols]
        y = df[target_col]
        
        if test_size:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # Train model with tuning
        self.model = self.model_engine.tune_hyperparameters(
            X_train, y_train,
            problem_type=self.problem_type
        )
        
        # Evaluate if test set provided
        if X_test is not None:
            self._evaluate(X_test, y_test)
        
        print("✅ Hyperparameter tuning and training complete")
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained - call train() first")
        return self.model.predict(X)
    
    def cross_validate(self, df, feature_cols, target_col, cv=5):
        """Perform cross-validation."""
        if self.model_engine is None:
            raise ValueError("No model engine set - call set_model_engine() first")
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Create fresh model for CV
        model = self.model_engine.create_model(self.problem_type)
        
        # Get scoring metric based on problem type
        scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'
        
        # Perform CV
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not trained - call train() first")
        
        if not hasattr(self.model, 'feature_importances_'):
            print("Warning: Model does not support feature importance")
            return None
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        # Return top N features
        return {
            'features': feature_names[indices[:top_n]].tolist(),
            'importances': importances[indices[:top_n]].tolist()
        }
    
    def get_model_summary(self):
        """Get summary of model training."""
        return {
            'model_type': self.problem_type,
            'engine': self.model_engine.get_engine_name() if self.model_engine else None,
            'trained': self.model is not None,
            'metrics': self.metrics
        }
    
    def _evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        if self.model is None:
            return
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics based on problem type
        if self.problem_type == 'regression':
            from sklearn.metrics import r2_score, mean_squared_error
            self.metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        else:
            from sklearn.metrics import accuracy_score, f1_score
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted')
            } 