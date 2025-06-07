"""
Scikit-learn Model Engines
Config-driven implementations using scikit-learn models.
All models follow sklearnModelName convention.
"""

import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import (
    Ridge, RidgeClassifier,
    Lasso,
    ElasticNet, LogisticRegression,
    LinearRegression
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from models.base import ModelEngine

class SklearnRandomForest(ModelEngine):
    """Random Forest model engine with config-driven parameters."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create Random Forest model based on problem type."""
        params = {**self.get_default_params(), **kwargs}
        params['random_state'] = random_state
        
        if problem_type == 'regression':
            return RandomForestRegressor(**params)
        else:
            return RandomForestClassifier(**params)


class SklearnRidge(ModelEngine):
    """Ridge regression/classification engine."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create Ridge model based on problem type."""
        params = {**self.get_default_params(), **kwargs}
        
        if problem_type == 'regression':
            params['random_state'] = random_state
            return Ridge(**params)
        else:
            params['random_state'] = random_state
            return RidgeClassifier(**params)


class SklearnLasso(ModelEngine):
    """Lasso regression/classification engine."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create Lasso model based on problem type."""
        params = {**self.get_default_params(), **kwargs}
        
        if problem_type == 'regression':
            params['random_state'] = random_state
            return Lasso(**params)
        else:
            # Use LogisticRegression with L1 penalty for classification
            params['random_state'] = random_state
            params['penalty'] = 'l1'
            params['solver'] = 'liblinear'  # Required for L1 penalty
            return LogisticRegression(**params)


class SklearnElasticNet(ModelEngine):
    """Elastic Net regression engine."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create Elastic Net model."""
        params = {**self.get_default_params(), **kwargs}
        params['random_state'] = random_state
        return ElasticNet(**params)


class SklearnLinear(ModelEngine):
    """Linear regression/classification engine."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create Linear model based on problem type."""
        params = {**self.get_default_params(), **kwargs}
        
        if problem_type == 'regression':
            return LinearRegression(**params)
        else:
            params['random_state'] = random_state
            return LogisticRegression(**params)


class SklearnSVM(ModelEngine):
    """Support Vector Machine engine."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create SVM model based on problem type."""
        params = {**self.get_default_params(), **kwargs}
        params['random_state'] = random_state
        
        if problem_type == 'regression':
            return SVR(**params)
        else:
            return SVC(**params)


class SklearnKNN(ModelEngine):
    """K-Nearest Neighbors engine."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create KNN model based on problem type."""
        params = {**self.get_default_params(), **kwargs}
        
        if problem_type == 'regression':
            return KNeighborsRegressor(**params)
        else:
            return KNeighborsClassifier(**params) 