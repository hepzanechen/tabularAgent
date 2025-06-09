"""
Advanced Model Engines
Config-driven implementations using advanced ML libraries.
Includes LightGBM, XGBoost, CatBoost, TabPFN, and Neural Networks.
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from models.base import ModelEngine

class LightGBMModel(ModelEngine):
    """LightGBM model engine with config-driven parameters."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create LightGBM model based on problem type."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        params = {**self.get_default_params(), **kwargs}
        params['random_state'] = random_state
        params['verbose'] = -1  # Suppress warnings
        
        if problem_type == 'regression':
            params['objective'] = params.get('objective', 'regression')
            return lgb.LGBMRegressor(**params)
        else:
            params['objective'] = params.get('objective', 'binary')
            return lgb.LGBMClassifier(**params)


class XGBoostModel(ModelEngine):
    """XGBoost model engine with config-driven parameters."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create XGBoost model based on problem type."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        params = {**self.get_default_params(), **kwargs}
        params['random_state'] = random_state
        
        if problem_type == 'regression':
            params['objective'] = params.get('objective', 'reg:squarederror')
            return xgb.XGBRegressor(**params)
        else:
            params['objective'] = params.get('objective', 'binary:logistic')
            return xgb.XGBClassifier(**params)


class CatBoostModel(ModelEngine):
    """CatBoost model engine with config-driven parameters."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create CatBoost model based on problem type."""
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
        except ImportError:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
        
        params = {**self.get_default_params(), **kwargs}
        params['random_state'] = random_state
        params['verbose'] = params.get('verbose', False)  # Suppress output by default
        
        if problem_type == 'regression':
            return CatBoostRegressor(**params)
        else:
            return CatBoostClassifier(**params)


class TabPFNModel(ModelEngine):
    """TabPFN model engine - TabPFN only supports classification with specific constraints."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create TabPFN model - only supports classification."""
        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            raise ImportError("TabPFN not installed. Install with: pip install tabpfn")
        
        if problem_type == 'regression':
            raise ValueError("TabPFN only supports classification tasks")
        
        params = {**self.get_default_params(), **kwargs}
        # TabPFN has specific constraints - max 1000 samples, max 100 features
        params['device'] = params.get('device', 'cpu')
        params['N_ensemble_configurations'] = params.get('N_ensemble_configurations', 32)
        
        return TabPFNClassifier(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              problem_type: str, random_state: int = 42, **custom_params):
        """Train TabPFN with constraint checks."""
        if problem_type == 'regression':
            raise ValueError("TabPFN only supports classification tasks")
        
        # Check TabPFN constraints
        if len(X_train) > 1000:
            print(f"⚠️ Warning: TabPFN works best with ≤1000 samples. Current: {len(X_train)}")
            print("Consider using a subset or different model.")
        
        if X_train.shape[1] > 100:
            print(f"⚠️ Warning: TabPFN works best with ≤100 features. Current: {X_train.shape[1]}")
            print("Consider feature selection or different model.")
        
        return super().train(X_train, y_train, problem_type, random_state, **custom_params)


class NeuralNetworkModel(ModelEngine):
    """Neural Network model engine using sklearn MLPRegressor/Classifier."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create Neural Network model based on problem type."""
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        
        params = {**self.get_default_params(), **kwargs}
        params['random_state'] = random_state
        params['max_iter'] = params.get('max_iter', 1000)
        
        if problem_type == 'regression':
            return MLPRegressor(**params)
        else:
            return MLPClassifier(**params)


class AutoSklearnModel(ModelEngine):
    """Auto-sklearn model engine for automated ML."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create Auto-sklearn model based on problem type."""
        try:
            import autosklearn.regression
            import autosklearn.classification
        except ImportError:
            raise ImportError("Auto-sklearn not installed. Install with: pip install auto-sklearn")
        
        params = {**self.get_default_params(), **kwargs}
        params['seed'] = random_state
        params['time_left_for_this_task'] = params.get('time_left_for_this_task', 120)
        params['per_run_time_limit'] = params.get('per_run_time_limit', 30)
        
        if problem_type == 'regression':
            return autosklearn.regression.AutoSklearnRegressor(**params)
        else:
            return autosklearn.classification.AutoSklearnClassifier(**params)


class H2OModel(ModelEngine):
    """H2O AutoML model engine."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create H2O AutoML model based on problem type."""
        try:
            import h2o
            from h2o.automl import H2OAutoML
        except ImportError:
            raise ImportError("H2O not installed. Install with: pip install h2o")
        
        # Initialize H2O if not already done
        if not h2o.connection().local_server:
            h2o.init()
        
        params = {**self.get_default_params(), **kwargs}
        params['seed'] = random_state
        params['max_models'] = params.get('max_models', 20)
        params['max_runtime_secs'] = params.get('max_runtime_secs', 300)
        
        return H2OAutoML(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              problem_type: str, random_state: int = 42, **custom_params):
        """Train H2O model with data conversion."""
        try:
            import h2o
        except ImportError:
            raise ImportError("H2O not installed. Install with: pip install h2o")
        
        # Convert pandas to H2O frame
        train_data = pd.concat([X_train, y_train], axis=1)
        h2o_train = h2o.H2OFrame(train_data)
        
        # Get target column name
        target_col = y_train.name
        
        model = self.create_model(problem_type, random_state, **custom_params)
        model.train(y=target_col, training_frame=h2o_train)
        
        return model


class OptunaModel(ModelEngine):
    """Optuna-based hyperparameter optimization wrapper for any sklearn model."""
    
    def __init__(self, base_model_class, config: dict = None):
        """
        Initialize with a base model class to optimize.
        
        Args:
            base_model_class: The sklearn model class to optimize
            config: Configuration including Optuna settings
        """
        super().__init__(config)
        self.base_model_class = base_model_class
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create base model - actual optimization happens in tune_hyperparameters."""
        return self.base_model_class(random_state=random_state, **kwargs)
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           problem_type: str, random_state: int = 42):
        """Use Optuna for hyperparameter optimization."""
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
        
        def objective(trial):
            # Define parameter search space based on model type
            params = {}
            param_space = self.tuning_config.get('param_space', {})
            
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Create and evaluate model
            model = self.create_model(problem_type, random_state, **params)
            scoring = 'r2' if problem_type == 'regression' else 'accuracy'
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            return scores.mean()
        
        # Run optimization
        n_trials = self.tuning_config.get('n_trials', 100)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"✅ Best score: {study.best_value:.4f}")
        print(f"✅ Best parameters: {study.best_params}")
        
        # Train final model with best parameters
        best_model = self.create_model(problem_type, random_state, **study.best_params)
        best_model.fit(X_train, y_train)
        
        return best_model 