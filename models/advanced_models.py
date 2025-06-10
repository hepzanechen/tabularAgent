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
    """TabPFN model engine - TabPFN v2.0 supports both classification and regression with specific constraints."""
    
    def create_model(self, problem_type: str, random_state: int = 42, **kwargs):
        """Create TabPFN model based on problem type."""
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError:
            raise ImportError("TabPFN not installed. Install with: pip install tabpfn")
        
        params = {**self.get_default_params(), **kwargs}
        
        # Auto-detect and use GPU if available
        if 'device' not in params:
            params['device'] = self._auto_detect_device()
        
        print(f"🎯 TabPFN using device: {params['device']}")
        
        if problem_type == 'regression':
            # Remove classification-specific parameters for regressor
            regressor_params = {k: v for k, v in params.items() 
                              if k not in ['N_ensemble_configurations']}
            return TabPFNRegressor(**regressor_params)
        else:
            params['N_ensemble_configurations'] = params.get('N_ensemble_configurations', 32)
            return TabPFNClassifier(**params)
    
    def _auto_detect_device(self):
        """Automatically detect the best available device for TabPFN."""
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # Check if we have enough GPU memory (TabPFN needs ~2-8GB depending on dataset)
                if gpu_memory >= 4.0:
                    print("✅ Sufficient GPU memory for TabPFN")
                    return device
                else:
                    print("⚠️ Limited GPU memory, using CPU as fallback")
                    return 'cpu'
            else:
                print("💻 No GPU available, using CPU")
                return 'cpu'
        except ImportError:
            print("⚠️ PyTorch not available for GPU detection, using CPU")
            return 'cpu'
        except Exception as e:
            print(f"⚠️ GPU detection failed ({e}), using CPU")
            return 'cpu'
    
    def _get_available_devices(self):
        """Get list of available devices for TabPFN."""
        available = ['cpu']  # CPU always available
        try:
            import torch
            if torch.cuda.is_available():
                # Check memory requirement
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 4.0:
                    available.append('cuda')
        except:
            pass
        return available
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              problem_type: str, random_state: int = 42, **custom_params):
        """Train TabPFN with constraint checks for both classification and regression."""
        
        # Check TabPFN constraints (updated for v2.0)
        if len(X_train) > 10000:
            print(f"⚠️ Warning: TabPFN works best with ≤10,000 samples. Current: {len(X_train)}")
            print("Consider using a subset or different model.")
        
        if X_train.shape[1] > 100:
            print(f"⚠️ Warning: TabPFN works best with ≤100 features. Current: {X_train.shape[1]}")
            print("Consider feature selection or different model.")
        
        print(f"🚀 Training TabPFN {problem_type} model with {len(X_train)} samples and {X_train.shape[1]} features")
        
        return super().train(X_train, y_train, problem_type, random_state, **custom_params)
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           problem_type: str, random_state: int = 42):
        """
        TabPFN v2.0 has limited hyperparameter tuning options.
        For regression: mainly device selection
        For classification: device and N_ensemble_configurations
        """
        if not self.supports_tuning():
            print("ℹ️ TabPFN: Using default parameters (limited tuning options)")
            return self.train(X_train, y_train, problem_type, random_state)
        
        print("🎛️ Starting TabPFN hyperparameter tuning...")
        
        # TabPFN-specific parameter tuning
        param_grid = self.tuning_config.get('param_grid', {})
        
        if problem_type == 'regression':
            # For regression, mainly tune device (if GPU available)
            simplified_grid = {k: v for k, v in param_grid.items() 
                             if k in ['device']}
        else:
            # For classification, can tune device and ensemble configurations
            simplified_grid = param_grid
        
        # Filter device options based on actual availability
        if 'device' in simplified_grid:
            available_devices = self._get_available_devices()
            simplified_grid['device'] = [d for d in simplified_grid['device'] 
                                       if d in available_devices]
            if not simplified_grid['device']:
                simplified_grid['device'] = ['cpu']  # Fallback
        
        if not simplified_grid:
            print("ℹ️ No suitable parameters to tune for TabPFN")
            return self.train(X_train, y_train, problem_type, random_state)
        
        print(f"🎛️ Tuning TabPFN parameters: {list(simplified_grid.keys())}")
        
        # Use parent tuning method with simplified grid
        original_grid = self.tuning_config['param_grid']
        self.tuning_config['param_grid'] = simplified_grid
        
        try:
            best_model = super().tune_hyperparameters(X_train, y_train, problem_type, random_state)
        finally:
            # Restore original grid
            self.tuning_config['param_grid'] = original_grid
        
        return best_model


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