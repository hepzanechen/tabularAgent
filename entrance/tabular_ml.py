"""
TabularML - Main Pipeline Orchestrator
Unified interface for modular tabular machine learning with config-driven engines.
Redesigned for flexible, decoupled feature engineering and model training.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from core.feature_miner import FeatureMiner
from core.model_trainer import ModelTrainer
from core.config_loader import config_loader
from core.feature_factory import feature_engine_factory
from core.model_factory import model_engine_factory

class TabularML:
    """
    Main pipeline orchestrator for tabular machine learning.
    Flexible, decoupled design allowing independent feature engineering and model training.
    Both components share and work on the same DataFrame state.
    """
    
    def __init__(self, df, target_col=None, problem_type='auto', data_type='mixed', config_file=None):
        """
        Initialize the pipeline.
        
        Args:
            df: pandas DataFrame with your data
            target_col: string, name of target column
            problem_type: 'regression', 'classification', or 'auto'
            data_type: 'time_series', 'cross_sectional', or 'mixed' (for feature engine selection)
            config_file: custom config directory path (optional)
        """
        # Data state management
        self.original_df = df.copy()
        self.current_df = df.copy()  # Working DataFrame
        self.target_col = target_col
        self.problem_type = problem_type
        self.data_type = data_type
        
        # Feature state management
        self.available_features = [col for col in df.columns if col != target_col] if target_col else list(df.columns)
        self.base_features = self.available_features.copy()
        self.engineered_features = []
        self.feature_history = []  # Track feature engineering history
        
        # Initialize core components (decoupled)
        self.feature_miner = FeatureMiner(self.base_features, data_type)
        self.model_trainer = ModelTrainer(problem_type)
        
        print(f"TabularML initialized: {df.shape[0]} samples, {df.shape[1]} columns")
        if target_col:
            print(f"Target: {target_col}, Data type: {data_type}")
        print(f"Available features: {len(self.available_features)}")
        
        # Show available engines
        self._show_available_engines()
    
    # ==================== FLEXIBLE FEATURE ENGINEERING ====================
    
    def add_feature_engine(self, engine_name, custom_config=None):
        """Add a feature engine (can be called anytime)."""
        self.feature_miner.add_engine(engine_name, custom_config)
        return self
    
    def apply_feature_engineering(self, features_to_use=None):
        """
        Apply feature engineering to current DataFrame.
        Can be called multiple times, builds on current state.
        
        Args:
            features_to_use: list of features to engineer (None = use all available)
        """
        if not self.feature_miner.feature_engines:
            print("No feature engines configured")
            return self
        
        # Use specified features or all available
        features = features_to_use or self.available_features
        
        print(f"Applying feature engineering to {len(features)} features...")
        
        # Apply feature engineering
        df_engineered, new_features = self.feature_miner.mine_features(
            self.current_df, self.target_col
        )
        
        # Update state
        self.current_df = self.feature_miner.prepare_features(df_engineered, features + new_features)
        self.engineered_features.extend(new_features)
        self.available_features = features + new_features
        
        # Track history
        self.feature_history.append({
            'action': 'feature_engineering',
            'engines_used': len(self.feature_miner.feature_engines),
            'new_features': len(new_features),
            'total_features': len(self.available_features)
        })
        
        print(f"✅ Added {len(new_features)} engineered features")
        print(f"Total available features: {len(self.available_features)}")
        
        return self
    
    def reset_features(self):
        """Reset to original features (remove all engineered features)."""
        self.current_df = self.original_df.copy()
        self.available_features = self.base_features.copy()
        self.engineered_features = []
        self.feature_history.append({'action': 'reset_features'})
        print("✅ Reset to original features")
        return self
    
    def drop_features(self, features_to_drop):
        """Drop specific features from current state."""
        features_to_drop = [f for f in features_to_drop if f in self.available_features]
        self.available_features = [f for f in self.available_features if f not in features_to_drop]
        self.current_df = self.current_df.drop(columns=features_to_drop, errors='ignore')
        
        self.feature_history.append({
            'action': 'drop_features',
            'dropped': features_to_drop,
            'remaining': len(self.available_features)
        })
        print(f"✅ Dropped {len(features_to_drop)} features")
        return self
    
    def select_features(self, features_to_keep):
        """Keep only specified features."""
        if self.target_col and self.target_col not in features_to_keep:
            features_to_keep = features_to_keep + [self.target_col]
        
        self.available_features = [f for f in features_to_keep if f != self.target_col]
        self.current_df = self.current_df[features_to_keep]
        
        self.feature_history.append({
            'action': 'select_features',
            'selected': len(self.available_features)
        })
        print(f"✅ Selected {len(self.available_features)} features")
        return self
    
    # ==================== FLEXIBLE MODEL TRAINING ====================
    
    def set_model_engine(self, engine_name, custom_config=None):
        """Set model engine (can be called anytime)."""
        self.model_trainer.set_model_engine(engine_name, custom_config)
        return self
    
    def train_model(self, features_to_use=None, test_size=None):
        """
        Train model on current DataFrame state.
        Can be called anytime with any feature set.
        
        Args:
            features_to_use: list of features to use (None = use all available)
            test_size: test set size (optional)
        """
        if self.target_col is None:
            raise ValueError("No target column specified")
        
        # Use specified features or all available
        features = features_to_use or self.available_features
        
        # Validate features exist
        missing_features = [f for f in features if f not in self.current_df.columns]
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")
        
        print(f"Training model with {len(features)} features...")
        
        # Train model
        self.model_trainer.train(self.current_df, features, self.target_col, test_size)
        
        print("✅ Model training complete")
        return self
    
    def quick_train(self, engine_name='random_forest', features_to_use=None, test_size=0.2):
        """
        Quick model training for exploration.
        
        Args:
            engine_name: model engine to use
            features_to_use: features to use (None = all available)
            test_size: test set size
        """
        print(f"🚀 Quick training with {engine_name}...")
        self.set_model_engine(engine_name)
        self.train_model(features_to_use, test_size)
        return self
    
    def train_on_raw_features(self, engine_name='random_forest', test_size=0.2):
        """Train model on original raw features only."""
        print("🔧 Training on raw features...")
        self.set_model_engine(engine_name)
        self.train_model(self.base_features, test_size)
        return self
    
    def tune_model(self, features_to_use=None, test_size=None):
        """
        Train model with hyperparameter tuning on current DataFrame state.
        
        Args:
            features_to_use: list of features to use (None = use all available)
            test_size: test set size (optional)
        """
        if self.target_col is None:
            raise ValueError("No target column specified")
        
        # Use specified features or all available
        features = features_to_use or self.available_features
        
        # Validate features exist
        missing_features = [f for f in features if f not in self.current_df.columns]
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")
        
        print(f"🎛️ Tuning model with {len(features)} features...")
        
        # Train model with tuning
        self.model_trainer.tune_and_train(self.current_df, features, self.target_col, test_size)
        
        print("✅ Model tuning and training complete")
        return self
    
    def train_on_raw_features_with_tuning(self, engine_name='sklearnLasso', test_size=0.2):
        """Train model on original raw features with hyperparameter tuning."""
        print("🎛️ Training on raw features with hyperparameter tuning...")
        self.set_model_engine(engine_name)
        self.tune_model(self.base_features, test_size)
        return self
    
    # ==================== PREDICTION & EVALUATION ====================
    
    def predict(self, new_data, features_to_use=None):
        """Make predictions using current model.
        
        Args:
            new_data: pandas DataFrame containing test data
            features_to_use: list of features to use (None = use all available)
        """
        if self.model_trainer.model is None:
            raise ValueError("Model not trained - call train_model() first")
        
        if new_data is None:
            raise ValueError("Test data must be provided")
            
        features = features_to_use or self.available_features
        
        # Validate features exist in test data
        missing_features = [f for f in features if f not in new_data.columns]
        if missing_features:
            raise ValueError(f"Features not found in test data: {missing_features}")
            
        X = new_data[features]
        return self.model_trainer.predict(X)
    
    def cross_validate(self, features_to_use=None, cv=5):
        """Perform cross-validation on current state."""
        features = features_to_use or self.available_features
        return self.model_trainer.cross_validate(self.current_df, features, self.target_col, cv)
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance from current model."""
        return self.model_trainer.get_feature_importance(top_n)
    
    # ==================== STATE MANAGEMENT & SUMMARY ====================
    
    def get_current_state(self):
        """Get current pipeline state."""
        return {
            'df_shape': self.current_df.shape,
            'available_features': len(self.available_features),
            'base_features': len(self.base_features),
            'engineered_features': len(self.engineered_features),
            'model_trained': self.model_trainer.model is not None,
            'model_engine': self.model_trainer.model_engine.get_engine_name() if self.model_trainer.model_engine else None,
            'feature_history_steps': len(self.feature_history)
        }
    
    def feature_summary(self):
        """Show feature engineering summary."""
        print(f"\n{'='*50}")
        print("FEATURE ENGINEERING SUMMARY")
        print(f"{'='*50}")
        print(f"Base features: {len(self.base_features)}")
        print(f"Engineered features: {len(self.engineered_features)}")
        print(f"Total available: {len(self.available_features)}")
        print(f"DataFrame shape: {self.current_df.shape}")
        
        if self.feature_history:
            print(f"\nFeature History ({len(self.feature_history)} steps):")
            for i, step in enumerate(self.feature_history[-3:], 1):  # Show last 3 steps
                print(f"  {i}. {step}")
        
        return self
    
    def model_summary(self):
        """Show model training summary."""
        ms = self.model_trainer.get_model_summary()
        print(f"\n{'='*50}")
        print("MODEL TRAINING SUMMARY")
        print(f"{'='*50}")
        print(f"Model Type: {ms['model_type']}")
        print(f"Engine: {ms.get('engine', 'None')}")
        print(f"Trained: {ms['trained']}")
        
        if ms['metrics']:
            print("\nPerformance Metrics:")
            for metric, value in ms['metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        return self
    
    def summary(self):
        """Show complete pipeline summary."""
        state = self.get_current_state()
        print(f"\n{'='*60}")
        print("TABULAR ML PIPELINE STATE")
        print(f"{'='*60}")
        print(f"Data: {state['df_shape'][0]} samples × {state['df_shape'][1]} columns")
        print(f"Target: {self.target_col}")
        print(f"Problem: {self.problem_type}")
        
        print(f"\nFeatures: {state['base_features']} base + {state['engineered_features']} engineered = {state['available_features']} total")
        print(f"Model: {state['model_engine'] or 'None'} ({'Trained' if state['model_trained'] else 'Not trained'})")
        print(f"Pipeline steps: {state['feature_history_steps']}")
        
        return self
    
    # ==================== CONVENIENCE METHODS ====================
    
    def use_config(self, feature_engines=None, model_engine=None):
        """Configure pipeline using YAML config settings."""
        print("Configuring pipeline from YAML settings...")
        
        if feature_engines is not None:
            self.feature_miner.feature_engines = []
            for engine_name in feature_engines:
                self.feature_miner.add_engine(engine_name)
        
        if model_engine is not None:
            self.model_trainer.set_model_engine(model_engine)
        
        return self
    
    def auto_configure(self):
        """Auto-configure pipeline based on data type and problem type."""
        print(f"Auto-configuring for {self.data_type} data and {self.problem_type} problem...")
        
        default_engines = config_loader.get_feature_defaults(self.data_type)
        self.use_config(feature_engines=default_engines)
        
        if self.problem_type != 'auto':
            default_model = config_loader.get_model_defaults(self.problem_type)
            self.model_trainer.set_model_engine(default_model)
        
        return self
    
    def list_engines(self):
        """List all available engines from configuration."""
        engines = config_loader.list_available_engines()
        
        print("\n🔧 Available Engines:")
        print("=" * 50)
        
        print("📊 Feature Engines:")
        for engine in engines['feature_engines']:
            try:
                config = config_loader.get_feature_engine_config(engine)
                status = "✅ Enabled" if config.get('enabled', False) else "⭕ Disabled"
                print(f"  • {engine:15} {status:12} - {config.get('description', '')}")
            except:
                print(f"  • {engine:15} ❌ Error")
        
        print("\n🤖 Model Engines:")
        for engine in engines['model_engines']:
            try:
                info = model_engine_factory.get_engine_info(engine)
                tuning = "🎛️" if info.get('supports_tuning', False) else "🔧"
                print(f"  • {engine:15} {tuning} - {info.get('description', '')}")
            except:
                print(f"  • {engine:15} ❌ Error")
        
        return engines
    
    def _show_available_engines(self):
        """Show brief summary of available engines."""
        try:
            engines = config_loader.list_available_engines()
            enabled_features = config_loader.get_enabled_feature_engines()
            
            print(f"Available: {len(engines['feature_engines'])} feature engines, {len(engines['model_engines'])} model engines")
            if enabled_features:
                print(f"Auto-enabled: {', '.join(enabled_features)}")
            print("Use .list_engines() for details or .auto_configure() for quick setup")
        except Exception as e:
            print(f"Note: Config not fully loaded: {str(e)}")


# ==================== CONVENIENCE FUNCTIONS ====================

def quick_ml(df, target_col, data_type='mixed', feature_engines=None, model_engine=None):
    """
    Quick setup for common use cases with config-driven approach.
    """
    ml = TabularML(df, target_col=target_col, data_type=data_type)
    
    if feature_engines or model_engine:
        ml.use_config(feature_engines=feature_engines, model_engine=model_engine)
    else:
        ml.auto_configure()
    
    return ml


def config_ml(df, target_col, config_path="config"):
    """
    Setup pipeline using custom configuration directory.
    """
    ml = TabularML(df, target_col=target_col, config_file=config_path)
    return ml


if __name__ == "__main__":
    print("TabularML Framework - Flexible & Decoupled Edition")
    print("="*60)
    print("Enhanced pipeline orchestrator with flexible feature engineering and model training")
    print("\nKey Features:")
    print("✓ Decoupled feature engineering and model training")
    print("✓ Train models anytime on any feature set")
    print("✓ Iterative feature engineering and model exploration")
    print("✓ Shared DataFrame state management")
    print("✓ Feature history tracking")
    print("\nFor examples, see documentation or use .list_engines()") 