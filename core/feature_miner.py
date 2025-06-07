"""
Feature Mining Orchestrator
Manages feature engineering engines and preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from core.config_loader import config_loader
from core.feature_factory import feature_engine_factory

class FeatureMiner:
    """
    Core feature mining orchestrator for tabular data.
    Manages feature engineering engines and preprocessing pipeline.
    """
    
    def __init__(self, feature_cols=None, data_type='mixed'):
        """
        Initialize feature miner.
        
        Args:
            feature_cols: list of column names to use as base features
            data_type: type of data for engine selection ('time_series', 'cross_sectional', 'mixed')
        """
        self.feature_cols = feature_cols
        self.data_type = data_type
        self.scaler = None
        self.label_encoders = {}
        self.engineered_features = []
        self.feature_engines = []
        
        # Auto-load enabled engines from config
        self._auto_load_engines()
    
    def add_engine(self, engine_name, custom_config=None):
        """Add a feature engine by name."""
        try:
            engine = feature_engine_factory.create_engine(engine_name, custom_config)
            self.feature_engines.append(engine)
            print(f"Added feature engine: {engine.get_engine_name()}")
        except Exception as e:
            print(f"Warning: Failed to add engine '{engine_name}': {str(e)}")
        return self
    
    def mine_features(self, df, target_col=None):
        """
        Main feature mining pipeline.
        
        Args:
            df: pandas DataFrame
            target_col: target column name (excluded from features)
            
        Returns:
            tuple: (DataFrame with engineered features, list of all feature columns)
        """
        # Set base feature columns if not specified
        if self.feature_cols is None:
            if target_col:
                self.feature_cols = [col for col in df.columns if col != target_col]
            else:
                self.feature_cols = list(df.columns)
        
        df_processed = df.copy()
        
        # Apply feature engines
        for engine in self.feature_engines:
            if engine.is_enabled():
                df_processed, new_features = engine.create_features(df_processed, self.feature_cols)
                self.engineered_features.extend(new_features)
                print(f"Applied engine '{engine.get_engine_name()}': +{len(new_features)} features")
        
        # Update feature columns to include engineered features
        all_features = self.feature_cols + self.engineered_features
        
        # Remove features with all NaN values
        all_features = [col for col in all_features if col in df_processed.columns and not df_processed[col].isna().all()]
        
        return df_processed, all_features
    
    def prepare_features(self, df, feature_cols):
        """
        Prepare features for modeling (handle missing values, encode categoricals, scale).
        Uses config-driven preprocessing.
        
        Args:
            df: pandas DataFrame
            feature_cols: list of feature column names
            
        Returns:
            DataFrame with prepared features
        """
        df_prepared = df.copy()
        
        # Get preprocessing config
        preprocessing_config = config_loader.get_preprocessing_config()
        
        # Fill missing values using config
        missing_config = preprocessing_config.get('missing_values', {})
        for col in feature_cols:
            if col in df_prepared.columns:
                if df_prepared[col].dtype in ['object', 'category']:
                    strategy = missing_config.get('categorical_strategy', 'mode')
                    if strategy == 'mode':
                        df_prepared[col] = df_prepared[col].fillna(df_prepared[col].mode().iloc[0] if not df_prepared[col].mode().empty else 'missing')
                    else:
                        df_prepared[col] = df_prepared[col].fillna(missing_config.get('constant_value', 'missing'))
                else:
                    strategy = missing_config.get('numeric_strategy', 'median')
                    if strategy == 'median':
                        df_prepared[col] = df_prepared[col].fillna(df_prepared[col].median())
                    elif strategy == 'mean':
                        df_prepared[col] = df_prepared[col].fillna(df_prepared[col].mean())
                    else:
                        df_prepared[col] = df_prepared[col].fillna(0)
        
        # Encode categorical variables using config
        encoding_config = preprocessing_config.get('encoding', {})
        categorical_cols = df_prepared[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        for col in categorical_cols:
            if col in df_prepared.columns:
                method = encoding_config.get('categorical_method', 'label')
                if method == 'label':
                    le = LabelEncoder()
                    df_prepared[col] = le.fit_transform(df_prepared[col].astype(str))
                    self.label_encoders[col] = le
        
        # Scale numeric features using config
        scaling_config = preprocessing_config.get('scaling', {})
        scaling_method = scaling_config.get('method', 'standard')
        
        if scaling_method != 'none':
            numeric_features = df_prepared[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_features:
                if scaling_method == 'standard':
                    self.scaler = StandardScaler()
                # Can add other scalers here based on config
                df_prepared[numeric_features] = self.scaler.fit_transform(df_prepared[numeric_features])
        
        return df_prepared
    
    def get_feature_summary(self):
        """Get summary of engineered features."""
        return {
            'base_features': len(self.feature_cols) if self.feature_cols else 0,
            'engineered_features': len(self.engineered_features),
            'total_features': len(self.feature_cols or []) + len(self.engineered_features),
            'engines_used': len(self.feature_engines)
        }
    
    def _auto_load_engines(self):
        """Auto-load enabled engines from configuration."""
        try:
            enabled_engines = config_loader.get_enabled_feature_engines()
            for engine_name in enabled_engines:
                self.add_engine(engine_name)
        except Exception as e:
            print(f"Note: Could not auto-load engines from config: {str(e)}") 