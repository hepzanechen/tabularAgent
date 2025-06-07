"""
Basic Feature Engines
Config-driven implementations for basic statistical feature transformations.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from features.base import FeatureEngine

class BasicStatsEngine(FeatureEngine):
    """Basic statistical transformations engine with config-driven parameters."""
    
    def create_features(self, df: pd.DataFrame, base_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create basic statistical features from numeric columns.
        
        Args:
            df: Input DataFrame
            base_features: List of base feature column names
            
        Returns:
            tuple: (DataFrame with new features, list of new feature names)
        """
        if not self.is_enabled():
            return df, []
        
        df_processed = df.copy()
        new_features = []
        
        # Get numeric columns only
        numeric_cols = df_processed[base_features].select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            # Skip if column has negative values for log/sqrt (unless clipping is enabled)
            has_negative = (df_processed[col] < 0).any()
            has_zero = (df_processed[col] == 0).any()
            
            # Log transformation
            if self.params.get('include_log', True):
                if has_negative or has_zero:
                    if self.params.get('clip_outliers', True):
                        # Use log1p for positive values
                        log_col = f"{col}_log1p"
                        df_processed[log_col] = np.log1p(np.maximum(df_processed[col], 0))
                        new_features.append(log_col)
                else:
                    log_col = f"{col}_log"
                    df_processed[log_col] = np.log(df_processed[col])
                    new_features.append(log_col)
            
            # Square root transformation
            if self.params.get('include_sqrt', True):
                if has_negative:
                    if self.params.get('clip_outliers', True):
                        # Use sqrt of absolute value
                        sqrt_col = f"{col}_sqrt_abs"
                        df_processed[sqrt_col] = np.sqrt(np.abs(df_processed[col]))
                        new_features.append(sqrt_col)
                else:
                    sqrt_col = f"{col}_sqrt"
                    df_processed[sqrt_col] = np.sqrt(df_processed[col])
                    new_features.append(sqrt_col)
            
            # Square transformation
            if self.params.get('include_square', True):
                square_col = f"{col}_square"
                df_processed[square_col] = df_processed[col] ** 2
                new_features.append(square_col)
            
            # Reciprocal transformation
            if self.params.get('include_reciprocal', False):
                if not has_zero:
                    recip_col = f"{col}_reciprocal"
                    df_processed[recip_col] = 1.0 / df_processed[col]
                    new_features.append(recip_col)
                elif self.params.get('clip_outliers', True):
                    # Add small epsilon to avoid division by zero
                    recip_col = f"{col}_reciprocal_safe"
                    epsilon = 1e-8
                    df_processed[recip_col] = 1.0 / (df_processed[col] + epsilon)
                    new_features.append(recip_col)
        
        # Handle outliers in new features if enabled
        if self.params.get('clip_outliers', True) and new_features:
            threshold = self.params.get('outlier_threshold', 3.0)
            for col in new_features:
                if col in df_processed.columns:
                    # Use Z-score based outlier detection
                    mean_val = df_processed[col].mean()
                    std_val = df_processed[col].std()
                    
                    if std_val > 0:  # Avoid division by zero
                        z_scores = np.abs((df_processed[col] - mean_val) / std_val)
                        outlier_mask = z_scores > threshold
                        
                        if outlier_mask.any():
                            # Clip outliers to threshold boundaries
                            lower_bound = mean_val - threshold * std_val
                            upper_bound = mean_val + threshold * std_val
                            df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
        
        return df_processed, new_features