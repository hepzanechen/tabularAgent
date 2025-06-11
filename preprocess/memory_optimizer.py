"""
Memory Optimization Utilities
=============================

This module provides utilities for reducing memory usage of pandas DataFrames
through intelligent dtype optimization and memory management strategies.

Functions:
- reduce_mem_usage: Optimize DataFrame memory by downcasting numeric types
- optimize_dataframe_memory: Enhanced memory optimization with additional features
- get_memory_usage: Get detailed memory usage statistics
- compare_memory_usage: Compare memory usage before/after optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List


def reduce_mem_usage(dataframe: pd.DataFrame, dataset: str = "Dataset") -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by optimizing numeric dtypes.
    
    Function adapted from: https://www.kaggle.com/code/ravaghi/drw-crypto-market-prediction-ensemble
    
    Args:
        dataframe: Input pandas DataFrame to optimize
        dataset: Name of the dataset for logging purposes
        
    Returns:
        DataFrame with optimized memory usage
        
    Example:
        >>> df_optimized = reduce_mem_usage(df, "training_data")
        Reducing memory usage for: training_data
        --- Memory usage before: 152.34 MB
        --- Memory usage after: 45.67 MB
        --- Decreased memory usage by 70.0%
    """
    print(f'Reducing memory usage for: {dataset}')
    initial_mem_usage = dataframe.memory_usage(deep=True).sum() / 1024**2
    
    # Create a copy to avoid modifying the original
    df_optimized = dataframe.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # Skip non-numeric columns
        if col_type == 'object' or col_type.name == 'category':
            continue
            
        c_min = df_optimized[col].min()
        c_max = df_optimized[col].max()
        
        # Optimize integer types
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df_optimized[col] = df_optimized[col].astype(np.int64)
        
        # Optimize float types
        elif str(col_type)[:5] == 'float':
            # Check if values can fit in float16 (be more conservative)
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                # Additional check for precision loss
                test_conversion = df_optimized[col].astype(np.float16).astype(np.float64)
                if np.allclose(df_optimized[col], test_conversion, rtol=1e-3):
                    df_optimized[col] = df_optimized[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_optimized[col] = df_optimized[col].astype(np.float32)
            else:
                df_optimized[col] = df_optimized[col].astype(np.float64)

    final_mem_usage = df_optimized.memory_usage(deep=True).sum() / 1024**2
    
    print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
    print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
    
    if initial_mem_usage > 0:
        reduction_percent = 100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage
        print(f'--- Decreased memory usage by {reduction_percent:.1f}%\n')
    else:
        print('--- No memory reduction achieved\n')

    return df_optimized


def optimize_dataframe_memory(
    dataframe: pd.DataFrame, 
    dataset_name: str = "Dataset",
    optimize_strings: bool = True,
    optimize_categories: bool = True,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Enhanced memory optimization with additional features.
    
    Args:
        dataframe: Input pandas DataFrame
        dataset_name: Name for logging purposes
        optimize_strings: Whether to optimize string columns to categories
        optimize_categories: Whether to optimize existing categorical columns
        inplace: Whether to modify the original DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    if not inplace:
        df = dataframe.copy()
    else:
        df = dataframe
    
    print(f'🔧 Optimizing memory usage for: {dataset_name}')
    initial_mem = get_memory_usage(df)
    
    # First apply numeric optimization
    df = reduce_mem_usage(df, dataset_name)
    
    # Optimize string columns to categories if beneficial
    if optimize_strings:
        for col in df.select_dtypes(include=['object']).columns:
            # Only convert to category if it reduces memory
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    original_mem = df[col].memory_usage(deep=True)
                    df[col] = df[col].astype('category')
                    new_mem = df[col].memory_usage(deep=True)
                    
                    if new_mem >= original_mem:  # If no benefit, revert
                        df[col] = df[col].astype('object')
                    else:
                        print(f'   ✅ Converted {col} to category (saved {(original_mem-new_mem)/1024**2:.2f} MB)')
    
    # Optimize existing categorical columns
    if optimize_categories:
        for col in df.select_dtypes(include=['category']).columns:
            df[col] = df[col].cat.remove_unused_categories()
    
    final_mem = get_memory_usage(df)
    total_reduction = ((initial_mem - final_mem) / initial_mem) * 100
    
    print(f'🎯 Total memory reduction: {total_reduction:.1f}% ({initial_mem:.2f} MB → {final_mem:.2f} MB)\n')
    
    return df


def get_memory_usage(df: pd.DataFrame) -> float:
    """
    Get total memory usage of a DataFrame in MB.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory usage in megabytes
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def get_detailed_memory_info(df: pd.DataFrame) -> Dict:
    """
    Get detailed memory usage information by column and dtype.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with detailed memory statistics
    """
    memory_info = {
        'total_mb': get_memory_usage(df),
        'by_column': {},
        'by_dtype': {},
        'shape': df.shape
    }
    
    # Memory by column
    for col in df.columns:
        memory_info['by_column'][col] = {
            'dtype': str(df[col].dtype),
            'memory_mb': df[col].memory_usage(deep=True) / 1024**2,
            'null_count': df[col].isnull().sum(),
            'unique_count': df[col].nunique()
        }
    
    # Memory by dtype
    for dtype in df.dtypes.unique():
        cols_with_dtype = df.select_dtypes(include=[dtype]).columns
        total_mem = sum(df[col].memory_usage(deep=True) for col in cols_with_dtype) / 1024**2
        memory_info['by_dtype'][str(dtype)] = {
            'columns': list(cols_with_dtype),
            'total_memory_mb': total_mem,
            'column_count': len(cols_with_dtype)
        }
    
    return memory_info


def compare_memory_usage(df_before: pd.DataFrame, df_after: pd.DataFrame, 
                        dataset_name: str = "Dataset") -> None:
    """
    Compare memory usage between two DataFrames and print detailed comparison.
    
    Args:
        df_before: DataFrame before optimization
        df_after: DataFrame after optimization  
        dataset_name: Name for logging purposes
    """
    mem_before = get_memory_usage(df_before)
    mem_after = get_memory_usage(df_after)
    reduction = ((mem_before - mem_after) / mem_before) * 100
    
    print(f"📊 Memory Comparison for {dataset_name}")
    print("=" * 50)
    print(f"Before optimization: {mem_before:.2f} MB")
    print(f"After optimization:  {mem_after:.2f} MB")
    print(f"Memory saved:        {mem_before - mem_after:.2f} MB")
    print(f"Reduction:           {reduction:.1f}%")
    
    # Dtype comparison
    print("\n📈 Dtype Changes:")
    for col in df_before.columns:
        if str(df_before[col].dtype) != str(df_after[col].dtype):
            before_mem = df_before[col].memory_usage(deep=True) / 1024**2
            after_mem = df_after[col].memory_usage(deep=True) / 1024**2
            print(f"  {col}: {df_before[col].dtype} → {df_after[col].dtype} "
                  f"({before_mem:.2f} MB → {after_mem:.2f} MB)")


def suggest_memory_optimizations(df: pd.DataFrame) -> List[str]:
    """
    Analyze DataFrame and suggest memory optimization strategies.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        List of optimization suggestions
    """
    suggestions = []
    
    # Check for potential categorical conversions
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.1:
            suggestions.append(f"Convert '{col}' to category (only {unique_ratio:.1%} unique values)")
    
    # Check for oversized numeric types
    for col in df.select_dtypes(include=[np.number]).columns:
        if str(df[col].dtype) == 'int64':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                suggestions.append(f"Downcast '{col}' from int64 to int8")
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                suggestions.append(f"Downcast '{col}' from int64 to int16")
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                suggestions.append(f"Downcast '{col}' from int64 to int32")
        
        elif str(df[col].dtype) == 'float64':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                suggestions.append(f"Downcast '{col}' from float64 to float32")
    
    # Check for unused categories
    for col in df.select_dtypes(include=['category']).columns:
        if hasattr(df[col].cat, 'remove_unused_categories'):
            unused = len(df[col].cat.categories) - df[col].nunique()
            if unused > 0:
                suggestions.append(f"Remove {unused} unused categories from '{col}'")
    
    return suggestions 