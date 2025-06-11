#!/usr/bin/env python3
"""
Fix for ColumnNotFoundError in plotting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fix_plotting_error(df_train, target_column=None):
    """
    Fix the plotting error by checking columns and handling target variable.
    
    Args:
        df_train: Your DataFrame (could be Polars or Pandas)
        target_column: Name of the target column (if known)
    
    Returns:
        df_pandas: DataFrame ready for plotting
        actual_target: Correct target column name
    """
    
    # Convert Polars to Pandas if needed (for seaborn compatibility)
    if hasattr(df_train, 'to_pandas'):
        print("📊 Converting Polars DataFrame to Pandas for plotting...")
        df_pandas = df_train.to_pandas()
    else:
        df_pandas = df_train.copy()
    
    # Check available columns
    print("📋 Available columns:")
    for i, col in enumerate(df_pandas.columns):
        print(f"  {i}: {col}")
    
    # Find the target column
    if target_column and target_column in df_pandas.columns:
        actual_target = target_column
        print(f"✅ Target column '{actual_target}' found")
    else:
        # Try common target column names
        possible_targets = ['label', 'target', 'y', 'class', 'outcome']
        actual_target = None
        
        for possible in possible_targets:
            if possible in df_pandas.columns:
                actual_target = possible
                print(f"✅ Found target column: '{actual_target}'")
                break
        
        if actual_target is None:
            # Use the last column as target (common ML convention)
            actual_target = df_pandas.columns[-1]
            print(f"⚠️ No standard target column found. Using last column: '{actual_target}'")
    
    return df_pandas, actual_target


def create_safe_plots(df_pandas, target_column):
    """
    Create plots safely with proper error handling.
    
    Args:
        df_pandas: Pandas DataFrame
        target_column: Name of the target column
    """
    
    print(f"📊 Creating plots for target column: '{target_column}'")
    
    # Check if target column is numeric for appropriate plots
    is_numeric = pd.api.types.is_numeric_dtype(df_pandas[target_column])
    print(f"📈 Target column is numeric: {is_numeric}")
    
    # Create figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax = ax.flatten()
    
    try:
        # Plot 1: Box plot
        if is_numeric:
            sns.boxplot(x=df_pandas[target_column], ax=ax[0])
            ax[0].set_title(f'Box Plot: {target_column}')
        else:
            # For categorical targets, use count plot
            sns.countplot(data=df_pandas, x=target_column, ax=ax[0])
            ax[0].set_title(f'Count Plot: {target_column}')
            ax[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Histogram/Distribution
        if is_numeric:
            sns.histplot(data=df_pandas, x=target_column, kde=True, ax=ax[1])
            ax[1].set_title(f'Distribution: {target_column}')
        else:
            # For categorical, show value counts
            value_counts = df_pandas[target_column].value_counts()
            ax[1].bar(range(len(value_counts)), value_counts.values)
            ax[1].set_xticks(range(len(value_counts)))
            ax[1].set_xticklabels(value_counts.index, rotation=45)
            ax[1].set_title(f'Value Counts: {target_column}')
        
        # Plot 3: Statistics summary
        ax[2].axis('off')
        if is_numeric:
            stats_text = f"""
            Statistics for {target_column}:
            Mean: {df_pandas[target_column].mean():.4f}
            Median: {df_pandas[target_column].median():.4f}
            Std: {df_pandas[target_column].std():.4f}
            Min: {df_pandas[target_column].min():.4f}
            Max: {df_pandas[target_column].max():.4f}
            Missing: {df_pandas[target_column].isnull().sum()}
            """
        else:
            unique_vals = df_pandas[target_column].nunique()
            most_common = df_pandas[target_column].mode()[0] if len(df_pandas[target_column].mode()) > 0 else "N/A"
            stats_text = f"""
            Statistics for {target_column}:
            Unique values: {unique_vals}
            Most common: {most_common}
            Missing: {df_pandas[target_column].isnull().sum()}
            Total records: {len(df_pandas)}
            """
        
        ax[2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        ax[2].set_title('Statistics Summary')
        
        # Plot 4: Missing values pattern (if any)
        missing_count = df_pandas.isnull().sum().sum()
        if missing_count > 0:
            missing_by_col = df_pandas.isnull().sum()
            missing_by_col = missing_by_col[missing_by_col > 0]
            if len(missing_by_col) > 0:
                ax[3].bar(range(len(missing_by_col)), missing_by_col.values)
                ax[3].set_xticks(range(len(missing_by_col)))
                ax[3].set_xticklabels(missing_by_col.index, rotation=45)
                ax[3].set_title('Missing Values by Column')
                ax[3].set_ylabel('Missing Count')
            else:
                ax[3].axis('off')
                ax[3].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
        else:
            ax[3].axis('off')
            ax[3].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Plots created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("📊 DataFrame info:")
        print(f"   Shape: {df_pandas.shape}")
        print(f"   Columns: {list(df_pandas.columns)}")
        print(f"   Target column type: {df_pandas[target_column].dtype}")


# Quick fix function for immediate use
def quick_fix_and_plot(df_train, target=None):
    """
    Quick fix for the plotting error - use this directly.
    
    Usage:
        quick_fix_and_plot(df_train)
        # or specify target:
        quick_fix_and_plot(df_train, target='your_column_name')
    """
    
    print("🔧 Quick Fix for Plotting Error")
    print("=" * 40)
    
    # Fix the DataFrame and find target
    df_fixed, actual_target = fix_plotting_error(df_train, target)
    
    # Create the plots
    create_safe_plots(df_fixed, actual_target)
    
    return df_fixed, actual_target


if __name__ == "__main__":
    # Example usage
    print("🧪 Example fix for plotting error")
    print("Replace 'df_train' and 'target' with your actual variables:")
    print()
    print("# Your original code that caused the error:")
    print("# sns.boxplot(x=df_train[target], ax=ax[0])")
    print()
    print("# Fixed version:")
    print("from fix_plotting_error import quick_fix_and_plot")
    print("df_fixed, actual_target = quick_fix_and_plot(df_train)") 
 