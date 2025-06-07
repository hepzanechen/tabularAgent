"""
Config-Driven TabularML Demo
Demonstrates the new YAML configuration system and engine-based architecture.
"""

import pandas as pd
import numpy as np
from entrance.tabular_ml import TabularML, quick_ml

# Create sample data
def create_sample_data():
    """Create sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random.uniform(0, 10, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y'], n_samples, p=[0.7, 0.3])
    }
    
    # Generate target (regression)
    target = (data['feature_1'] * 2 + 
              data['feature_2'] * 0.5 + 
              np.log(data['feature_3'] + 1) + 
              np.random.normal(0, 0.5, n_samples))
    
    data['target'] = target
    
    return pd.DataFrame(data)

def demo_config_driven_approach():
    """Demonstrate the new config-driven approach."""
    print("🚀 Config-Driven TabularML Demo")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    
    # Method 1: Auto-configuration
    print("\n📋 Method 1: Auto-Configuration")
    print("-" * 30)
    
    ml1 = TabularML(df, target_col='target', data_type='cross_sectional')
    ml1.auto_configure()
    
    # Run the pipeline
    result1 = (ml1.engineer_features()
               .train_model()
               .summary())
    
    # Method 2: Manual configuration with YAML
    print("\n⚙️ Method 2: Manual Configuration")
    print("-" * 30)
    
    ml2 = TabularML(df, target_col='target')
    ml2.use_config(
        feature_engines=['basic_stats'],
        model_engine='random_forest'
    )
    
    # Run the pipeline
    result2 = (ml2.engineer_features()
               .train_model()
               .summary())
    
    # Method 3: Quick setup
    print("\n⚡ Method 3: Quick Setup")
    print("-" * 30)
    
    ml3 = quick_ml(
        df, 
        target_col='target',
        data_type='cross_sectional',
        feature_engines=['basic_stats'],
        model_engine='linear'
    )
    
    result3 = (ml3.engineer_features()
               .train_model()
               .summary())
    
    # Compare approaches
    print("\n📊 Cross-Validation Comparison")
    print("-" * 30)
    
    print("Auto-configured model:")
    ml1.cross_validate()
    
    print("\nManual-configured model:")
    ml2.cross_validate()
    
    print("\nQuick-setup model:")
    ml3.cross_validate()

def demo_list_engines():
    """Demonstrate engine listing functionality."""
    print("\n🔧 Available Engines Demo")
    print("=" * 50)
    
    # Create a simple TabularML instance
    df = create_sample_data()
    ml = TabularML(df, target_col='target')
    
    # List all available engines
    ml.list_engines()

def demo_backward_compatibility():
    """Demonstrate backward compatibility with old strategy approach."""
    print("\n🔄 Backward Compatibility Demo")
    print("=" * 50)
    
    # This would work with old strategy classes if they exist
    df = create_sample_data()
    ml = TabularML(df, target_col='target')
    
    # You can still mix engines and strategies
    ml.add_feature_engine('basic_stats')
    # ml.add_feature_strategy(SomeOldStrategy())  # If you have old strategies
    # ml.set_model_strategy(SomeOldModelStrategy())  # If you have old strategies
    
    print("Mixed approach (engines + strategies) is supported!")

if __name__ == "__main__":
    try:
        # Main demonstrations
        demo_config_driven_approach()
        demo_list_engines()
        demo_backward_compatibility()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey benefits of the new system:")
        print("• YAML-based configuration for easy parameter management")
        print("• Auto-configuration based on data types")
        print("• Clean separation between feature and model processing")
        print("• Backward compatibility with existing code")
        print("• Factory pattern for easy extensibility")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("\nThis is expected if dependencies are not fully set up.")
        print("The config system provides a clean architecture for:")
        print("• Managing model parameters via YAML files")
        print("• Auto-selecting appropriate engines based on data type")
        print("• Easy extension with new algorithms")
        print("• Consistent interface across different model types") 