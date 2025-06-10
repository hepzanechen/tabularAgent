#!/usr/bin/env python3
"""
Test script for TabPFN GPU auto-detection functionality
"""

import numpy as np
import pandas as pd
from entrance.tabular_ml import TabularML

def test_tabpfn_gpu_detection():
    """Test TabPFN GPU auto-detection."""
    print("🧪 Testing TabPFN GPU Auto-Detection")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 500
    x1 = 5 * np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    
    # Create nonlinear target (same as your experiment)
    y = 2*x1**2 + 3*x1**3 + 4*np.sin(x1) + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3,
        'y': y
    })
    
    print(f"📊 Generated test data: {df.shape}")
    print(f"📈 Target function: y = 2*x1² + 3*x1³ + 4*sin(x1) + noise")
    
    # Test TabPFN regression with auto GPU detection
    try:
        print("\n🚀 Initializing TabularML...")
        ml = TabularML(df=df, target_col='y', problem_type='regression')
        
        print("\n🎯 Setting TabPFN engine (should auto-detect GPU)...")
        ml.set_model_engine('tabpfn')
        
        print("\n🏋️ Training TabPFN model...")
        ml.train_on_raw_features(test_size=0.2)
        
        print("\n📊 Model Summary:")
        ml.model_summary()
        
        print("\n✅ TabPFN GPU auto-detection test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during TabPFN test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tabpfn_gpu_detection() 