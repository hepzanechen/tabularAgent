# Config-Driven TabularML System Overview

## 🎯 What We've Implemented

You asked for a more flexible, config-driven system with better decoupling between features and models. Here's what we've built:

### ✅ **Configuration Files**
- `config/models.yaml` - Model engines, parameters, and hyperparameter tuning
- `config/features.yaml` - Feature engines, preprocessing, and data handling

### ✅ **Engine-Based Architecture** 
- Replaced "strategies" with "**engines**" (keeping "strategy" for your future use)
- `FeatureEngine` - Base class for feature engineering
- `ModelEngine` - Base class for model training
- Factory pattern for config-driven instantiation

### ✅ **Clean Main Interface**
- Simplified `TabularML` class with config-driven methods
- Auto-configuration based on data types
- Backward compatibility with existing code

## 🏗️ **New Architecture**

```
TabularML (Main Interface)
├── FeatureMiner (Orchestrator)
│   ├── FeatureEngineFactory → YAML Config → Concrete Engines
│   └── Legacy Strategy Support
└── ModelTrainer (Orchestrator)
    ├── ModelEngineFactory → YAML Config → Concrete Engines
    └── Legacy Strategy Support
```

## 🚀 **Usage Examples**

### **1. Auto-Configuration (Simplest)**
```python
from entrance.tabular_ml import TabularML

# Auto-configure based on data type
ml = TabularML(df, target_col='price', data_type='cross_sectional')
ml.auto_configure()
result = ml.engineer_features().train_model()
```

### **2. Config-Driven Setup**
```python
# Explicit configuration from YAML
ml = TabularML(df, target_col='price')
ml.use_config(
    feature_engines=['basic_stats', 'polynomial'], 
    model_engine='random_forest'
)
result = ml.engineer_features().train_model()
```

### **3. Quick Setup**
```python
from entrance.tabular_ml import quick_ml

# One-liner setup
ml = quick_ml(df, target_col='price', feature_engines=['basic_stats'])
result = ml.engineer_features().train_model()
```

## 📝 **Configuration Management**

### **Model Configuration (`config/models.yaml`)**
```yaml
engines:
  random_forest:
    class: RandomForestEngine
    default_params:
      n_estimators: 100
      max_depth: 20
    tuning:
      method: "grid_search"
      param_grid:
        n_estimators: [50, 100, 200]
        max_depth: [10, 20, null]
```

### **Feature Configuration (`config/features.yaml`)**
```yaml
engines:
  basic_stats:
    class: BasicStatsEngine
    enabled: true
    params:
      include_log: true
      include_sqrt: true
      clip_outliers: true
```

## 🔧 **Key Benefits**

### **1. Flexibility**
- **YAML Configuration**: Change parameters without code changes
- **Engine Factory**: Add new algorithms easily
- **Auto-Selection**: Smart defaults based on data types

### **2. Clean Interface**
- **Decoupled**: Features and models are separate concerns
- **Config-Driven**: Parameters managed in YAML files
- **Method Chaining**: Clean, readable pipeline building

### **3. Extensibility**
- **Engine Pattern**: Easy to add new algorithms
- **Factory System**: Dynamic loading and instantiation
- **Backward Compatible**: Works with existing strategies

### **4. Better Parameter Management**
- **Centralized Config**: All parameters in YAML files
- **Hyperparameter Tuning**: Built-in grid search, random search support
- **Environment-Specific**: Different configs for dev/prod

## 🛠️ **How to Extend**

### **Add New Model Engine**
1. Create engine class inheriting from `ModelEngine`
2. Add to `config/models.yaml`
3. Register in factory (automatic)

### **Add New Feature Engine**
1. Create engine class inheriting from `FeatureEngine`
2. Add to `config/features.yaml`
3. Register in factory (automatic)

### **Custom Configuration**
```python
# Override default config
ml.use_config(
    feature_engines=['custom_engine'],
    model_engine='custom_model'
)

# Custom parameters
ml.set_model_engine('random_forest', custom_config={
    'default_params': {'n_estimators': 500}
})
```

## 📊 **Comparison: Before vs After**

### **Before (Strategy Pattern)**
```python
# Manual strategy creation and parameter setting
from strategies import RandomForestStrategy, PolynomialStrategy

ml = TabularML(df, target_col='price')
ml.add_feature_strategy(PolynomialStrategy(degree=2))
ml.set_model_strategy(RandomForestStrategy(n_estimators=100))
```

### **After (Config-Driven Engines)**
```python
# Clean, config-driven approach
ml = TabularML(df, target_col='price', data_type='cross_sectional')
ml.auto_configure()  # Reads from YAML, selects best engines
# OR
ml.use_config(feature_engines=['polynomial'], model_engine='random_forest')
```

## 🎛️ **Available Engines**

### **Feature Engines**
- `basic_stats` - Log, sqrt, square transformations
- `rolling_stats` - Rolling window statistics (planned)
- `lag_features` - Time-based lag features (planned)
- `polynomial` - Polynomial combinations (planned)
- `interactions` - Feature interactions (planned)

### **Model Engines**
- `random_forest` - Random Forest (regression/classification)
- `linear` - Linear/Logistic regression
- `xgboost` - XGBoost (configuration ready, implementation planned)

## 🔍 **New Interface Methods**

### **Configuration Methods**
- `.auto_configure()` - Auto-setup based on data type
- `.use_config(feature_engines, model_engine)` - Manual config
- `.list_engines()` - Show all available engines

### **Engine Methods**
- `.add_feature_engine(name, config)` - Add feature engine
- `.set_model_engine(name, config)` - Set model engine

### **Legacy Support**
- `.add_feature_strategy()` - Old strategy support
- `.set_model_strategy()` - Old strategy support

## 📁 **New File Structure**

```
├── config/
│   ├── models.yaml      # Model configurations
│   └── features.yaml    # Feature configurations
├── core/
│   ├── config_loader.py    # YAML configuration management
│   ├── feature_factory.py  # Feature engine factory
│   └── model_factory.py    # Model engine factory
├── engines/
│   ├── base_engines.py     # Abstract base classes
│   ├── feature_engines/    # Concrete feature engines
│   └── model_engines/      # Concrete model engines
├── entrance/
│   └── tabular_ml.py       # Enhanced main interface
└── example_config_demo.py  # Usage examples
```

## 🚀 **Getting Started**

1. **Install new dependency**: `pip install PyYAML>=6.0`
2. **Use the new interface**:
   ```python
   from entrance.tabular_ml import TabularML
   ml = TabularML(df, target_col='target')
   ml.auto_configure()
   result = ml.engineer_features().train_model()
   ```
3. **Customize via YAML**: Edit `config/models.yaml` and `config/features.yaml`
4. **Explore engines**: Use `ml.list_engines()` to see available options

The system maintains **100% backward compatibility** while providing a much more flexible, config-driven approach to tabular machine learning! 