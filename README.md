# TabularML - Modular DataFrame ML Framework

A **lightweight, modular** machine learning framework designed around the strategy pattern. Works with any tabular data by simply specifying feature columns and target columns, then plugging in feature mining and model training strategies.

## 🎯 Philosophy: Modular Strategy-Based ML

This framework is built on **clean separation of concerns**:

- **🔧 Core Orchestrators**: Lightweight classes that manage the ML pipeline
- **🧩 Strategy Pattern**: Pluggable strategies for feature engineering and model training  
- **📦 Modular Design**: Concrete strategies in separate modules, core stays minimal
- **🔄 Easy Extension**: Add new strategies without modifying core code

**Perfect for**: Any tabular data - financial markets, Kaggle competitions, weather prediction, podcast analytics, house prices, customer churn, etc.

## 🏗️ Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TABULAR ML FRAMEWORK                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   FEATURE MINER     │    │     MODEL TRAINER           │ │
│  │   (Orchestrator)    │    │     (Orchestrator)          │ │
│  │                     │    │                             │ │
│  │ • Manages strategies│    │ • Manages strategies        │ │
│  │ • Preprocessing     │    │ • Training pipeline         │ │
│  │ • Feature validation│    │ • Cross-validation          │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│            │                               │                │
│            ▼                               ▼                │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │ FEATURE STRATEGIES  │    │    MODEL STRATEGIES         │ │
│  │                     │    │                             │ │
│  │ • RollingStats      │    │ • RandomForest              │ │
│  │ • LagFeatures       │    │ • XGBoost                   │ │
│  │ • Polynomial        │    │ • NeuralNetworks            │ │
│  │ • Interactions      │    │ • Ensembles                 │ │
│  │ • Domain-specific   │    │ • Custom models             │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              TABULAR ML (Main Orchestrator)                 │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Basic Usage
```python
import pandas as pd
from tabular_ml import TabularML
from strategies.feature_strategies.basic_features import RollingStatsStrategy
from strategies.model_strategies.sklearn_models import RandomForestStrategy

# 1. Load your data
df = pd.read_csv('your_data.csv')

# 2. Create pipeline with strategies
ml = (TabularML(df, target_col='your_target')
      .add_feature_strategy(RollingStatsStrategy())
      .set_model_strategy(RandomForestStrategy())
      .engineer_features()
      .train_model())

# 3. Get results
ml.get_feature_importance()
ml.summary()
```

### Custom Strategies
```python
from feature_miner import FeatureStrategy
from model_trainer import ModelStrategy

# Create custom feature strategy
class MyFeatureStrategy(FeatureStrategy):
    def create_features(self, df, base_features):
        # Your feature engineering logic here
        return df, new_feature_names

# Create custom model strategy  
class MyModelStrategy(ModelStrategy):
    def train(self, X_train, y_train, problem_type, random_state):
        # Your model training logic here
        return trained_model
    
    def create_model(self, problem_type, random_state):
        # Create fresh model instance
        return model

# Use in pipeline
ml = (TabularML(df, target_col='target')
      .add_feature_strategy(MyFeatureStrategy())
      .set_model_strategy(MyModelStrategy())
      .engineer_features()
      .train_model())
```

## 📁 Project Structure

```
tabular_ml/
├── tabular_ml.py           # Main pipeline orchestrator
├── feature_miner.py        # Feature mining orchestrator + base strategy
├── model_trainer.py        # Model training orchestrator + base strategy
├── requirements.txt        # Minimal dependencies
├── strategies/
│   ├── feature_strategies/
│   │   ├── __init__.py
│   │   └── basic_features.py      # Example feature strategies
│   └── model_strategies/
│       ├── __init__.py
│       └── sklearn_models.py      # Example model strategies
└── README.md
```

## 🛠️ Core Components

### 1. **TabularML** (Main Orchestrator)
- Manages the overall ML pipeline
- Coordinates feature mining and model training
- Provides unified interface for any tabular data

### 2. **FeatureMiner** (Feature Orchestrator)  
- Manages feature engineering strategies
- Handles preprocessing (missing values, encoding, scaling)
- Validates and tracks feature transformations

### 3. **ModelTrainer** (Model Orchestrator)
- Manages model training strategies  
- Handles train/test splits and cross-validation
- Tracks model performance and feature importance

### 4. **Strategy Base Classes**
- `FeatureStrategy`: Abstract base for feature engineering
- `ModelStrategy`: Abstract base for model training
- Easy to extend with custom implementations

## 🧩 Strategy Examples

### Feature Strategies
- **RollingStatsStrategy**: Rolling mean, std, min, max
- **LagFeaturesStrategy**: Time-based lag features  
- **PolynomialStrategy**: Polynomial features (x², x³)
- **InteractionStrategy**: Feature interactions (A×B, A÷B)

### Model Strategies  
- **RandomForestStrategy**: Random Forest models
- **LinearStrategy**: Linear/Logistic regression
- **XGBoostStrategy**: XGBoost models
- **EnsembleStrategy**: Model ensembles

## ✨ Key Benefits

1. **🔧 Modular**: Core stays simple, strategies are pluggable
2. **📈 Scalable**: Easy to add new algorithms without touching core
3. **🎯 Focused**: Each class has single responsibility
4. **🔄 Reusable**: Strategies work across different datasets
5. **🧪 Testable**: Easy to unit test individual components
6. **📖 Readable**: Clear separation makes code self-documenting

## 📊 Usage Examples

### Any Tabular Data
```python
# Financial data
df = pd.read_csv('stock_prices.csv')
ml = TabularML(df, target_col='next_day_return')

# Kaggle competition  
df = pd.read_csv('house_prices.csv')
ml = TabularML(df, target_col='SalePrice')

# Customer analytics
df = pd.read_csv('customer_data.csv') 
ml = TabularML(df, target_col='churn_probability')
```

### Method Chaining
```python
results = (TabularML(df, target_col='target')
           .add_feature_strategy(RollingStatsStrategy())
           .add_feature_strategy(PolynomialStrategy()) 
           .set_model_strategy(RandomForestStrategy())
           .engineer_features()
           .train_model()
           .get_feature_importance())
```

## 🔧 Extending the Framework

### Adding New Feature Strategy
```python
# 1. Create strategy file: strategies/feature_strategies/my_features.py
from feature_miner import FeatureStrategy

class MyAwesomeStrategy(FeatureStrategy):
    def create_features(self, df, base_features):
        # Your logic here
        return modified_df, new_feature_names

# 2. Use it
from strategies.feature_strategies.my_features import MyAwesomeStrategy
ml.add_feature_strategy(MyAwesomeStrategy())
```

### Adding New Model Strategy
```python
# 1. Create strategy file: strategies/model_strategies/my_models.py  
from model_trainer import ModelStrategy

class MyAwesomeModel(ModelStrategy):
    def train(self, X_train, y_train, problem_type, random_state):
        # Your training logic
        return trained_model
    
    def create_model(self, problem_type, random_state):
        # Your model creation logic
        return fresh_model

# 2. Use it
from strategies.model_strategies.my_models import MyAwesomeModel
ml.set_model_strategy(MyAwesomeModel())
```

## 📋 Installation

```bash
pip install -r requirements.txt
```

## 🎉 Run Examples

```bash
python tabular_ml.py
```

## 🎯 Design Principles

1. **Strategy Pattern**: Easy to plug in new algorithms
2. **Single Responsibility**: Each class does one thing well
3. **Open/Closed**: Open for extension, closed for modification
4. **Composition over Inheritance**: Strategies are composed, not inherited
5. **Minimal Core**: Keep orchestrators lightweight and focused

## 📝 Next Steps

The framework is designed to grow incrementally:

1. **Start Simple**: Use basic strategies
2. **Add Complexity**: Implement domain-specific strategies
3. **Scale Up**: Add ensemble methods, AutoML strategies
4. **Optimize**: Add performance monitoring, model versioning

**The goal**: Make it easy to experiment with different approaches while keeping the core framework clean and maintainable! 


🧠 core/ (Framework Logic)
├── feature_miner.py    # Feature orchestration
├── model_trainer.py    # Model orchestration
├── feature_factory.py  # Feature engine creation
└── model_factory.py    # Model engine creation

🔧 features/ (Feature Domain)
├── base.py            # FeatureEngine base
└── basic_stats.py     # Implementation

🔧 models/ (Model Domain)
├── base.py            # ModelEngine base
└── sklearn_models.py  # Implementation