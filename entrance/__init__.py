"""
Entrance Package - Main Framework Components
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from entrance.tabular_ml import TabularML
from core.feature_miner import FeatureMiner
from core.model_trainer import ModelTrainer

__all__ = ['TabularML', 'FeatureMiner', 'ModelTrainer']