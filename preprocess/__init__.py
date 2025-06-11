"""
Preprocessing Module for TabularML
==================================

Core preprocessing utilities for memory optimization and data preparation.

Available functions:
- reduce_mem_usage: Optimize DataFrame memory by downcasting numeric types
- optimize_dataframe_memory: Enhanced memory optimization with categorical conversion
"""

from .memory_optimizer import reduce_mem_usage, optimize_dataframe_memory

__version__ = "1.0.0"
__all__ = [
    "reduce_mem_usage",
    "optimize_dataframe_memory"
] 