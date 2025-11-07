"""
Model comparison framework for Koopman operator learning.

This package provides comprehensive tools for comparing multiple neural
architectures on identical datasets with standardized metrics and protocols.
"""

from .model_comparator import ModelComparator, ModelMetrics, ComparisonResults
from .benchmark_runner import BenchmarkRunner, BenchmarkResults, MemoryMonitor
from .visualization_generator import ComparisonVisualizer
from .results_generator import ComprehensiveResultsGenerator

__all__ = [
    'ModelComparator',
    'ModelMetrics', 
    'ComparisonResults',
    'BenchmarkRunner',
    'BenchmarkResults',
    'MemoryMonitor',
    'ComparisonVisualizer',
    'ComprehensiveResultsGenerator'
]