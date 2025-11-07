"""
Visualization modules for publication-ready figures.

This package provides high-resolution visualization capabilities for:
- Fractal attractor visualizations
- Training curve plots
- Eigenvalue spectrum plots
- Comparative analysis figures
"""

from .publication_figures import PublicationFigureGenerator
from .fractals.fractal_visualizer import FractalVisualizer
from .training.training_visualizer import TrainingVisualizer
from .spectral.spectrum_visualizer import SpectrumVisualizer
from .result_documentation import (
    ResultDocumentationGenerator, 
    ExperimentConfig, 
    ModelResults, 
    ExperimentResults
)
from .figure_manager import FigureManager

__all__ = [
    'PublicationFigureGenerator',
    'FractalVisualizer', 
    'TrainingVisualizer',
    'SpectrumVisualizer',
    'ResultDocumentationGenerator',
    'ExperimentConfig',
    'ModelResults', 
    'ExperimentResults',
    'FigureManager'
]