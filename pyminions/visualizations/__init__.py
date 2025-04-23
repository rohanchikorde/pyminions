"""
Visualization generation module for model evaluation.
"""

from .classification_plots import generate_classification_plots
from .regression_plots import generate_regression_plots

__all__ = ['generate_classification_plots', 'generate_regression_plots']
