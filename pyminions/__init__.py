"""
Model Evaluator - A comprehensive model evaluation framework for machine learning.
"""

from .evaluator import ModelEvaluator
from .visualizations import generate_classification_plots, generate_regression_plots

__version__ = "0.1.0"
__all__ = ['ModelEvaluator', 'generate_classification_plots', 'generate_regression_plots']
