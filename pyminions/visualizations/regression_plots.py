"""
Visualization generation for regression tasks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.regressor import (
    PredictionError, ResidualsPlot
)

def generate_regression_plots(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = ".",
    model: object = None
):
    """
    Generate visualizations for regression tasks.
    
    Parameters
    ----------
    X_test : array-like
        Test features
    y_test : array-like
        True values
    y_pred : array-like
        Predicted values
    output_dir : str
        Directory to save plots
    model : object
        Trained model instance
    """
    if model is None:
        raise ValueError("Model must be provided for visualization generation")

    # Prediction Error Plot
    viz = PredictionError(model)
    viz.fit(X_test, y_test)
    viz.score(X_test, y_test)
    viz.show(outpath=os.path.join(output_dir, 'prediction_error.png'))
    plt.close()

    # Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), indices)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()

    # Residuals Plot
    viz = ResidualsPlot(model)
    viz.fit(X_test, y_test)
    viz.score(X_test, y_test)
    viz.show(outpath=os.path.join(output_dir, 'residuals_plot.png'))
    plt.close()

    # Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), indices)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
