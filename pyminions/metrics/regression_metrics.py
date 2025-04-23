"""
Metric calculation for regression tasks.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    explained_variance_score, max_error, median_absolute_error, 
    mean_squared_log_error, mean_poisson_deviance, mean_gamma_deviance, 
    mean_absolute_percentage_error
)

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calculate metrics for regression tasks.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    dict
        Dictionary containing calculated metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'median_absolute_error': median_absolute_error(y_true, y_pred),
    }
    # Advanced metrics with try/except for edge cases
    try:
        metrics['msle'] = mean_squared_log_error(y_true, y_pred)
        metrics['rmsle'] = np.sqrt(metrics['msle'])
    except Exception:
        metrics['msle'] = np.nan
        metrics['rmsle'] = np.nan
    try:
        metrics['mean_poisson_deviance'] = mean_poisson_deviance(y_true, y_pred)
    except Exception:
        metrics['mean_poisson_deviance'] = np.nan
    try:
        metrics['mean_gamma_deviance'] = mean_gamma_deviance(y_true, y_pred)
    except Exception:
        metrics['mean_gamma_deviance'] = np.nan
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        metrics['mape'] = np.nan
    return metrics
