"""
Interactive visualization generation using Plotly for model evaluation.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix

def generate_classification_interactive_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    importances: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Generate interactive Plotly visualizations for classification models.
    
    Parameters
    ----------
    y_test : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    y_prob : np.ndarray, optional
        Predicted probabilities
    feature_names : List[str], optional
        Names of features
    importances : np.ndarray, optional
        Feature importance values
    class_names : List[str], optional
        Names of classes
    
    Returns
    -------
    Dict[str, str]
        Dictionary with HTML strings for each interactive plot
    """
    results = {}
    
    # Get class names if not provided
    if class_names is None:
        class_names = list(map(str, np.unique(y_test)))
    
    # Confusion Matrix
    results['confusion_matrix'] = plot_confusion_matrix(
        y_test, y_pred, class_names=class_names
    ).to_html(full_html=False, include_plotlyjs='cdn')
    
    # ROC Curve (if probabilities are available)
    if y_prob is not None:
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            prob_positive = y_prob[:, 1]  # For binary classification
        else:
            prob_positive = y_prob
        results['roc_curve'] = plot_roc_curve(
            y_test, prob_positive
        ).to_html(full_html=False, include_plotlyjs=False)
    
    # Feature Importance (if available)
    if importances is not None and feature_names is not None:
        results['feature_importance'] = plot_feature_importance(
            importances, feature_names
        ).to_html(full_html=False, include_plotlyjs=False)
    
    return results

def generate_regression_interactive_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Optional[List[str]] = None,
    importances: Optional[np.ndarray] = None
) -> Dict[str, str]:
    """
    Generate interactive Plotly visualizations for regression models.
    
    Parameters
    ----------
    y_test : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    feature_names : List[str], optional
        Names of features
    importances : np.ndarray, optional
        Feature importance values
    
    Returns
    -------
    Dict[str, str]
        Dictionary with HTML strings for each interactive plot
    """
    results = {}
    
    # Prediction Error Plot
    results['prediction_error'] = plot_prediction_error(
        y_test, y_pred
    ).to_html(full_html=False, include_plotlyjs='cdn')
    
    # Residuals Plot
    results['residuals'] = plot_residuals(
        y_test, y_pred
    ).to_html(full_html=False, include_plotlyjs=False)
    
    # Feature Importance (if available)
    if importances is not None and feature_names is not None:
        results['feature_importance'] = plot_feature_importance(
            importances, feature_names
        ).to_html(full_html=False, include_plotlyjs=False)
    
    return results

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None) -> go.Figure:
    """
    Create an interactive confusion matrix plot.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : List[str], optional
        Names of classes
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # If class names not provided, use default names
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=500
    )
    
    return fig

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    """
    Create an interactive ROC curve plot.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Target scores (probabilities)
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (area = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='red', width=2)
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500,
        showlegend=True
    )
    
    return fig

def plot_feature_importance(importances: np.ndarray, feature_names: List[str]) -> go.Figure:
    """
    Create an interactive feature importance plot.
    
    Parameters
    ----------
    importances : np.ndarray
        Feature importance values
    feature_names : List[str]
        Names of features
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    # Calculate percentages
    percentages = (sorted_importances / np.sum(sorted_importances)) * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=sorted_names,
        y=percentages,
        marker_color='skyblue',
        text=[f'{p:.1f}%' for p in percentages],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance (%)',
        width=800,
        height=500
    )
    
    return fig

def plot_prediction_error(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create an interactive prediction error plot for regression models.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add scatter plot of predicted vs true values
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(color='blue', opacity=0.6),
        name='Predictions'
    ))
    
    # Add ideal prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Ideal',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title='Prediction Error',
        xaxis_title='True Values',
        yaxis_title='Predicted Values',
        width=700,
        height=500
    )
    
    return fig

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create an interactive residuals plot for regression models.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    # Add scatter plot of residuals vs predicted values
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='orange', opacity=0.6),
        name='Residuals'
    ))
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='red',
        name='Zero Line'
    )
    
    fig.update_layout(
        title='Residuals Plot',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals',
        width=700,
        height=500
    )
    
    return fig
