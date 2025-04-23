"""
Visualization functions for model evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.inspection import permutation_importance
import seaborn as sns
from typing import Optional, List


def generate_classification_plots(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    output_dir: str,
    model: Optional[object] = None,
    feature_names: Optional[List[str]] = None
) -> None:
    """
    Generate various plots for classification model evaluation.
    
    Parameters
    ----------
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities
    output_dir : str
        Directory to save plots
    model : object, optional
        The trained model for feature importance
    feature_names : list of str, optional
        Names of features for better visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report.png'))
    plt.close(fig)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close(fig)
    
    # 3. ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close(fig)
    
    # 4. Feature Importance
    if model is not None and feature_names is not None:
        try:
            # Try to get feature importance from model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                # Use permutation importance as fallback
                result = permutation_importance(model, X_test, y_test, n_repeats=10)
                importances = result.importances_mean
            
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(X_test.shape[1]), importances[indices],
                    color="r", align="center")
            plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.xlim([-1, X_test.shape[1]])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not generate feature importance plot - {str(e)}")

def generate_regression_plots(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    model: Optional[object] = None
) -> None:
    """
    Generate various plots for regression model evaluation.
    
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
    model : object, optional
        The trained model for feature importance
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prediction vs Actual
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_error.png'))
    plt.close(fig)
    
    # 2. Residuals Plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_plot.png'))
    plt.close(fig)
