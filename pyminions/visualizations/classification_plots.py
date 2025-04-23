"""
Visualization generation for classification tasks.
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.classifier import (
    ClassificationReport, ROCAUC, PrecisionRecallCurve, ConfusionMatrix
)

def generate_classification_plots(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    output_dir: str = ".",
    model: object = None,
    feature_names: Optional[list] = None
):
    """
    Generate visualizations for classification tasks.
    
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
    model : object
        Trained model instance
    feature_names : list, optional
        List of feature names for visualization
    """
    if model is None:
        raise ValueError("Model must be provided for visualization generation")

    # Classification Report
    viz = ClassificationReport(model)
    viz.fit(X_test, y_test)
    viz.score(X_test, y_test)
    viz.show(outpath=os.path.join(output_dir, 'classification_report.png'))
    plt.close()

    # Confusion Matrix
    cm = ConfusionMatrix(model)
    cm.fit(X_test, y_test)
    cm.score(X_test, y_test)
    cm.show(outpath=os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.xticks(range(len(feature_importance)), indices)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()

    if y_prob is not None:
        # ROC Curve
        viz = ROCAUC(model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.show(outpath=os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

        # Precision-Recall Curve
        viz = PrecisionRecallCurve(model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.show(outpath=os.path.join(output_dir, 'precision_recall.png'))
        plt.close()
