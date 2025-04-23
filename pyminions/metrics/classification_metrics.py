"""
Metric calculation for classification tasks.
"""

from typing import Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, hamming_loss, 
    jaccard_score, fbeta_score, log_loss, brier_score_loss
)

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> dict:
    """
    Calculate metrics for classification tasks.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities for each class
        
    Returns
    -------
    dict
        Dictionary containing calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard': jaccard_score(y_true, y_pred, average='weighted'),
        'fbeta_2': fbeta_score(y_true, y_pred, beta=2, average='weighted'),
    }
    # Specificity, FPR, FNR, Brier, Log Loss
    from sklearn.metrics import confusion_matrix
    import numpy as np
    labels = np.unique(y_true)
    if len(labels) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_prob[:, 1])
                metrics['brier'] = brier_score_loss(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                metrics['log_loss'] = log_loss(y_true, y_prob)
                # Brier not defined for multiclass in sklearn
        except ValueError as e:
            print(f"Warning: ROC-AUC or log_loss/brier could not be calculated - {str(e)}")

    if y_prob is not None:
        try:
            # For binary classification, use the positive class probabilities
            if y_prob.shape[1] == 2:
                # For binary classification, use the probabilities of the positive class
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # For multi-class, use OvR approach
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError as e:
            print(f"Warning: ROC-AUC score could not be calculated - {str(e)}")
    
    return metrics
