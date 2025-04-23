# Define which metrics are basic vs advanced for classification and regression
def get_metric_groups(task_type: str):
    if task_type == 'classification':
        basic = [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
        ]
        advanced = [
            'mcc', 'cohen_kappa', 'balanced_accuracy', 'specificity', 'fpr', 'fnr',
            'brier', 'log_loss', 'fbeta_2', 'hamming_loss', 'jaccard'
        ]
    else:
        basic = [
            'mse', 'rmse', 'mae', 'r2'
        ]
        advanced = [
            'explained_variance', 'max_error', 'median_absolute_error', 'msle', 'rmsle',
            'mean_poisson_deviance', 'mean_gamma_deviance', 'mape'
        ]
    return basic, advanced
