"""
Dynamic, plain-language interpretation utilities for PyMinions model evaluation outputs and plots.
"""
from typing import Dict, List, Optional
import numpy as np

def interpret_classification_metrics(metrics: Dict) -> str:
    acc = metrics.get('accuracy')
    prec = metrics.get('precision')
    rec = metrics.get('recall')
    f1 = metrics.get('f1')
    roc_auc = metrics.get('roc_auc')
    mcc = metrics.get('mcc')
    kappa = metrics.get('cohen_kappa')
    bal_acc = metrics.get('balanced_accuracy')
    specificity = metrics.get('specificity')
    fpr = metrics.get('fpr')
    fnr = metrics.get('fnr')
    brier = metrics.get('brier')
    logloss = metrics.get('log_loss')
    fbeta2 = metrics.get('fbeta_2')
    hamming = metrics.get('hamming_loss')
    jaccard = metrics.get('jaccard')
    summary = []
    if acc is not None:
        summary.append(f"Accuracy: {acc:.2f} — the proportion of correct predictions out of all cases.")
    if prec is not None:
        summary.append(f"Precision: {prec:.2f} — when the model predicts positive, this is the fraction that are actually positive.")
    if rec is not None:
        summary.append(f"Recall: {rec:.2f} — the fraction of actual positives the model correctly identifies.")
    if f1 is not None:
        summary.append(f"F1 Score: {f1:.2f} — harmonic mean of precision and recall; balances false positives and negatives.")
    if roc_auc is not None:
        summary.append(f"ROC-AUC: {roc_auc:.2f} — measures the model's ability to distinguish between classes (1.0 is perfect, 0.5 is random).")
    if mcc is not None:
        summary.append(f"Matthews Correlation Coefficient (MCC): {mcc:.2f} — a balanced measure even for imbalanced datasets; 1 is perfect, 0 is random, -1 is total disagreement.")
    if kappa is not None:
        summary.append(f"Cohen's Kappa: {kappa:.2f} — agreement between model and true labels, adjusted for chance.")
    if bal_acc is not None:
        summary.append(f"Balanced Accuracy: {bal_acc:.2f} — average of recall for each class, useful for imbalanced datasets.")
    if specificity is not None:
        summary.append(f"Specificity: {specificity:.2f} — true negative rate, the fraction of actual negatives correctly identified.")
    if fpr is not None:
        summary.append(f"False Positive Rate (FPR): {fpr:.2f} — proportion of actual negatives incorrectly classified as positive.")
    if fnr is not None:
        summary.append(f"False Negative Rate (FNR): {fnr:.2f} — proportion of actual positives missed by the model.")
    if brier is not None:
        summary.append(f"Brier Score: {brier:.4f} — measures the accuracy of probabilistic predictions (lower is better).")
    if logloss is not None:
        summary.append(f"Log Loss: {logloss:.4f} — penalizes false classifications with high confidence (lower is better).")
    if fbeta2 is not None:
        summary.append(f"F2 Score: {fbeta2:.2f} — like F1 but gives more weight to recall (missed positives).")
    if hamming is not None:
        summary.append(f"Hamming Loss: {hamming:.4f} — fraction of labels incorrectly predicted (lower is better).")
    if jaccard is not None:
        summary.append(f"Jaccard Score: {jaccard:.2f} — similarity between predicted and true labels (1 is perfect).")
    return '<ul>' + ''.join(f'<li>{s}</li>' for s in summary) + '</ul>' if summary else ''

def interpret_regression_metrics(metrics: Dict) -> str:
    mse = metrics.get('mse')
    rmse = metrics.get('rmse')
    mae = metrics.get('mae')
    r2 = metrics.get('r2')
    explained_var = metrics.get('explained_variance')
    max_err = metrics.get('max_error')
    medae = metrics.get('median_absolute_error')
    msle = metrics.get('msle')
    rmsle = metrics.get('rmsle')
    poisson = metrics.get('mean_poisson_deviance')
    gamma = metrics.get('mean_gamma_deviance')
    mape = metrics.get('mape')
    summary = []
    if r2 is not None:
        summary.append(f"R²: {r2:.2f} — proportion of variance explained by the model (1 is perfect, 0 means no predictive power).")
    if explained_var is not None:
        summary.append(f"Explained Variance: {explained_var:.2f} — how much of the target's variation is captured by the model.")
    if mse is not None:
        summary.append(f"MSE: {mse:.2f} — mean squared error; average squared difference between predicted and actual values.")
    if rmse is not None:
        summary.append(f"RMSE: {rmse:.2f} — root mean squared error; average error magnitude in original units.")
    if mae is not None:
        summary.append(f"MAE: {mae:.2f} — mean absolute error; average absolute difference between predictions and actual values.")
    if max_err is not None:
        summary.append(f"Max Error: {max_err:.2f} — largest single prediction error observed.")
    if medae is not None:
        summary.append(f"Median Absolute Error: {medae:.2f} — median of all absolute errors.")
    if msle is not None:
        summary.append(f"MSLE: {msle:.4f} — mean squared log error; penalizes underestimates more when values are large.")
    if rmsle is not None:
        summary.append(f"RMSLE: {rmsle:.4f} — root mean squared log error; interpretable on the log scale.")
    if poisson is not None:
        summary.append(f"Mean Poisson Deviance: {poisson:.4f} — goodness-of-fit for count data (lower is better).")
    if gamma is not None:
        summary.append(f"Mean Gamma Deviance: {gamma:.4f} — goodness-of-fit for positive continuous data (lower is better).")
    if mape is not None:
        summary.append(f"MAPE: {mape:.2%} — mean absolute percentage error; average percent error in predictions.")
    return '<ul>' + ''.join(f'<li>{s}</li>' for s in summary) + '</ul>' if summary else ''

def interpret_feature_importance(importances: Optional[List[float]], feature_names: Optional[List[str]]) -> str:
    if importances is None or feature_names is None:
        return "Feature importance is not available."
    sorted_idx = np.argsort(importances)[::-1]
    top_feats = [(feature_names[i], importances[i]) for i in sorted_idx[:5]]
    abs_importances = [abs(imp) for _, imp in top_feats]
    total = sum(abs_importances)
    summary = [f"The most influential features for the model are:"]
    for (name, imp), abs_imp in zip(top_feats, abs_importances):
        pct = (abs_imp / total * 100) if total else 0
        summary.append(f"'{name}' (importance: {pct:.1f}% of total)")
    return '<ul>' + ''.join(f'<li>{s}</li>' for s in summary) + '</ul>'

import json
import os

def interpret_shap_summary(top_feats_json: str = None) -> str:
    if top_feats_json and os.path.exists(top_feats_json):
        with open(top_feats_json, 'r') as f:
            top_feats = json.load(f)
        summary = [
            "The SHAP summary plot shows which features most influence the model's predictions overall.",
            "Top contributing features (percent of total impact):"
        ]
        # top_feats is a list of (name, mean_abs)
        total = sum(abs(mean_abs) for _, mean_abs in top_feats)
        for name, mean_abs in top_feats:
            pct = (abs(mean_abs) / total * 100) if total else 0
            summary.append(f"'{name}' ({pct:.1f}% of total impact)")
        return '<ul>' + ''.join(f'<li>{s}</li>' for s in summary) + '</ul>'
    return '<ul><li>The SHAP summary plot shows which features most influence the model\'s predictions overall. Features at the top have the biggest impact. Red points mean higher feature values; blue means lower. The plot helps you see which features push predictions higher or lower.</li></ul>'

def interpret_shap_local(local_feats_json: str = None) -> str:
    if local_feats_json and os.path.exists(local_feats_json):
        with open(local_feats_json, 'r') as f:
            top_feats = json.load(f)
        summary = [
            "For this prediction, the most influential features were:"
        ]
        abs_values = [abs(value) for _, value, _ in top_feats]
        total = sum(abs_values)
        for (name, value, actual), abs_v in zip(top_feats, abs_values):
            direction = "increased" if value > 0 else "decreased"
            pct = (abs_v / total * 100) if total else 0
            summary.append(f"'{name}' (value: {actual:.2f}) {direction} the prediction by {pct:.1f}% of total feature impact")
        return '<ul>' + ''.join(f'<li>{s}</li>' for s in summary) + '</ul>'
    return '<ul><li>The SHAP local plot explains why the model made a specific prediction for an individual sample. Features in red pushed the prediction higher, while blue pushed it lower. The longer the bar, the bigger the effect.</li></ul>'

def interpret_lime_local(local_feats_json: str = None) -> str:
    if local_feats_json and os.path.exists(local_feats_json):
        with open(local_feats_json, 'r') as f:
            top_feats = json.load(f)
        summary = ["For this prediction, LIME found:"]
        abs_values = [abs(value) for _, value in top_feats]
        total = sum(abs_values)
        for (name, value), abs_v in zip(top_feats, abs_values):
            direction = "increased" if value > 0 else "decreased"
            pct = (abs_v / total * 100) if total else 0
            summary.append(f"'{name}' {direction} the prediction by {pct:.1f}% of total feature impact")
        return '<ul>' + ''.join(f'<li>{s}</li>' for s in summary) + '</ul>'
    return '<ul><li>The LIME plot shows which features were most important for this specific prediction. Positive values push the prediction up, negative values push it down. This helps explain the model\'s reasoning in plain terms.</li></ul>'

def interpret_pdp_ice(pdp_ranges_json: str = None) -> str:
    if pdp_ranges_json and os.path.exists(pdp_ranges_json):
        with open(pdp_ranges_json, 'r') as f:
            pdp_ranges = json.load(f)
        summary = ["Partial Dependence and ICE plots show how changing a feature's value affects the model's prediction."]
        abs_ranges = [abs(stats['range']) for stats in pdp_ranges.values()]
        total = sum(abs_ranges)
        for (feat, stats), abs_rng in zip(pdp_ranges.items(), abs_ranges):
            rng = stats['range']
            if abs_rng < 1e-3:
                summary.append(f"Changing '{feat}' has very little effect on predictions.")
            else:
                direction = "increases" if stats['max'] > stats['min'] else "decreases"
                pct = (abs_rng / total * 100) if total else 0
                summary.append(f"As '{feat}' increases, the prediction {direction} by about {pct:.1f}% of total feature impact.")
        return '<ul>' + ''.join(f'<li>{s}</li>' for s in summary) + '</ul>'
    return '<ul><li>Partial Dependence and ICE plots show how changing a feature\'s value affects the model\'s prediction. If the line is flat, the feature has little effect. If it goes up or down, the feature strongly influences the prediction.</li></ul>'
