"""
Explainability utilities for PyMinions: SHAP, LIME, PDP, ICE plots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

# SHAP
import shap
# LIME
from lime.lime_tabular import LimeTabularExplainer
# PDP/ICE
from sklearn.inspection import PartialDependenceDisplay


def get_top_features(model, X, feature_names, top_n=5):
    # Try model.feature_importances_ or fallback to all features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
    else:
        indices = np.arange(min(top_n, X.shape[1]))
    return [feature_names[i] for i in indices] if feature_names is not None else indices.tolist()


def plot_shap_summary(model, X, output_dir, feature_names=None, top_n=5):
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
    plt.close()
    # Save top features by mean(|SHAP|)
    abs_shap = np.abs(shap_values.values)
    # Handle multiclass: take mean across outputs if needed
    if abs_shap.ndim == 3:
        # shape: (samples, outputs, features) -> average over outputs
        mean_abs_shap = np.mean(abs_shap, axis=(0, 1))
    elif abs_shap.ndim == 2:
        mean_abs_shap = np.mean(abs_shap, axis=0)
    else:
        raise ValueError("Unexpected SHAP values shape: {}".format(abs_shap.shape))
    idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    # Ensure feature_names is a list
    if feature_names is not None:
        feature_names_list = list(feature_names)
    else:
        feature_names_list = [str(i) for i in range(len(mean_abs_shap))]
    top_feats = [(feature_names_list[int(i)], float(mean_abs_shap[int(i)])) for i in idx]
    with open(os.path.join(output_dir, 'shap_summary_top.json'), 'w') as f:
        import json; json.dump(top_feats, f)
    return top_feats


def plot_shap_local(model, X, output_dir, feature_names=None, sample_indices=None, top_n=5):
    import shap
    import json
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    if sample_indices is None:
        sample_indices = [0]
    local_contribs = {}
    for idx in sample_indices:
        plt.figure(figsize=(10, 4))
        # Handle multi-class or multi-output: select the first output if needed
        if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            # shape: (samples, outputs, features)
            values = shap_values.values[idx, 0]
            base_value = shap_values.base_values[idx, 0]
            data = shap_values.data[idx]
            single_expl = shap.Explanation(
                values=values,
                base_values=base_value,
                data=data,
                feature_names=shap_values.feature_names
            )
        else:
            values = shap_values.values[idx]
            base_value = shap_values.base_values[idx] if hasattr(shap_values, 'base_values') else None
            data = shap_values.data[idx]
            single_expl = shap.Explanation(
                values=values,
                base_values=base_value,
                data=data,
                feature_names=shap_values.feature_names
            )
        shap.plots.waterfall(single_expl, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_local_{idx}.png'))
        plt.close()
        # Save top local contributions
        abs_vals = np.abs(values)
        idxs = np.argsort(abs_vals)[::-1][:top_n]
        top_feats = [(feature_names[i] if feature_names else str(i), float(values[i]), float(data[i])) for i in idxs]
        local_contribs[idx] = top_feats
        with open(os.path.join(output_dir, f'shap_local_{idx}_top.json'), 'w') as f:
            json.dump(top_feats, f)
    return local_contribs


def plot_lime_local(model, X, y, output_dir, feature_names=None, sample_indices=None, mode='classification', top_n=5):
    import json
    os.makedirs(output_dir, exist_ok=True)
    if sample_indices is None:
        sample_indices = [0]
    explainer = LimeTabularExplainer(X, feature_names=feature_names, class_names=None, mode=mode)
    lime_contribs = {}
    for idx in sample_indices:
        exp = explainer.explain_instance(X[idx], model.predict_proba if mode=='classification' else model.predict, num_features=10)
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lime_local_{idx}.png'))
        plt.close(fig)
        # Save top LIME contributions
        exp_list = exp.as_list(label=1 if mode=='classification' else None)
        top_feats = exp_list[:top_n]
        lime_contribs[idx] = top_feats
        with open(os.path.join(output_dir, f'lime_local_{idx}_top.json'), 'w') as f:
            json.dump(top_feats, f)
    return lime_contribs


def plot_pdp_ice(model, X, output_dir, feature_names=None, features=None, kind="both"):
    import json
    os.makedirs(output_dir, exist_ok=True)
    # features: list of feature names or indices
    if features is None:
        features = list(range(min(5, X.shape[1])))
    if feature_names is not None:
        feature_names_list = feature_names
    else:
        feature_names_list = [str(i) for i in range(X.shape[1])]
    pdp_ranges = {}
    for feat in features:
        feat_idx = feature_names_list.index(feat) if isinstance(feat, str) else feat
        disp = PartialDependenceDisplay.from_estimator(
            model, X, [feat_idx], kind=kind, feature_names=feature_names_list
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pdp_ice_{feature_names_list[feat_idx]}.png'))
        # Extract PDP/ICE curve range for interpretation robustly
        ydatas = []
        def extract_ydatas(obj):
            if hasattr(obj, 'get_ydata'):
                ydatas.append(obj.get_ydata())
            elif isinstance(obj, (list, tuple, np.ndarray)):
                for o in obj:
                    extract_ydatas(o)
        extract_ydatas(disp.lines_)
        if ydatas:
            all_y = np.concatenate([np.asarray(y) for y in ydatas])
            pd_min, pd_max = float(np.min(all_y)), float(np.max(all_y))
            pdp_ranges[feature_names_list[feat_idx]] = {'min': pd_min, 'max': pd_max, 'range': pd_max - pd_min}
        plt.close()
    with open(os.path.join(output_dir, 'pdp_ice_ranges.json'), 'w') as f:
        json.dump(pdp_ranges, f)
    return pdp_ranges
