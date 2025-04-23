"""
HTML report generation for model evaluation results.
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
from jinja2 import Environment, FileSystemLoader

def generate_html_report(
    output_dir: str,
    model_type: str,
    basic_metrics: Dict,
    advanced_metrics: Dict,
    interp_basic: str,
    interp_advanced: str,
    metrics: Dict,
    extra_info: Dict,
    interpretation_metrics: Optional[str] = None,
    interpretation_feature_importance: Optional[str] = None,
    interpretation_shap_summary: Optional[str] = None,
    interpretation_shap_local: Optional[str] = None,
    interpretation_lime_local: Optional[str] = None,
    interpretation_pdp_ice: Optional[str] = None,
    classification_report_path: Optional[str] = None,
    confusion_matrix_path: Optional[str] = None,
    roc_curve_path: Optional[str] = None,
    precision_recall_path: Optional[str] = None,
    feature_importance_path: Optional[str] = None,
    prediction_error_path: Optional[str] = None,
    residuals_plot_path: Optional[str] = None,
    shap_summary_path: Optional[str] = None,
    shap_local_path: Optional[str] = None,
    lime_local_path: Optional[str] = None,
    pdp_ice_paths: Optional[List[str]] = None,
    plotly_confusion_matrix_html: Optional[str] = None,
    plotly_roc_html: Optional[str] = None,
    plotly_feature_importance_html: Optional[str] = None,
    plotly_prediction_error_html: Optional[str] = None,
    plotly_residuals_html: Optional[str] = None
) -> str:
    """
    Generate an HTML report for model evaluation results.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the HTML report
    model_type : str
        Type of model ('classification' or 'regression')
    basic_metrics : Dict
        Dictionary of basic metrics
    advanced_metrics : Dict
        Dictionary of advanced metrics
    interp_basic : str
        Interpretation of basic metrics
    interp_advanced : str
        Interpretation of advanced metrics
    metrics : Dict
        Dictionary of all metrics
    extra_info : Dict
        Additional information to include in the report
    interpretation_metrics : str, optional
        Interpretation of metrics
    interpretation_feature_importance : str, optional
        Interpretation of feature importance
    interpretation_shap_summary : str, optional
        Interpretation of SHAP summary
    interpretation_shap_local : str, optional
        Interpretation of SHAP local
    interpretation_lime_local : str, optional
        Interpretation of LIME local
    interpretation_pdp_ice : str, optional
        Interpretation of PDP/ICE
    classification_report_path : str, optional
        Path to classification report image
    confusion_matrix_path : str, optional
        Path to confusion matrix image
    roc_curve_path : str, optional
        Path to ROC curve image
    precision_recall_path : str, optional
        Path to precision-recall curve image
    feature_importance_path : str, optional
        Path to feature importance image
    prediction_error_path : str, optional
        Path to prediction error image
    residuals_plot_path : str, optional
        Path to residuals plot image
    shap_summary_path : str, optional
        Path to SHAP summary image
    shap_local_path : str, optional
        Path to SHAP local image
    lime_local_path : str, optional
        Path to LIME local image
    pdp_ice_paths : List[str], optional
        Paths to PDP/ICE images
    plotly_confusion_matrix_html : str, optional
        HTML for interactive confusion matrix
    plotly_roc_html : str, optional
        HTML for interactive ROC curve
    plotly_feature_importance_html : str, optional
        HTML for interactive feature importance
    plotly_prediction_error_html : str, optional
        HTML for interactive prediction error
    plotly_residuals_html : str, optional
        HTML for interactive residuals plot
    
    Returns
    -------
    str
        Path to the generated HTML report
    """
    # Set up Jinja2 environment
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('evaluation_report.html')
    
    # Prepare context for template
    context = {
        'model_type': model_type,
        'evaluation_date': extra_info.get('evaluation_date', ''),
        'basic_metrics': basic_metrics,
        'advanced_metrics': advanced_metrics,
        'interp_basic': interp_basic,
        'interp_advanced': interp_advanced,
        'interpretation_metrics': interpretation_metrics,
        'interpretation_feature_importance': interpretation_feature_importance,
        'interpretation_shap_summary': interpretation_shap_summary,
        'interpretation_shap_local': interpretation_shap_local,
        'interpretation_lime_local': interpretation_lime_local,
        'interpretation_pdp_ice': interpretation_pdp_ice,
        'classification_report_path': classification_report_path,
        'confusion_matrix_path': confusion_matrix_path,
        'roc_curve_path': roc_curve_path,
        'precision_recall_path': precision_recall_path,
        'feature_importance_path': feature_importance_path,
        'prediction_error_path': prediction_error_path,
        'residuals_plot_path': residuals_plot_path,
        'shap_summary_path': shap_summary_path,
        'shap_local_path': shap_local_path,
        'lime_local_path': lime_local_path,
        'pdp_ice_paths': pdp_ice_paths or [],
        'plotly_confusion_matrix_html': plotly_confusion_matrix_html,
        'plotly_roc_html': plotly_roc_html,
        'plotly_feature_importance_html': plotly_feature_importance_html,
        'plotly_prediction_error_html': plotly_prediction_error_html,
        'plotly_residuals_html': plotly_residuals_html
    }
    
    # Render template
    html_content = template.render(**context)
    
    # Save HTML report
    html_path = os.path.join(output_dir, 'evaluation_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path
