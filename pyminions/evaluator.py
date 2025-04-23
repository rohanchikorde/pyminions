"""
Main model evaluation class that orchestrates the evaluation process.
"""

import os
import json
import datetime
import mlflow
from typing import Optional, Dict, List
from sklearn.base import BaseEstimator
import numpy as np
from pathlib import Path

from .metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics
)
from .metrics.metric_groups import get_metric_groups
from .visualizations import generate_classification_plots, generate_regression_plots
from .visualizations.interactive_plots import (
    generate_classification_interactive_plots,
    generate_regression_interactive_plots
)
from .reporting.html_report import generate_html_report

class ModelEvaluator:
    """
    A comprehensive model evaluation framework that combines multiple libraries
    to provide detailed analysis of machine learning models.
    
    Features:
    - Comprehensive metric calculation for classification and regression tasks
    - Interactive HTML reports with Plotly visualizations
    - Static visualizations as fallbacks
    - MLflow experiment tracking
    - Detailed interpretations of metrics and model behavior
    
    Interactive Visualizations:
    - Classification: confusion matrix, ROC curve, feature importance
    - Regression: prediction error plot, residuals plot, feature importance
    
    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn compatible model
    task_type : str
        Type of ML task - either 'classification' or 'regression'
    experiment_name : str, optional
        Name for MLflow experiment tracking
    output_dir : str, optional
        Directory to save evaluation results
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        task_type: str,
        experiment_name: str = "model_evaluation",
        output_dir: str = "evaluation_results"
    ):
        self.model = model
        self.task_type = task_type.lower()
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, self.timestamp)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_experiment(experiment_name)
        
        if self.task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")

    def evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        evaluation_split: Optional[str] = None,
        target_variable: Optional[str] = None,
        model_source: Optional[str] = None,
        explain_features: Optional[list] = None
    ) -> Dict:
        """
        Perform comprehensive model evaluation with interactive visualizations.
        
        This method evaluates the model performance, generates both static and interactive
        visualizations, and creates an HTML report with detailed metrics and interpretations.
        
        Interactive visualizations include:
        - For classification: confusion matrix, ROC curve, feature importance
        - For regression: prediction error plot, residuals plot, feature importance
        
        Parameters
        ----------
        X_train : array-like
            Training features
        X_test : array-like
            Test features
        y_train : array-like
            Training labels
        y_test : array-like
            Test labels
        feature_names : list of str, optional
            Names of features for better visualization
        evaluation_split : str, optional
            Name of the evaluation split (e.g., 'test', 'validation')
        target_variable : str, optional
            Name of the target variable being predicted
        model_source : str, optional
            Description of the model source or version
        explain_features : list, optional
            List of feature indices to focus on for explanations
        
        Returns
        -------
        dict
            Dictionary containing calculated metrics
        
        Notes
        -----
        The evaluation results, including interactive HTML report and all visualizations,
        are saved to the output directory specified during initialization.
        """
        
        with mlflow.start_run():
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_prob = None
            if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)

            # Calculate metrics
            metrics = (
                calculate_classification_metrics(y_test, y_pred, y_prob)
                if self.task_type == 'classification'
                else calculate_regression_metrics(y_test, y_pred)
            )

            # Log metrics to MLflow
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Generate visualizations
            if self.task_type == 'classification':
                generate_classification_plots(
                    X_test, y_test, y_pred, y_prob,
                    output_dir=self.output_dir,
                    model=self.model,
                    feature_names=feature_names
                )
            else:
                generate_regression_plots(
                    X_test, y_test, y_pred,
                    output_dir=self.output_dir,
                    model=self.model
                )

            # EXPLAINABILITY: SHAP, LIME, PDP/ICE
            from .visualizations.explainability import (
                plot_shap_summary, plot_shap_local, plot_lime_local, plot_pdp_ice, get_top_features
            )
            # Decide features for PDP/ICE
            pdp_features = explain_features
            if pdp_features is None:
                pdp_features = get_top_features(self.model, X_test, feature_names, top_n=5)
            # SHAP summary (global)
            plot_shap_summary(self.model, X_test, self.output_dir, feature_names=feature_names)
            # SHAP local (first test sample)
            plot_shap_local(self.model, X_test, self.output_dir, feature_names=feature_names, sample_indices=[0])
            # LIME local (first test sample)
            plot_lime_local(self.model, X_test, y_test, self.output_dir, feature_names=feature_names, sample_indices=[0], mode=self.task_type)
            # PDP/ICE for selected features
            plot_pdp_ice(self.model, X_test, self.output_dir, feature_names=feature_names, features=pdp_features, kind="both")

            # Log artifacts
            mlflow.log_artifacts(self.output_dir)
            mlflow.sklearn.log_model(self.model, "model")

            # Save metrics to JSON
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            # Prepare extra info for report
            # Dynamically fill only if not provided or empty
            split_val = evaluation_split if evaluation_split not in (None, "") else f"Train/Test: {len(X_train)}/{len(X_test)} ({len(X_test)/(len(X_train)+len(X_test)):.2%} test)"
            target_val = target_variable if target_variable not in (None, "") else None
            model_src_val = model_source if model_source not in (None, "") else type(self.model).__name__
            extra_info = {
                'evaluation_split': split_val,
                'dataset_size': len(X_test),
                'target_variable': target_val,
                'model_source': model_src_val,
                'pdp_features': pdp_features
            }
            # --- INTERPRETATIONS ---
            from .interpretation import (
                interpret_classification_metrics, interpret_regression_metrics,
                interpret_feature_importance, interpret_shap_summary, interpret_shap_local,
                interpret_lime_local, interpret_pdp_ice
            )
            # Feature importance for interpretation (if available)
            importances = getattr(self.model, 'feature_importances_', None)
            # Interpret metrics
            if self.task_type == 'classification':
                interpretation_metrics = interpret_classification_metrics(metrics)
            else:
                interpretation_metrics = interpret_regression_metrics(metrics)
            # Interpret feature importance
            interpretation_feature_importance = interpret_feature_importance(importances, feature_names)
            # Interpret explainability plots with extracted numbers
            shap_summary_json = os.path.join(self.output_dir, 'shap_summary_top.json')
            shap_local_json = os.path.join(self.output_dir, 'shap_local_0_top.json')
            lime_local_json = os.path.join(self.output_dir, 'lime_local_0_top.json')
            pdp_ice_json = os.path.join(self.output_dir, 'pdp_ice_ranges.json')
            interpretation_shap_summary = interpret_shap_summary(shap_summary_json)
            interpretation_shap_local = interpret_shap_local(shap_local_json)
            interpretation_lime_local = interpret_lime_local(lime_local_json)
            interpretation_pdp_ice = interpret_pdp_ice(pdp_ice_json)

            # Generate interactive plots using the modular components
            plotly_html_components = {}
            if self.task_type == 'classification':
                # Get class names from y_test or model
                if hasattr(self.model, 'classes_'):
                    class_names = list(map(str, self.model.classes_))
                else:
                    class_names = list(map(str, np.unique(y_test)))
                
                # Generate classification interactive plots
                plotly_html_components = generate_classification_interactive_plots(
                    y_test=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    feature_names=feature_names,
                    importances=importances,
                    class_names=class_names
                )
            else:
                # Generate regression interactive plots
                plotly_html_components = generate_regression_interactive_plots(
                    y_test=y_test,
                    y_pred=y_pred,
                    feature_names=feature_names,
                    importances=importances
                )
            
            # Extract HTML components for the report
            plotly_confusion_matrix_html = plotly_html_components.get('confusion_matrix', '')
            plotly_roc_html = plotly_html_components.get('roc_curve', '')
            plotly_feature_importance_html = plotly_html_components.get('feature_importance', '')
            plotly_prediction_error_html = plotly_html_components.get('prediction_error', '')
            plotly_residuals_html = plotly_html_components.get('residuals', '')
            
            # Prepare static plot paths for the HTML report
            pdp_features = extra_info.get('pdp_features', [])
            if self.task_type == 'classification':
                # Classification plot paths
                classification_report_path = os.path.join(self.output_dir, 'classification_report.png')
                confusion_matrix_path = os.path.join(self.output_dir, 'confusion_matrix.png')
                roc_curve_path = os.path.join(self.output_dir, 'roc_curve.png')
                precision_recall_path = os.path.join(self.output_dir, 'precision_recall.png')
                feature_importance_path = os.path.join(self.output_dir, 'feature_importance.png')
                # Set regression plot paths to None
                prediction_error_path = None
                residuals_plot_path = None
            else:
                # Regression plot paths
                prediction_error_path = os.path.join(self.output_dir, 'prediction_error.png')
                residuals_plot_path = os.path.join(self.output_dir, 'residuals_plot.png')
                feature_importance_path = os.path.join(self.output_dir, 'feature_importance.png')
                # Set classification plot paths to None
                classification_report_path = None
                confusion_matrix_path = None
                roc_curve_path = None
                precision_recall_path = None
            
            # Common explainability plot paths
            shap_summary_path = os.path.join(self.output_dir, 'shap_summary.png')
            shap_local_path = os.path.join(self.output_dir, 'shap_local_0.png')
            lime_local_path = os.path.join(self.output_dir, 'lime_local_0.png')
            pdp_ice_paths = [os.path.join(self.output_dir, f'pdp_ice_{str(f)}.png') for f in pdp_features]
            # Split metrics
            basic_metrics, advanced_metrics = get_metric_groups(self.task_type)
            basic_metrics_dict = {k: v for k, v in metrics.items() if k in basic_metrics}
            advanced_metrics_dict = {k: v for k, v in metrics.items() if k in advanced_metrics}
            # Split interpretations
            from .interpretation import interpret_classification_metrics, interpret_regression_metrics
            if self.task_type == 'classification':
                interp_basic = interpret_classification_metrics({k: metrics[k] for k in basic_metrics if k in metrics})
                interp_advanced = interpret_classification_metrics({k: metrics[k] for k in advanced_metrics if k in metrics})
            else:
                interp_basic = interpret_regression_metrics({k: metrics[k] for k in basic_metrics if k in metrics})
                interp_advanced = interpret_regression_metrics({k: metrics[k] for k in advanced_metrics if k in metrics})
            # Convert static plot paths to relative paths for HTML embedding
            def make_rel(path):
                if path is None:
                    return None
                return os.path.basename(path)
            def make_rel_list(paths):
                return [os.path.basename(p) for p in paths] if paths else []

            self._generate_html_report(
                basic_metrics=basic_metrics_dict,
                advanced_metrics=advanced_metrics_dict,
                interp_basic=interp_basic,
                interp_advanced=interp_advanced,
                metrics=metrics,  # for backward compatibility
                extra_info=extra_info,
                interpretation_metrics=interpretation_metrics,
                interpretation_feature_importance=interpretation_feature_importance,
                interpretation_shap_summary=interpretation_shap_summary,
                interpretation_shap_local=interpretation_shap_local,
                interpretation_lime_local=interpretation_lime_local,
                interpretation_pdp_ice=interpretation_pdp_ice,
                # Static plot paths (relative)
                classification_report_path=make_rel(classification_report_path),
                confusion_matrix_path=make_rel(confusion_matrix_path),
                roc_curve_path=make_rel(roc_curve_path),
                precision_recall_path=make_rel(precision_recall_path),
                feature_importance_path=make_rel(feature_importance_path),
                prediction_error_path=make_rel(prediction_error_path),
                residuals_plot_path=make_rel(residuals_plot_path),
                shap_summary_path=make_rel(shap_summary_path),
                shap_local_path=make_rel(shap_local_path),
                lime_local_path=make_rel(lime_local_path),
                pdp_ice_paths=make_rel_list(pdp_ice_paths),
                # Interactive plot HTML
                plotly_confusion_matrix_html=plotly_confusion_matrix_html,
                plotly_roc_html=plotly_roc_html,
                plotly_feature_importance_html=plotly_feature_importance_html,
                plotly_prediction_error_html=plotly_prediction_error_html,
                plotly_residuals_html=plotly_residuals_html
            )

        def export_pdf_report(self):
            from pyminions.reporting.pdf_export import export_html_to_pdf
            html_path = os.path.join(self.output_dir, 'evaluation_report.html')
            return export_html_to_pdf(html_path)


            print(f"Evaluation complete! Results saved to: {self.output_dir}")
            print(f"Metrics: {metrics}")
            
            return metrics

    def _generate_html_report(self, basic_metrics: Dict = None, advanced_metrics: Dict = None, interp_basic: str = '', interp_advanced: str = '', metrics: Dict = None, extra_info: Dict = None,
                              interpretation_metrics: str = "",
                              interpretation_feature_importance: str = "",
                              interpretation_shap_summary: str = "",
                              interpretation_shap_local: str = "",
                              interpretation_lime_local: str = "",
                              interpretation_pdp_ice: str = "",
                              classification_report_path: str = "",
                              confusion_matrix_path: str = "",
                              roc_curve_path: str = "",
                              precision_recall_path: str = "",
                              feature_importance_path: str = "",
                              prediction_error_path: str = "",
                              residuals_plot_path: str = "",
                              shap_summary_path: str = "",
                              shap_local_path: str = "",
                              lime_local_path: str = "",
                              pdp_ice_paths: List[str] = None,
                              plotly_confusion_matrix_html: str = "",
                              plotly_roc_html: str = "",
                              plotly_feature_importance_html: str = "",
                              plotly_prediction_error_html: str = "",
                              plotly_residuals_html: str = "") -> str:
        """Generate HTML report for model evaluation results using the modular reporting component."""
        # Use the modular HTML report generator
        return generate_html_report(
            output_dir=self.output_dir,
            model_type=self.task_type,
            basic_metrics=basic_metrics,
            advanced_metrics=advanced_metrics,
            interp_basic=interp_basic,
            interp_advanced=interp_advanced,
            metrics=metrics,
            extra_info=extra_info,
            interpretation_metrics=interpretation_metrics,
            interpretation_feature_importance=interpretation_feature_importance,
            interpretation_shap_summary=interpretation_shap_summary,
            interpretation_shap_local=interpretation_shap_local,
            interpretation_lime_local=interpretation_lime_local,
            interpretation_pdp_ice=interpretation_pdp_ice,
            classification_report_path=classification_report_path,
            confusion_matrix_path=confusion_matrix_path,
            roc_curve_path=roc_curve_path,
            precision_recall_path=precision_recall_path,
            feature_importance_path=feature_importance_path,
            prediction_error_path=prediction_error_path,
            residuals_plot_path=residuals_plot_path,
            shap_summary_path=shap_summary_path,
            shap_local_path=shap_local_path,
            lime_local_path=lime_local_path,
            pdp_ice_paths=pdp_ice_paths,
            plotly_confusion_matrix_html=plotly_confusion_matrix_html,
            plotly_roc_html=plotly_roc_html,
            plotly_feature_importance_html=plotly_feature_importance_html,
            plotly_prediction_error_html=plotly_prediction_error_html,
            plotly_residuals_html=plotly_residuals_html
        )
