"""
Model Evaluation Framework for Machine Learning Models.
Integrates scikit-learn, yellowbrick, and mlflow for comprehensive model evaluation.
"""

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

# ML Libraries
import mlflow
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from yellowbrick.classifier import (
    ClassificationReport, ROCAUC, PrecisionRecallCurve, ConfusionMatrix
)
from yellowbrick.regressor import (
    PredictionError, ResidualsPlot
)

class ModelEvaluator:
    """
    A comprehensive model evaluation framework that combines multiple libraries
    to provide detailed analysis of machine learning models.
    
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

    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except ValueError as e:
                print(f"Warning: ROC-AUC score could not be calculated - {str(e)}")
        
        return metrics

    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def _generate_classification_plots(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ):
        """Generate classification visualizations using Yellowbrick."""
        # Classification Report
        viz = ClassificationReport(self.model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.show(outpath=os.path.join(self.output_dir, 'classification_report.png'))
        plt.close()

        # Confusion Matrix
        cm = ConfusionMatrix(self.model)
        cm.fit(X_test, y_test)
        cm.score(X_test, y_test)
        cm.show(outpath=os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

        if y_prob is not None:
            # ROC Curve
            viz = ROCAUC(self.model)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            viz.show(outpath=os.path.join(self.output_dir, 'roc_auc.png'))
            plt.close()

            # Precision-Recall Curve
            viz = PrecisionRecallCurve(self.model)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            viz.show(outpath=os.path.join(self.output_dir, 'precision_recall.png'))
            plt.close()

    def _generate_regression_plots(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray
    ):
        """Generate regression visualizations using Yellowbrick."""
        # Prediction Error Plot
        viz = PredictionError(self.model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.show(outpath=os.path.join(self.output_dir, 'prediction_error.png'))
        plt.close()

        # Residuals Plot
        viz = ResidualsPlot(self.model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.show(outpath=os.path.join(self.output_dir, 'residuals.png'))
        plt.close()

    def evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Perform comprehensive model evaluation.
        
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
        """
        
        with mlflow.start_run():
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_prob = None
            if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)

            # Calculate metrics
            metrics = (
                self._calculate_classification_metrics(y_test, y_pred, y_prob)
                if self.task_type == 'classification'
                else self._calculate_regression_metrics(y_test, y_pred)
            )

            # Log metrics to MLflow
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Generate visualizations
            if self.task_type == 'classification':
                self._generate_classification_plots(X_test, y_test, y_pred, y_prob)
            else:
                self._generate_regression_plots(X_test, y_test, y_pred)

            # Log artifacts
            mlflow.log_artifacts(self.output_dir)
            mlflow.sklearn.log_model(self.model, "model")

            # Save metrics to JSON
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            print(f"Evaluation complete! Results saved to: {self.output_dir}")
            print(f"Metrics: {metrics}")
            
            return metrics

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                       random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        task_type='classification',
        experiment_name='random_forest_example'
    )

    # Run evaluation
    metrics = evaluator.evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
