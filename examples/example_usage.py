"""
Example usage of the model evaluation framework.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pyminions import ModelEvaluator

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

# Create feature names
feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]

# Run evaluation
metrics = evaluator.evaluate(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names,
    evaluation_split="Train/Test: 800/200 (20% test)",
    target_variable="Class label",
    model_source="Pre-trained RandomForestClassifier"
)

print("\nEvaluation Results:")
print("=" * 50)
print(f"Metrics: {metrics}")
print(f"\nResults saved to: {evaluator.output_dir}")

# --- Regression Example ---
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# Create regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train regression model
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(Xr_train, yr_train)

# Create regression evaluator
reg_evaluator = ModelEvaluator(
    model=reg_model,
    task_type='regression',
    experiment_name='random_forest_regression_example'
)

reg_feature_names = [f'Reg_Feature_{i+1}' for i in range(X_reg.shape[1])]

# Run regression evaluation
reg_metrics = reg_evaluator.evaluate(
    X_train=Xr_train,
    X_test=Xr_test,
    y_train=yr_train,
    y_test=yr_test,
    feature_names=reg_feature_names,
    evaluation_split="Train/Test: 800/200 (20% test)",
    target_variable="Target value",
    model_source="Pre-trained RandomForestRegressor"
)

print("\nRegression Evaluation Results:")
print("=" * 50)
print(f"Regression Metrics: {reg_metrics}")
print(f"\nRegression results saved to: {reg_evaluator.output_dir}")
