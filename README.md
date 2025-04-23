# Model Evaluation Framework

A comprehensive Python package for evaluating machine learning models (classification and regression). This framework integrates multiple popular libraries to provide a complete evaluation suite:

- **Scikit-learn**: Performance metrics calculation
- **Yellowbrick**: Model visualization
- **Plotly**: Interactive visualizations
- **MLflow**: Experiment tracking

## Installation

```bash
pip install model-evaluator
```

## Features

- Comprehensive model evaluation for both classification and regression tasks
- Automated metric calculation and visualization generation
- Interactive HTML reports with Plotly visualizations
  - Classification: interactive confusion matrix, ROC curve, and feature importance
  - Regression: interactive prediction error plot, residuals plot, and feature importance
- Static visualizations as fallbacks when interactive options aren't available
- MLflow integration for experiment tracking
- Support for scikit-learn compatible models
- Easy-to-use interface with clear documentation
- Modular design for extensibility

## Usage

### Classification Example
```python
from pyminions import ModelEvaluator
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature names (optional, for better plots)
feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]

# Create evaluator
evaluator = ModelEvaluator(
    model=model,
    task_type='classification',
    experiment_name='rf_classification_example'
)

# Run evaluation
metrics = evaluator.evaluate(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names
)

print("Evaluation Results:")
print(metrics)
print(f"Results saved to: {evaluator.output_dir}")
```

### Regression Example
```python
from pyminions import ModelEvaluator
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Create regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train)

# Feature names (optional)
feature_names = [f'Reg_Feature_{i+1}' for i in range(X.shape[1])]

# Create regression evaluator
reg_evaluator = ModelEvaluator(
    model=reg_model,
    task_type='regression',
    experiment_name='rf_regression_example'
)

# Run regression evaluation
reg_metrics = reg_evaluator.evaluate(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names
)

print("Regression Results:")
print(reg_metrics)
print(f"Results saved to: {reg_evaluator.output_dir}")
```

---

- After running, check the output directory for an HTML report, plots, and metrics.
- MLflow logs will be created for experiment tracking.
- Feature importance and residuals plots are included for both classification and regression if supported by the model.


## Output

The framework generates:
1. Comprehensive metrics report
2. Visualizations (classification report, confusion matrix, ROC curve, etc.)
3. MLflow logs with metrics and artifacts
4. Saved results in a timestamped directory

## Package Structure

```
model-evaluator/
├── pyminions/
│   ├── __init__.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── classification_metrics.py
│   │   └── regression_metrics.py
│   ├── visualizations/
│   │   ├── __init__.py
│   │   ├── classification_plots.py
│   │   └── regression_plots.py
│   ├── utils/
│   │   └── __init__.py
│   └── evaluator.py
├── examples/
│   └── example_usage.py
├── setup.py
├── requirements.txt
└── README.md
```

## Development

To install the package in development mode:

```bash
pip install -e .
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT License
