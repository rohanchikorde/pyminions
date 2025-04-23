import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    fig = ff.create_annotated_heatmap(z=cm, x=class_names, y=class_names, colorscale='Blues', showscale=True)
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig

def plot_feature_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = [importances[i] for i in indices]
    fig = go.Figure(go.Bar(x=sorted_vals, y=sorted_names, orientation='h'))
    fig.update_layout(title='Feature Importance', xaxis_title='Importance', yaxis_title='Feature')
    return fig

def plot_prediction_error(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', marker=dict(color='blue', opacity=0.6)))
    fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)], mode='lines', name='Ideal', line=dict(dash='dash', color='red')))
    fig.update_layout(title='Prediction Error', xaxis_title='True Values', yaxis_title='Predicted Values')
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(color='orange', opacity=0.6)))
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.update_layout(title='Residuals Plot', xaxis_title='Predicted Values', yaxis_title='Residuals')
    return fig
