import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data import create_tenure_bins, get_churn_rate_by_segment
from utils.model import (
    get_confusion_matrix_data, get_roc_curve_data, 
    get_pr_curve_data, get_feature_importance
)
import warnings
warnings.filterwarnings('ignore')

def plot_churn_distribution(df):
    """
    Plot the distribution of churn.
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    plotly.graph_objects.Figure
    """
    churn_counts = df['Churn'].value_counts()
    
    fig = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        title="Customer Churn Distribution",
        color_discrete_map={'No': '#2E8B57', 'Yes': '#DC143C'}
    )
    
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        height=400
    )
    
    return fig

def plot_numeric_histograms(df, column):
    """
    Plot histogram for numeric column by churn status.
    
    Parameters:
    df: pandas DataFrame
    column: str - column name
    
    Returns:
    plotly.graph_objects.Figure
    """
    fig = px.histogram(
        df, 
        x=column, 
        color='Churn',
        title=f'Distribution of {column} by Churn Status',
        marginal='rug',
        color_discrete_map={'No': '#2E8B57', 'Yes': '#DC143C'},
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Count',
        height=500
    )
    
    return fig

def plot_categorical_vs_churn(df, column):
    """
    Plot categorical feature vs churn rate.
    
    Parameters:
    df: pandas DataFrame
    column: str - column name
    
    Returns:
    plotly.graph_objects.Figure
    """
    # Calculate churn rate by category
    churn_rate = df.groupby(column)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    churn_rate.columns = [column, 'Churn_Rate']
    
    # Count by category
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'Count']
    
    # Merge data
    plot_data = pd.merge(churn_rate, counts, on=column)
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for churn rate
    fig.add_trace(
        go.Bar(
            x=plot_data[column],
            y=plot_data['Churn_Rate'],
            name='Churn Rate %',
            marker_color='#DC143C',
            opacity=0.7
        ),
        secondary_y=False,
    )
    
    # Add line chart for counts
    fig.add_trace(
        go.Scatter(
            x=plot_data[column],
            y=plot_data['Count'],
            mode='lines+markers',
            name='Customer Count',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_xaxes(title_text=column)
    fig.update_yaxes(title_text="Churn Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Customer Count", secondary_y=True)
    
    fig.update_layout(
        title=f'Churn Rate by {column}',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_cohort_analysis(df):
    """
    Plot cohort analysis: Contract type vs Tenure bins.
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    plotly.graph_objects.Figure
    """
    # Create tenure bins
    df_copy = df.copy()
    df_copy['Tenure_Bin'] = create_tenure_bins(df_copy)
    
    # Calculate churn rate by cohort
    cohort_data = df_copy.groupby(['Contract', 'Tenure_Bin'])['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()
    
    # Pivot for heatmap
    pivot_data = cohort_data.pivot(index='Contract', columns='Tenure_Bin', values='Churn')
    
    # Create heatmap
    fig = px.imshow(
        pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='RdYlBu_r',
        aspect='auto',
        title='Churn Rate (%) by Contract Type and Tenure',
        labels={'color': 'Churn Rate (%)'}
    )
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if not pd.isna(pivot_data.iloc[i, j]):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{pivot_data.iloc[i, j]:.1f}%",
                    showarrow=False,
                    font=dict(color="white", size=12)
                )
    
    fig.update_layout(height=400)
    
    return fig

def plot_confusion_matrix(model, X_test, y_test, threshold=0.5):
    """
    Plot confusion matrix.
    
    Parameters:
    model: trained sklearn model
    X_test: test features
    y_test: test target
    threshold: float - decision threshold
    
    Returns:
    plotly.graph_objects.Figure
    """
    cm = get_confusion_matrix_data(model, X_test, y_test, threshold)
    
    labels = ['No Churn', 'Churn']
    
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        aspect='auto',
        title=f'Confusion Matrix (Threshold: {threshold})',
        labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'}
    )
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black", size=14)
            )
    
    fig.update_layout(height=400)
    
    return fig

def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve.
    
    Parameters:
    model: trained sklearn model
    X_test: test features
    y_test: test target
    
    Returns:
    plotly.graph_objects.Figure
    """
    fpr, tpr, auc_score = get_roc_curve_data(model, X_test, y_test)
    
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='#2E8B57', width=2)
        )
    )
    
    # Add diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        showlegend=True
    )
    
    return fig

def plot_pr_curve(model, X_test, y_test):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    model: trained sklearn model
    X_test: test features
    y_test: test target
    
    Returns:
    plotly.graph_objects.Figure
    """
    precision, recall, auc_score = get_pr_curve_data(model, X_test, y_test)
    
    # Calculate baseline (random classifier)
    baseline = y_test.mean()
    
    fig = go.Figure()
    
    # Add PR curve
    fig.add_trace(
        go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {auc_score:.3f})',
            line=dict(color='#2E8B57', width=2)
        )
    )
    
    # Add baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name=f'Random Classifier (AUC = {baseline:.3f})',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=400,
        showlegend=True
    )
    
    return fig

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """
    Plot feature importance.
    
    Parameters:
    model: trained sklearn model
    feature_names: list of feature names
    model_name: str - name of the model
    top_n: int - number of top features to show
    
    Returns:
    plotly.graph_objects.Figure
    """
    importance_df = get_feature_importance(model, feature_names, model_name)
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Features - {model_name}',
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

def plot_shap_summary(model, X_test_sample, feature_names):
    """
    Plot SHAP summary plot.
    
    Parameters:
    model: trained sklearn model
    X_test_sample: sample of test data
    feature_names: list of feature names
    
    Returns:
    matplotlib figure or None
    """
    try:
        import shap
        
        # Get the classifier from the pipeline
        classifier = model.named_steps['classifier']
        
        # Transform the data using the preprocessor
        X_transformed = model.named_steps['preprocessor'].transform(X_test_sample)
        
        # Create explainer based on model type
        if hasattr(classifier, 'predict_proba'):  # Tree-based models
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            
            # For binary classification, use positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
        else:  # Linear models
            explainer = shap.LinearExplainer(classifier, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
        fig = plt.gcf()
        
        return fig
        
    except ImportError:
        # SHAP not available, return None
        return None
    except Exception as e:
        # Any other error, return None and print for debugging
        print(f"Error creating SHAP plot: {e}")
        return None

def plot_threshold_analysis(model, X_test, y_test):
    """
    Plot metrics vs threshold analysis.
    
    Parameters:
    model: trained sklearn model
    X_test: test features
    y_test: test target
    
    Returns:
    plotly.graph_objects.Figure
    """
    from utils.model import calculate_threshold_metrics
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics_df = calculate_threshold_metrics(model, X_test, y_test, thresholds)
    
    fig = go.Figure()
    
    metrics_to_plot = ['Precision', 'Recall', 'F1', 'Accuracy']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for metric, color in zip(metrics_to_plot, colors):
        fig.add_trace(
            go.Scatter(
                x=metrics_df['Threshold'],
                y=metrics_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=color, width=2),
                marker=dict(size=6)
            )
        )
    
    fig.update_layout(
        title='Model Performance vs Decision Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_metrics_comparison_chart(cv_results_df):
    """
    Create a radar chart comparing model metrics.
    
    Parameters:
    cv_results_df: pandas DataFrame with CV results
    
    Returns:
    plotly.graph_objects.Figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC']
    
    fig = go.Figure()
    
    for _, row in cv_results_df.iterrows():
        values = [row[metric] for metric in metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=row['Model'],
            line=dict(width=2),
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Model Performance Comparison",
        showlegend=True,
        height=500
    )
    
    return fig
