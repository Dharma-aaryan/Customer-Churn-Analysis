# Churn Insights Dashboard

## Overview

Churn Insights is a Streamlit-based machine learning dashboard that analyzes customer churn patterns using the Telco Customer Churn dataset. The application provides end-to-end functionality for data preprocessing, model training, evaluation, and visualization through an interactive multi-tab interface. It focuses on identifying customers likely to churn and understanding the key factors driving churn behavior.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses Streamlit as the web framework, providing a multi-tab dashboard interface. The UI is organized around session state management to persist data and model artifacts across user interactions. The main application flow is orchestrated through `app.py`, which handles file uploads, data processing, model training, and visualization rendering.

### Data Processing Pipeline
The system implements a robust data preprocessing pipeline using scikit-learn's ColumnTransformer approach. The pipeline automatically identifies categorical and numerical features, applies OneHotEncoder for categorical variables, and StandardScaler for numerical features. Class imbalance is handled through two strategies: balanced class weights for linear models and SMOTE (Synthetic Minority Oversampling Technique) for tree-based models within cross-validation.

### Machine Learning Architecture
The modeling framework supports multiple algorithms through a unified pipeline structure:
- Logistic Regression with balanced class weights for interpretability
- Random Forest Classifier with optional SMOTE integration for handling class imbalance
- Cross-validation using StratifiedKFold to ensure representative train/test splits
- Comprehensive evaluation metrics including accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC

### Visualization Framework
The visualization layer uses Plotly for interactive charts and Matplotlib/Seaborn for statistical plots. SHAP integration provides model explainability through feature importance and summary plots. The visualization utilities are modularized to support various chart types including distribution plots, confusion matrices, ROC curves, precision-recall curves, and cohort analysis.

### Data Flow Architecture
The application follows a clear separation of concerns with utilities organized into three main modules:
- `utils/data.py`: Data loading, cleaning, and preprocessing functions
- `utils/model.py`: Model creation, training, and evaluation functions  
- `utils/viz.py`: Visualization and plotting functions

## External Dependencies

### Core Framework
- **Streamlit**: Web application framework for the dashboard interface
- **pandas/numpy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing tools

### Visualization Libraries
- **Plotly**: Interactive plotting for dashboard visualizations
- **Matplotlib/Seaborn**: Statistical plotting and chart generation
- **SHAP**: Model explainability and feature importance analysis

### Machine Learning Extensions
- **imbalanced-learn**: SMOTE implementation for handling class imbalance
- **scipy**: Statistical functions and scientific computing

### Data Requirements
- **Telco Customer Churn Dataset**: Default dataset with standard columns (customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, etc.)
- **CSV Upload Support**: Flexible data input through file upload interface