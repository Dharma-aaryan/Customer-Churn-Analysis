import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

from utils.data import load_and_clean_data, preprocess_data
from utils.model import train_models, evaluate_model_at_threshold, get_best_model, export_predictions
from utils.viz import (
    plot_churn_distribution, plot_numeric_histograms, plot_categorical_vs_churn,
    plot_cohort_analysis, plot_confusion_matrix, plot_roc_curve, 
    plot_pr_curve, plot_feature_importance, plot_shap_summary
)

# Page configuration
st.set_page_config(
    page_title="Churn Insights Dashboard",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

def main():
    st.title("Churn Insights Dashboard")
    st.markdown("---")
    
    # File upload section
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload your own customer churn dataset or use the default Telco dataset"
    )
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = load_and_clean_data(df)
            st.sidebar.success("Custom dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            st.session_state.data = load_and_clean_data()
    else:
        # Load default dataset
        if st.session_state.data is None:
            st.session_state.data = load_and_clean_data()
        st.sidebar.info("Using default Telco Customer Churn dataset")
    
    if st.session_state.data is None:
        st.error("Could not load data. Please check your file or try the default dataset.")
        return
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Explore Data", 
        "Modeling", 
        "Explainability", 
        "What We Learned"
    ])
    
    with tab1:
        overview_tab()
    
    with tab2:
        explore_data_tab()
    
    with tab3:
        modeling_tab()
    
    with tab4:
        explainability_tab()
    
    with tab5:
        insights_tab()

def overview_tab():
    st.header("Dataset Overview")
    
    df = st.session_state.data
    
    # Key questions box
    st.info("""
    **Key Questions We'll Answer:**
    - Who churns? What are the characteristics of customers who leave?
    - Which factors drive churn? What are the most important predictors?
    - How well can we predict churn? What's our model performance?
    """)
    
    # Dataset snapshot
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Snapshot")
        st.metric("Total Customers", f"{len(df):,}")
        st.metric("Total Features", len(df.columns) - 1)  # Excluding target
        
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
    with col2:
        st.subheader("Data Quality")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        
        if missing_data.sum() == 0:
            st.success("No missing values detected")
        else:
            missing_df = pd.DataFrame({
                'Missing Count': missing_data[missing_data > 0],
                'Missing %': missing_pct[missing_pct > 0]
            })
            st.dataframe(missing_df)
    
    # KPI Cards
    st.subheader("Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        avg_tenure = df['tenure'].mean()
        st.metric("Average Tenure", f"{avg_tenure:.1f} months")
    
    with kpi_col2:
        avg_monthly = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    
    with kpi_col3:
        contract_dist = df['Contract'].value_counts(normalize=True) * 100
        month_to_month_pct = contract_dist.get('Month-to-month', 0)
        st.metric("Month-to-Month %", f"{month_to_month_pct:.1f}%")
    
    with kpi_col4:
        fiber_customers = (df['InternetService'] == 'Fiber optic').mean() * 100
        st.metric("Fiber Optic %", f"{fiber_customers:.1f}%")
    
    # Contract type breakdown
    st.subheader("Contract Type Distribution")
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).round(1)
    contract_counts = df['Contract'].value_counts()
    
    contract_df = pd.DataFrame({
        'Customer Count': contract_counts,
        'Churn Rate %': contract_churn
    })
    st.dataframe(contract_df, use_container_width=True)
    
    # Churn distribution visualization
    st.subheader("Churn Distribution")
    fig = plot_churn_distribution(df)
    st.plotly_chart(fig, use_container_width=True)

def explore_data_tab():
    st.header("Exploratory Data Analysis")
    
    df = st.session_state.data
    
    st.info("**What to look for:** Patterns in customer behavior that might indicate higher churn risk")
    
    # Numeric features analysis
    st.subheader("Numeric Features Distribution")
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_numeric = st.selectbox("Select numeric feature", numeric_cols)
    
    fig_numeric = plot_numeric_histograms(df, selected_numeric)
    st.plotly_chart(fig_numeric, use_container_width=True)
    
    # Categorical features analysis
    st.subheader("Categorical Features vs Churn Rate")
    categorical_cols = ['Contract', 'InternetService', 'PaymentMethod', 'SeniorCitizen']
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_categorical = st.selectbox("Select categorical feature", categorical_cols)
    
    fig_categorical = plot_categorical_vs_churn(df, selected_categorical)
    st.plotly_chart(fig_categorical, use_container_width=True)
    
    # Cohort analysis
    st.subheader("Cohort Analysis: Contract Type vs Tenure")
    st.info("**What to look for:** How churn rates vary by contract length and customer tenure")
    
    fig_cohort = plot_cohort_analysis(df)
    st.plotly_chart(fig_cohort, use_container_width=True)
    
    # Key insights from EDA
    st.subheader("Key Observations")
    st.markdown("""
    **Common patterns to watch for:**
    - **Month-to-month contracts** typically show higher churn rates
    - **Fiber optic customers** may have different churn patterns than DSL
    - **Higher monthly charges** often correlate with increased churn risk
    - **Shorter tenure** customers are generally more likely to churn
    """)

def modeling_tab():
    st.header("Machine Learning Models")
    
    df = st.session_state.data
    
    # Model configuration sidebar
    st.sidebar.header("Model Configuration")
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest"],
        help="Choose between Logistic Regression (with class_weight='balanced') or Random Forest"
    )
    
    use_smote = st.sidebar.checkbox(
        "Use SMOTE for Random Forest",
        value=False,
        help="Apply SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance"
    )
    
    scoring_metric = st.sidebar.selectbox(
        "Primary Scoring Metric",
        ["pr_auc", "roc_auc", "f1", "precision", "recall"],
        help="Metric used to select the best model"
    )
    
    # Decision threshold slider
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability threshold for classification (higher = more conservative)"
    )
    
    # Train models button
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... This may take a moment."):
            # Preprocess data
            X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
            
            # Train models
            models, cv_results = train_models(
                X_train, y_train, 
                preprocessor, 
                feature_names,
                use_smote=use_smote
            )
            
            # Store in session state
            st.session_state.models = models
            st.session_state.preprocessor = preprocessor
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.cv_results = cv_results
            st.session_state.feature_names = feature_names
            
        st.success("Models trained successfully!")
    
    # Display results if models are trained
    if st.session_state.models is not None:
        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        cv_results = st.session_state.cv_results
        
        # Cross-validation results
        st.subheader("Cross-Validation Results")
        st.info("**Cross-validation** provides more reliable estimates by testing on multiple data splits")
        
        cv_df = pd.DataFrame(cv_results).round(4)
        st.dataframe(cv_df, use_container_width=True)
        
        # Test set evaluation at selected threshold
        st.subheader(f"Test Set Performance (Threshold: {threshold})")
        
        test_results = {}
        for name, model in models.items():
            if name.lower().replace(' ', '_') == selected_model.lower().replace(' ', '_'):
                metrics = evaluate_model_at_threshold(model, X_test, y_test, threshold)
                test_results[name] = metrics
        
        if test_results:
            test_df = pd.DataFrame(test_results).round(4)
            st.dataframe(test_df, use_container_width=True)
            
            # Performance visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                model_name = selected_model
                if model_name in models:
                    fig_cm = plot_confusion_matrix(models[model_name], X_test, y_test, threshold)
                    st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.subheader("ROC Curve")
                if model_name in models:
                    fig_roc = plot_roc_curve(models[model_name], X_test, y_test)
                    st.plotly_chart(fig_roc, use_container_width=True)
            
            # Precision-Recall curve
            st.subheader("Precision-Recall Curve")
            st.info("**PR curves** are more informative than ROC curves for imbalanced datasets")
            
            if model_name in models:
                fig_pr = plot_pr_curve(models[model_name], X_test, y_test)
                st.plotly_chart(fig_pr, use_container_width=True)
    
    # Methodology explanation
    with st.expander("How We Built These Models"):
        st.markdown("""
        **Data Preprocessing:**
        1. **Removed customerID** - Not predictive
        2. **Converted TotalCharges** to numeric, median imputation for missing values
        3. **Automatic feature detection** - Separated categorical and numerical features
        4. **Train/test split** - 80/20 stratified split (random_state=42)
        5. **Pipeline creation** - OneHotEncoder for categoricals, StandardScaler for numerics
        
        **Class Imbalance Handling:**
        - **Logistic Regression**: Uses class_weight='balanced'
        - **Random Forest**: Option to use SMOTE (Synthetic Minority Oversampling)
        
        **Model Evaluation:**
        - **5-fold Stratified Cross-Validation**
        - **Multiple metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
        - **Threshold analysis**: Interactive threshold adjustment
        """)

def explainability_tab():
    st.header("Model Explainability")
    
    if st.session_state.models is None:
        st.warning("Please train models first in the Modeling tab.")
        return
    
    models = st.session_state.models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    feature_names = st.session_state.feature_names
    
    # Model selection for explainability
    model_name = st.selectbox(
        "Select Model for Explanation",
        list(models.keys())
    )
    
    if model_name in models:
        model = models[model_name]
        
        # Feature importance
        st.subheader("Feature Importance")
        st.info("**Feature importance** shows which factors most influence churn predictions")
        
        fig_importance = plot_feature_importance(model, feature_names, model_name)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # SHAP Analysis (optional if available)
        st.subheader("SHAP Analysis")
        
        try:
            import shap
            
            with st.spinner("Generating SHAP explanations..."):
                fig_shap = plot_shap_summary(model, X_test.iloc[:100], feature_names)  # Use subset for speed
                if fig_shap is not None:
                    st.pyplot(fig_shap)
                    
                    # SHAP insights
                    st.subheader("SHAP Insights")
                    st.markdown("""
                    **Key SHAP Insights:**
                    - **Red dots** = Higher feature values that increase churn risk
                    - **Blue dots** = Lower feature values that decrease churn risk
                    - **Position on x-axis** = Impact magnitude on prediction
                    
                    **Common patterns:**
                    - **Month-to-Month contracts** typically increase churn risk
                    - **Higher MonthlyCharges** often push predictions toward churn
                    - **Longer tenure** generally decreases churn probability
                    """)
                else:
                    st.info("SHAP visualization not available for this model type")
                    
        except ImportError:
            st.info("""
            **SHAP Analysis Unavailable**
            
            SHAP (SHapley Additive exPlanations) provides advanced model explanations but is not available in this environment.
            
            **What SHAP would show:**
            - Individual feature contributions to each prediction
            - Global feature importance across all predictions
            - Interactive visualizations of feature effects
            
            For detailed model explanations, see the Feature Importance section below.
            """)
        except Exception as e:
            st.warning(f"Could not generate SHAP plots: {str(e)}")
        
        # Model-specific explanations
        if model_name and "Logistic" in model_name:
            st.subheader("Logistic Regression Coefficients")
            st.info("**Positive coefficients** increase churn probability, **negative coefficients** decrease it")
            
            try:
                # Get coefficients from the pipeline
                classifier = model.named_steps['classifier']
                coefficients = classifier.coef_[0]
                
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients,
                    'Abs_Coefficient': np.abs(coefficients)
                }).sort_values('Abs_Coefficient', ascending=False)
                
                # Show top 15 features
                st.dataframe(coef_df.head(15), use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not extract coefficients: {str(e)}")

def insights_tab():
    st.header("What We Learned")
    
    if st.session_state.models is None:
        st.warning("Please train models first to see insights.")
        return
    
    df = st.session_state.data
    models = st.session_state.models
    cv_results = st.session_state.cv_results
    
    # Get best model
    best_model_name = get_best_model(cv_results)
    
    st.subheader("Key Findings")
    
    # Business insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **High-Risk Customer Segments:**
        - Month-to-Month contract customers
        - Fiber optic internet users (certain profiles)
        - Customers with higher monthly charges
        - New customers (shorter tenure)
        - Electronic check payment users
        """)
    
    with col2:
        st.success("""
        **Actionable Business Levers:**
        - Target retention offers to high-risk segments
        - Incentivize longer-term contracts
        - Review pricing strategy for high-charge segments
        - Improve onboarding for new customers
        - Encourage automatic payment methods
        """)
    
    # Model performance summary
    st.subheader("Model Performance Summary")
    
    best_cv_score = cv_results[cv_results['Model'] == best_model_name]
    if not best_cv_score.empty:
        best_score = best_cv_score.iloc[0]
        
        st.markdown(f"""
        **Best Model: {best_model_name}**
        - **PR-AUC**: {best_score['PR_AUC']:.3f}
        - **ROC-AUC**: {best_score['ROC_AUC']:.3f}
        - **F1-Score**: {best_score['F1']:.3f}
        - **Precision**: {best_score['Precision']:.3f}
        - **Recall**: {best_score['Recall']:.3f}
        """)
    
    # Download section
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Metrics", type="secondary"):
            metrics_json = json.dumps(cv_results.to_dict('records'), indent=2)
            st.download_button(
                label="Download metrics.json",
                data=metrics_json,
                file_name="metrics.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Download Feature Importance", type="secondary"):
            try:
                model = models[best_model_name]
                feature_names = st.session_state.feature_names
                
                if "Logistic" in best_model_name:
                    coefficients = model.named_steps['classifier'].coef_[0]
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': np.abs(coefficients)
                    }).sort_values('Importance', ascending=False)
                else:
                    # Random Forest feature importance
                    importances = model.named_steps['classifier'].feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                
                csv_buffer = StringIO()
                importance_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download feature_importance.csv",
                    data=csv_buffer.getvalue(),
                    file_name="feature_importance.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error generating feature importance: {str(e)}")
    
    with col3:
        if st.button("Download Predictions", type="secondary"):
            try:
                model = models[best_model_name]
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                predictions_df = export_predictions(model, X_test, y_test)
                
                csv_buffer = StringIO()
                predictions_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download predictions.csv",
                    data=csv_buffer.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
    
    # Glossary
    st.subheader("Glossary")
    
    with st.expander("Technical Terms Explained"):
        st.markdown("""
        **ROC-AUC**: Area Under the ROC Curve - measures the model's ability to distinguish between classes (0-1, higher is better)
        
        **PR-AUC**: Area Under the Precision-Recall Curve - more informative for imbalanced datasets (0-1, higher is better)
        
        **Precision**: Of all customers predicted to churn, what percentage actually churned? (TP/(TP+FP))
        
        **Recall**: Of all customers who actually churned, what percentage did we correctly identify? (TP/(TP+FN))
        
        **F1-Score**: Harmonic mean of precision and recall - balances both metrics
        
        **SMOTE**: Synthetic Minority Oversampling Technique - creates synthetic examples to balance classes
        
        **One-Hot Encoding**: Converts categorical variables into binary columns (0s and 1s)
        
        **Pipeline**: Chains preprocessing and modeling steps to prevent data leakage
        
        **Cross-Validation**: Tests model performance on multiple train/validation splits for reliability
        """)

if __name__ == "__main__":
    main()
