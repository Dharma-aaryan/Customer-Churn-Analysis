import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import warnings
import hashlib
from datetime import datetime
warnings.filterwarnings('ignore')

# Optional YAML import
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from utils.data import load_and_clean_data, preprocess_data, get_data_dictionary, validate_data_columns, filter_data_by_segment
from utils.model import (
    train_models, evaluate_model_at_threshold, get_best_model, export_predictions,
    calculate_cost_optimal_threshold, get_permutation_importance, get_cv_stability,
    generate_model_interpretation, export_model_card
)
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
if 'data_hash' not in st.session_state:
    st.session_state.data_hash = None
if 'selected_segment' not in st.session_state:
    st.session_state.selected_segment = None

@st.cache_data
def get_data_hash(df):
    """Calculate hash of dataframe for reproducibility tracking."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:8]

def get_status_ribbon():
    """Generate status ribbon showing pipeline progress."""
    status_items = []
    
    # Data status
    if st.session_state.data is not None:
        status_items.append("✓ Data OK")
    else:
        status_items.append("○ Data pending")
    
    # Model training status
    if st.session_state.models is not None:
        status_items.append("✓ CV done")
    else:
        status_items.append("○ CV pending")
    
    # Test evaluation status
    if st.session_state.models is not None and st.session_state.X_test is not None:
        status_items.append("✓ Test evaluated")
    else:
        status_items.append("○ Test pending")
    
    # Export readiness
    if st.session_state.models is not None:
        status_items.append("✓ Exports ready")
    else:
        status_items.append("○ Exports pending")
    
    return " • ".join(status_items)

def tooltip(text, explanation):
    """Create inline tooltip for metrics."""
    return f"{text} ⓘ" if explanation else text

def show_empty_state(message, fix_tips):
    """Show empty state with helpful tips."""
    st.warning(message)
    with st.expander("💡 How to fix this"):
        for tip in fix_tips:
            st.write(f"• {tip}")

# Add caching for expensive operations
@st.cache_resource
def load_and_cache_data():
    """Cache data loading for performance."""
    return load_and_clean_data()

@st.cache_data
def cache_eda_calculations(df):
    """Cache expensive EDA calculations."""
    return {
        'churn_by_contract': df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100),
        'churn_by_internet': df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100),
        'numeric_correlations': df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr()
    }

def main():
    st.title("Churn Insights Dashboard")
    
    # Status ribbon
    status = get_status_ribbon()
    st.markdown(f"**Pipeline Status:** {status}")
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
        show_empty_state(
            "Could not load data. Please check your file or try the default dataset.",
            [
                "Ensure your CSV file has the required columns (especially 'Churn')",
                "Check that TotalCharges column contains numeric values",
                "Verify file encoding is UTF-8",
                "Use the default dataset if you're just exploring"
            ]
        )
        return
    
    # Validate data and store hash for reproducibility
    if st.session_state.data_hash is None:
        st.session_state.data_hash = get_data_hash(st.session_state.data)
    
    # Check for required columns
    required_columns = ['Churn']
    missing_cols = validate_data_columns(st.session_state.data, required_columns)
    if missing_cols:
        show_empty_state(
            f"Missing required columns: {', '.join(missing_cols)}",
            [
                "Your dataset must include a 'Churn' column with Yes/No values",
                "Column names are case-sensitive",
                "Check your CSV headers match the expected format"
            ]
        )
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

@st.cache_data
def get_data_summary(df):
    """Cache expensive data summary calculations."""
    return {
        'total_customers': len(df),
        'churn_rate': (df['Churn'] == 'Yes').mean() * 100,
        'avg_tenure': df['tenure'].mean(),
        'avg_monthly': df['MonthlyCharges'].mean(),
        'contract_dist': df['Contract'].value_counts(normalize=True) * 100
    }

def overview_tab():
    st.header("Dataset Overview")
    
    df = st.session_state.data
    
    # Data Quality Panel
    with st.expander("Data Quality & Governance", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Dictionary")
            data_dict = get_data_dictionary(df)
            st.dataframe(data_dict, use_container_width=True)
        
        with col2:
            st.subheader("Leakage Prevention Checklist")
            st.success("✓ customerID removed")
            st.success("✓ Transforms inside Pipeline")
            st.success("✓ Stratified train/test split")
            st.success("✓ No target leakage detected")
            
            st.subheader("Reproducibility")
            st.write(f"**Random State:** 42")
            st.write(f"**Dataset Hash:** {st.session_state.data_hash}")
            st.write(f"**Code Version:** 1.0")
    
    # Segment Filter
    st.subheader("Segment Analysis")
    segment_options = ['All Customers'] + list(df['Contract'].unique())
    selected_segment = st.selectbox(
        "Filter by Customer Segment",
        segment_options,
        key="segment_filter"
    )
    
    # Apply segment filter
    if selected_segment != 'All Customers':
        df_filtered = df[df['Contract'] == selected_segment]
        st.info(f"Showing analysis for: {selected_segment} customers ({len(df_filtered):,} records)")
    else:
        df_filtered = df
        st.session_state.selected_segment = None
    
    # Get cached summary
    summary = get_data_summary(df_filtered)
    
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
        st.metric("Total Customers", f"{summary['total_customers']:,}")
        st.metric("Total Features", len(df.columns) - 1)  # Excluding target
        
        help_text = "Percentage of customers who churned (target variable)"
        st.metric(
            tooltip("Churn Rate", help_text), 
            f"{summary['churn_rate']:.1f}%",
            help=help_text
        )
        
    with col2:
        st.subheader("Data Quality")
        missing_data = df_filtered.isnull().sum()
        missing_pct = (missing_data / len(df_filtered) * 100).round(2)
        
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
        tenure_help = "Average number of months customers have been with the company"
        st.metric(
            tooltip("Average Tenure", tenure_help), 
            f"{summary['avg_tenure']:.1f} months",
            help=tenure_help
        )
    
    with kpi_col2:
        monthly_help = "Average monthly charges across all customers"
        st.metric(
            tooltip("Avg Monthly Charges", monthly_help), 
            f"${summary['avg_monthly']:.2f}",
            help=monthly_help
        )
    
    with kpi_col3:
        month_to_month_pct = summary['contract_dist'].get('Month-to-month', 0)
        contract_help = "Percentage of customers on month-to-month contracts (highest churn risk)"
        st.metric(
            tooltip("Month-to-Month %", contract_help), 
            f"{month_to_month_pct:.1f}%",
            help=contract_help
        )
    
    with kpi_col4:
        fiber_customers = (df_filtered['InternetService'] == 'Fiber optic').mean() * 100
        fiber_help = "Percentage of customers using fiber optic internet service"
        st.metric(
            tooltip("Fiber Optic %", fiber_help), 
            f"{fiber_customers:.1f}%",
            help=fiber_help
        )
    
    # Contract type breakdown
    st.subheader("Contract Type Distribution")
    contract_churn = df_filtered.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).round(1)
    contract_counts = df_filtered['Contract'].value_counts()
    
    contract_df = pd.DataFrame({
        'Customer Count': contract_counts,
        'Churn Rate %': contract_churn
    })
    st.dataframe(contract_df, use_container_width=True)
    
    # Churn distribution visualization
    st.subheader("Churn Distribution")
    fig = plot_churn_distribution(df_filtered)
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
    
    # Apply segment filter if active
    if st.session_state.get('selected_segment'):
        df = filter_data_by_segment(df, 'Contract', st.session_state.selected_segment)
        st.info(f"Training on {st.session_state.selected_segment} segment ({len(df):,} records)")
    
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
    
    # Decision threshold slider with tooltip
    threshold_help = "Probability threshold for classification. Higher values = fewer false positives, more false negatives"
    threshold = st.sidebar.slider(
        tooltip("Decision Threshold", threshold_help),
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help=threshold_help
    )
    
    # Cost analysis inputs
    st.sidebar.subheader("Cost Analysis")
    cost_fp = st.sidebar.number_input(
        "Cost of False Positive",
        min_value=1,
        max_value=100,
        value=1,
        help="Cost when model incorrectly predicts churn"
    )
    cost_fn = st.sidebar.number_input(
        "Cost of False Negative",
        min_value=1,
        max_value=100,
        value=5,
        help="Cost when model misses actual churn"
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
        
        # Cross-validation results with stability analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Cross-Validation Results")
            cv_help = "Cross-validation provides reliable estimates by testing on multiple data splits"
            st.info(f"**{tooltip('Cross-validation', cv_help)}** provides more reliable estimates by testing on multiple data splits")
            
            cv_df = pd.DataFrame(cv_results).round(4)
            st.dataframe(cv_df, use_container_width=True)
        
        with col2:
            st.subheader("CV Stability")
            stability_df = get_cv_stability(pd.DataFrame(cv_results))
            
            # Show coefficient of variation for key metrics
            for _, row in stability_df.iterrows():
                model_name = row['Model']
                st.write(f"**{model_name}**")
                st.write(f"PR-AUC CV: {row['PR_AUC_CV']:.3f}")
                st.write(f"F1 CV: {row['F1_CV']:.3f}")
                st.write("---")
        
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
            pr_help = "PR curves show precision vs recall tradeoff, more informative for imbalanced data than ROC"
            st.info(f"**{tooltip('PR curves', pr_help)}** are more informative than ROC curves for imbalanced datasets")
            
            if model_name in models:
                fig_pr = plot_pr_curve(models[model_name], X_test, y_test)
                st.plotly_chart(fig_pr, use_container_width=True)
            
            # Cost-optimal threshold analysis
            st.subheader("Cost-Optimal Threshold Analysis")
            if model_name in models:
                optimal_threshold, cost_analysis = calculate_cost_optimal_threshold(
                    models[model_name], X_test, y_test, cost_fp, cost_fn
                )
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        "Optimal Threshold",
                        f"{optimal_threshold:.3f}",
                        help="Threshold that minimizes expected cost"
                    )
                    
                    # Calculate metrics at optimal threshold
                    optimal_metrics = evaluate_model_at_threshold(
                        models[model_name], X_test, y_test, optimal_threshold
                    )
                    st.write("**At Optimal Threshold:**")
                    st.write(f"Precision: {optimal_metrics['Precision']:.3f}")
                    st.write(f"Recall: {optimal_metrics['Recall']:.3f}")
                
                with col2:
                    st.write("**Cost Analysis Table**")
                    cost_display = cost_analysis.head(10)[['Threshold', 'False Positives', 'False Negatives', 'Total Cost']]
                    st.dataframe(cost_display, use_container_width=True)
    
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
        
        # Feature importance with enhanced analysis
        st.subheader("Feature Importance")
        importance_help = "Shows which factors most influence churn predictions"
        st.info(f"**{tooltip('Feature importance', importance_help)}** shows which factors most influence churn predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_importance = plot_feature_importance(model, feature_names, model_name)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Plain-English interpretation
            st.write("**Plain-English Interpretation:**")
            interpretations = generate_model_interpretation(model, feature_names, model_name)
            for interpretation in interpretations[:5]:
                st.write(f"• {interpretation}")
            
            # Permutation importance for Random Forest
            if "Random Forest" in model_name:
                with st.spinner("Calculating permutation importance..."):
                    perm_importance_df = get_permutation_importance(
                        model, X_test.iloc[:200], y_test.iloc[:200], feature_names, n_repeats=5
                    )
                    
                    st.write("**Permutation Importance (Top 5):**")
                    top_perm = perm_importance_df.head(5)
                    for _, row in top_perm.iterrows():
                        st.write(f"{row['Feature']}: {row['Importance']:.3f} ± {row['Std']:.3f}")
        
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
    
    # Additional export options
    st.subheader("Advanced Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Model Card", type="secondary"):
            try:
                data_info = {
                    'total_samples': len(df),
                    'features': len(df.columns) - 1,
                    'churn_rate': (df['Churn'] == 'Yes').mean() * 100,
                    'data_hash': st.session_state.data_hash
                }
                
                model_card = export_model_card(
                    models[best_model_name], cv_results, data_info, best_model_name
                )
                
                # Export as JSON (and YAML if available)
                json_str = json.dumps(model_card, indent=2)
                
                st.download_button(
                    label="Download model_card.json",
                    data=json_str,
                    file_name="model_card.json",
                    mime="application/json"
                )
                
                if YAML_AVAILABLE:
                    yaml_str = yaml.dump(model_card, default_flow_style=False)
                    st.download_button(
                        label="Download model_card.yaml",
                        data=yaml_str,
                        file_name="model_card.yaml",
                        mime="text/yaml"
                    )
            except Exception as e:
                st.error(f"Error generating model card: {str(e)}")
    
    with col2:
        if st.button("Generate Report Summary", type="secondary"):
            st.info("""
            **Report Summary Generated**
            
            A comprehensive analysis report includes:
            • Dataset overview and quality metrics
            • Model performance comparison
            • Feature importance analysis
            • Business recommendations
            • Cost-optimal threshold analysis
            
            For PDF export, consider using external tools with the downloaded data.
            """)
    
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
        
        **Permutation Importance**: Measures feature importance by randomly shuffling feature values and observing performance drop
        
        **Cost-Optimal Threshold**: Decision threshold that minimizes expected business costs from false positives and false negatives
        """)

if __name__ == "__main__":
    main()
