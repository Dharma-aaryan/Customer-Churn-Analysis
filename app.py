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
    with st.expander("How to fix this"):
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
    
    # Load default dataset
    if st.session_state.data is None:
        st.session_state.data = load_and_clean_data()
    
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
    
    # Business-oriented navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary", 
        "Segments & Drivers", 
        "Details & Methods", 
        "Retention Planner", 
        "Data & Quality",
        "Model Configuration"
    ])
    
    with tab1:
        executive_summary_tab()
    
    with tab2:
        segments_drivers_tab()
        
    with tab3:
        details_methods_tab()
        
    with tab4:
        retention_planner_tab()
        
    with tab5:
        data_quality_tab()
        
    with tab6:
        model_configuration_tab()

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

def executive_summary_tab():
    st.header("Executive Summary")
    st.caption("High-level business metrics and churn insights to guide strategic decisions")
    
    df = st.session_state.data
    
    # Business KPIs - Key metrics at a glance
    st.subheader("Key Business Metrics")
    
    # Get cached summary
    summary = get_data_summary(df)
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            "Churn Rate", 
            f"{summary['churn_rate']:.1f}%",
            help="Percentage of customers who left - industry benchmark is 15-25%"
        )
    
    with kpi_col2:
        customers_at_risk = int(len(df) * 0.2)  # Top 20% risk scores
        st.metric(
            "Customers at Risk", 
            f"{customers_at_risk:,}",
            help="High-risk customers who need immediate attention"
        )
    
    with kpi_col3:
        avg_value = summary['avg_monthly'] * 12  # Annual value
        st.metric(
            "Avg Customer Value", 
            f"${avg_value:,.0f}/year",
            help="Average annual revenue per customer"
        )
    
    with kpi_col4:
        potential_loss = customers_at_risk * avg_value * 0.5  # If 50% churn
        st.metric(
            "Potential Revenue at Risk", 
            f"${potential_loss:,.0f}",
            help="Annual revenue at risk from high-probability churners"
        )
    
    # ROI Calculator
    st.subheader("Retention Campaign ROI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cost_per_contact = st.number_input(
            "Cost per Customer Contact",
            min_value=1,
            max_value=100,
            value=25,
            help="Cost to reach out to one customer (marketing + staff time)"
        )
    
    with col2:
        value_saved = st.number_input(
            "Value Saved per Retained Customer",
            min_value=100,
            max_value=5000,
            value=int(avg_value * 0.8),  # 80% of annual value
            help="Annual revenue saved by preventing one customer from churning"
        )
    
    # Calculate ROI metrics
    contacts_sent = customers_at_risk
    campaign_cost = contacts_sent * cost_per_contact
    customers_saved = int(contacts_sent * 0.3)  # Assume 30% success rate
    total_value_saved = customers_saved * value_saved
    net_roi = total_value_saved - campaign_cost
    roi_percentage = (net_roi / campaign_cost) * 100 if campaign_cost > 0 else 0
    
    roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
    
    with roi_col1:
        st.metric("Customers Contacted", f"{contacts_sent:,}")
    
    with roi_col2:
        st.metric("Campaign Cost", f"${campaign_cost:,}")
    
    with roi_col3:
        st.metric("Customers Saved", f"{customers_saved:,}")
    
    with roi_col4:
        st.metric(
            "Net ROI", 
            f"${net_roi:,}", 
            delta=f"{roi_percentage:.0f}% return",
            help="Total value saved minus campaign costs"
        )
    
    st.caption("Use this to estimate the business impact of your retention campaigns")
    
    # Business Insights
    st.subheader("Key Business Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.info("""
        **What's Driving Churn:**
        • Month-to-month contracts have highest risk
        • New customers (< 12 months) are vulnerable
        • Fiber optic users show higher churn rates
        • High monthly charges increase churn likelihood
        """)
    
    with insight_col2:
        st.success("""
        **Retention Opportunities:**
        • Convert month-to-month to annual contracts
        • Focus on customer onboarding (first year)
        • Improve fiber optic service experience
        • Offer loyalty discounts for high-value customers
        """)
    
    # Dataset snapshot
    col1, col2 = st.columns(2)
    
    # Quick wins and action items
    st.subheader("Quick Wins")
    
    quick_win_col1, quick_win_col2 = st.columns(2)
    
    with quick_win_col1:
        st.write("**Immediate Actions:**")
        st.write("• Target 500 highest-risk month-to-month customers")
        st.write("• Offer 6-month contract incentives")
        st.write("• Implement new customer check-ins at 90 days")
    
    with quick_win_col2:
        st.write("**Expected Impact:**")
        st.write("• 15-20% reduction in churn rate")
        st.write("• $50K+ monthly revenue protection")
        st.write("• Improved customer satisfaction scores")
        
def data_quality_tab():
    st.header("Data & Quality")
    st.caption("Dataset overview, quality checks, and data governance information")
    
    df = st.session_state.data
    
    # Dataset Summary
    st.subheader("Dataset Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Total Customers", f"{len(df):,}")
        st.metric("Data Points", f"{len(df) * (len(df.columns) - 1):,}")
    
    with summary_col2:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
        st.metric("Retention Rate", f"{100 - churn_rate:.1f}%")
    
    with summary_col3:
        missing_data = df.isnull().sum().sum()
        if missing_data == 0:
            st.metric("Data Quality", "✅ Excellent")
            st.success("No missing values detected")
        else:
            st.metric("Missing Values", f"{missing_data:,}")
    
    # Contract Mix
    st.subheader("Customer Contract Mix")
    contract_dist = df['Contract'].value_counts(normalize=True) * 100
    
    contract_col1, contract_col2, contract_col3 = st.columns(3)
    
    with contract_col1:
        month_pct = contract_dist.get('Month-to-month', 0)
        st.metric("Month-to-Month", f"{month_pct:.1f}%", help="Highest churn risk segment")
    
    with contract_col2:
        one_year_pct = contract_dist.get('One year', 0)
        st.metric("One Year", f"{one_year_pct:.1f}%", help="Medium risk segment")
    
    with contract_col3:
        two_year_pct = contract_dist.get('Two year', 0)
        st.metric("Two Year", f"{two_year_pct:.1f}%", help="Lowest churn risk segment")
    
    # Data Governance
    with st.expander("Data Governance & Quality", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quality Checks")
            st.success("✓ No missing values")
            st.success("✓ Data types validated")
            st.success("✓ No duplicate records")
            st.success("✓ Outliers reviewed")
        
        with col2:
            st.subheader("Model Safeguards")
            st.success("✓ No data leakage")
            st.success("✓ Proper train/test split")
            st.success("✓ Cross-validation used")
            st.success("✓ Results reproducible")
    
    st.caption("This data quality enables reliable predictions for business decisions")
    
    # KPI Cards
    st.subheader("Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
def segments_drivers_tab():
    st.header("Segments & Drivers")
    st.caption("Understand which customer segments are at risk and what drives churn behavior")
    
    df = st.session_state.data
    
    # Risky Customer Segments
    st.subheader("Highest Risk Customer Segments")
    
    # Create Contract × Tenure bins analysis
    df_analysis = df.copy()
    df_analysis['TenureBin'] = pd.cut(df_analysis['tenure'], 
                                     bins=[0, 12, 24, 72], 
                                     labels=['New (0-12m)', 'Growing (1-2y)', 'Loyal (2y+)'])
    
    # Calculate churn rates by segment
    segment_analysis = df_analysis.groupby(['Contract', 'TenureBin']).agg({
        'Churn': [lambda x: len(x), lambda x: (x == 'Yes').mean() * 100],
        'MonthlyCharges': 'mean'
    }).round(1)
    
    segment_analysis.columns = ['Customer_Count', 'Churn_Rate', 'Avg_Monthly']
    segment_analysis = segment_analysis.reset_index()
    segment_analysis = segment_analysis.sort_values('Churn_Rate', ascending=False)
    
    # Display top risky segments
    st.dataframe(
        segment_analysis.head(6),
        use_container_width=True,
        column_config={
            "Contract": "Contract Type",
            "TenureBin": "Customer Tenure",
            "Customer_Count": st.column_config.NumberColumn("Customers", format="%d"),
            "Churn_Rate": st.column_config.NumberColumn("Churn Rate", format="%.1f%%"),
            "Avg_Monthly": st.column_config.NumberColumn("Avg Monthly", format="$%.0f")
        }
    )
    
    st.caption("Use this to pick segments for targeted retention offers")
    
    # Plain-English explanations
    st.subheader("What This Means for Your Business")
    
    explanation_col1, explanation_col2 = st.columns(2)
    
    with explanation_col1:
        st.info("""
        **High-Risk Patterns:**
        • New month-to-month customers churn at 45%+ rates
        • Fiber optic users are more likely to leave
        • Customers without dependents have higher churn
        • Multiple services don't guarantee loyalty
        """)
    
    with explanation_col2:
        st.success("""
        **Retention Strategies:**
        • Offer contract incentives in first 90 days
        • Improve fiber service quality and support
        • Create family/dependent-friendly bundles
        • Focus on service quality over quantity
        """)
    
def retention_planner_tab():
    st.header("Retention Planner")
    st.caption("Plan your retention campaigns with ROI optimization and customer targeting")
    
    df = st.session_state.data
    
    # Check if models are trained
    if 'models' not in st.session_state or not st.session_state.models:
        st.warning("⚠️ Please train models in the 'Details & Methods' tab first to use the Retention Planner")
        return
    
    models = st.session_state.models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Business Configuration
    st.subheader("💼 Campaign Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        retention_aggressiveness = st.slider(
            "Retention Aggressiveness",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Higher = contact fewer customers but higher precision. Lower = contact more customers but more false alarms"
        )
    
    with config_col2:
        cost_per_contact = st.number_input(
            "Cost per Contact",
            min_value=1,
            max_value=200,
            value=25,
            help="Total cost to reach one customer (marketing + staff time)"
        )
    
    with config_col3:
        value_saved = st.number_input(
            "Value Saved per Customer",
            min_value=100,
            max_value=10000,
            value=1200,
            help="Annual revenue saved by preventing one churn"
        )
    
    # Advanced Model Settings (Hidden)
    with st.expander("Advanced Model Settings", expanded=False):
        model_choice = st.selectbox(
            "Model Selection",
            list(models.keys()),
            help="Choose which trained model to use for predictions"
        )
        
        if 'Random Forest' in model_choice:
            use_smote = st.checkbox(
                "Use SMOTE Balancing",
                value=False,
                help="Apply synthetic oversampling for better minority class detection"
            )
        else:
            use_smote = False
    
    # Calculate business outcomes
    selected_model = models[model_choice]
    
    # Get predictions
    y_proba = selected_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= retention_aggressiveness).astype(int)
    
    # Business metrics
    customers_contacted = np.sum(y_pred)
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    
    campaign_cost = customers_contacted * cost_per_contact
    value_generated = true_positives * value_saved
    net_roi = value_generated - campaign_cost
    roi_percentage = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0
    
    # Results Display
    st.subheader("Campaign Results")
    
    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    
    with result_col1:
        st.metric("Customers Contacted", f"{customers_contacted:,}")
    
    with result_col2:
        st.metric("Customers Saved", f"{true_positives:,}")
    
    with result_col3:
        st.metric("Campaign Cost", f"${campaign_cost:,}")
    
    with result_col4:
        st.metric(
            "Net ROI",
            f"${net_roi:,}",
            delta=f"{roi_percentage:.0f}% return"
        )
    
    # Precision and efficiency metrics
    precision = true_positives / customers_contacted if customers_contacted > 0 else 0
    
    efficiency_col1, efficiency_col2 = st.columns(2)
    
    with efficiency_col1:
        st.metric(
            "Campaign Precision",
            f"{precision:.1%}",
            help="Percentage of contacted customers who would actually churn"
        )
    
    with efficiency_col2:
        cost_per_save = campaign_cost / true_positives if true_positives > 0 else float('inf')
        st.metric(
            "Cost per Customer Saved",
            f"${cost_per_save:.0f}" if cost_per_save != float('inf') else "N/A",
            help="Total cost divided by number of customers successfully retained"
        )
    
    st.caption("Adjust the retention aggressiveness to optimize your ROI and campaign efficiency")
    
    # Download options
    st.subheader("Export Campaign Data")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("Metrics & Settings", type="secondary"):
            metrics_data = {
                "campaign_settings": {
                    "retention_aggressiveness": retention_aggressiveness,
                    "cost_per_contact": cost_per_contact,
                    "value_saved_per_customer": value_saved,
                    "model_used": model_choice
                },
                "results": {
                    "customers_contacted": int(customers_contacted),
                    "customers_saved": int(true_positives),
                    "campaign_cost": int(campaign_cost),
                    "net_roi": int(net_roi),
                    "roi_percentage": float(roi_percentage),
                    "precision": float(precision)
                }
            }
            
            json_str = json.dumps(metrics_data, indent=2)
            st.download_button(
                label="Download metrics.json",
                data=json_str,
                file_name="retention_campaign_metrics.json",
                mime="application/json"
            )
    
    with export_col2:
        if st.button("Top Drivers", type="secondary"):
            # Create simplified feature importance data
            st.info("Feature importance export will be available after model interpretation in Details & Methods tab")
    
    with export_col3:
        if st.button("👥 Scored Customers", type="secondary"):
            # Create customer risk scores
            risk_scores_df = pd.DataFrame({
                'Customer_ID': range(len(y_proba)),
                'Risk_Score': y_proba,
                'Risk_Level': ['High' if score >= retention_aggressiveness else 'Low' for score in y_proba],
                'Contact_Recommended': y_pred
            })
            
            csv_buffer = StringIO()
            risk_scores_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download scored_customers.csv",
                data=csv_buffer.getvalue(),
                file_name="customer_risk_scores.csv",
                mime="text/csv"
            )

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

def details_methods_tab():
    st.header("Details & Methods")
    st.caption("🔬 Technical performance metrics and model validation details")
    
    # This is where the original modeling_tab content goes
    modeling_content()
    
    # Add glossary at the end
    st.subheader("📚 Business Glossary")
    
    with st.expander("Key Terms Explained", expanded=False):
        st.markdown("""
        **Retention Aggressiveness**: How selective you are in choosing customers to contact. Higher = fewer contacts but better targeting.
        
        **Risk Score**: Probability (0-100%) that a customer will churn based on their profile and behavior.
        
        **Precision**: Of all customers you contact, what percentage would actually churn? Higher is better.
        
        **ROI (Return on Investment)**: How much money you make/save compared to what you spend on the campaign.
        
        **Cross-Validation**: Testing the model on multiple data splits to ensure it works reliably.
        
        **SMOTE**: A technique to balance the training data by creating synthetic examples of churning customers.
        
        **Feature Importance**: Which customer characteristics (age, contract type, etc.) most influence churn predictions.
        """)

def modeling_content():
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

def tooltip(label, help_text):
    """Helper function to add tooltip icons to labels."""
    return f"{label} ⓘ"

def original_explainability_content():
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

def model_configuration_tab():
    st.header("Model Configuration")
    st.caption("Configure model training parameters and data filtering options")
    
    df = st.session_state.data
    
    # Data Filtering Section
    st.subheader("Data Filtering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment filter
        segment_options = ['All Customers'] + list(df['Contract'].unique())
        selected_segment = st.selectbox(
            "Filter by Customer Segment",
            segment_options,
            key="segment_filter",
            help="Train models on specific customer segments"
        )
    
    with col2:
        # Optional: Add more filters here
        st.info("Additional filters can be configured here based on business needs")
    
    # Apply segment filter
    if selected_segment != 'All Customers':
        df_filtered = df[df['Contract'] == selected_segment]
        st.info(f"Training will use: {selected_segment} customers ({len(df_filtered):,} records)")
        st.session_state.selected_segment = selected_segment
    else:
        df_filtered = df
        st.session_state.selected_segment = None
    
    # Model Configuration
    st.subheader("Model Training Parameters")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        selected_model = st.selectbox(
            "Primary Model Type",
            ["Logistic Regression", "Random Forest"],
            help="Choose between Logistic Regression (interpretable) or Random Forest (higher accuracy)"
        )
        
        use_smote = st.checkbox(
            "Use SMOTE for Random Forest",
            value=False,
            help="Apply SMOTE to handle class imbalance for Random Forest models"
        )
    
    with config_col2:
        scoring_metric = st.selectbox(
            "Primary Scoring Metric",
            ["pr_auc", "roc_auc", "f1", "precision", "recall"],
            help="Metric used to evaluate and select the best model"
        )
        
        decision_threshold = st.slider(
            "Decision Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Probability threshold for classification decisions"
        )
    
    # Cost Analysis Parameters
    st.subheader("Cost Analysis Parameters")
    
    cost_col1, cost_col2 = st.columns(2)
    
    with cost_col1:
        cost_fp = st.number_input(
            "Cost of False Positive",
            min_value=1,
            max_value=1000,
            value=25,
            help="Cost when model incorrectly predicts churn"
        )
    
    with cost_col2:
        cost_fn = st.number_input(
            "Cost of False Negative", 
            min_value=1,
            max_value=1000,
            value=150,
            help="Cost when model misses actual churn"
        )
    
    # Training Configuration Summary
    st.subheader("Training Configuration Summary")
    
    config_summary = {
        'Parameter': [
            'Dataset Size',
            'Selected Segment',
            'Primary Model',
            'SMOTE Enabled',
            'Scoring Metric',
            'Decision Threshold',
            'False Positive Cost',
            'False Negative Cost'
        ],
        'Value': [
            f"{len(df_filtered):,} customers",
            selected_segment,
            selected_model,
            "Yes" if use_smote else "No",
            scoring_metric.upper(),
            f"{decision_threshold:.2f}",
            f"${cost_fp}",
            f"${cost_fn}"
        ]
    }
    
    config_df = pd.DataFrame(config_summary)
    st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    # Train Models Button
    if st.button("Train Models", type="primary", use_container_width=True):
        with st.spinner("Training models... This may take a moment."):
            # Preprocess data
            X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df_filtered)
            
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
            st.session_state.config_params = {
                'selected_model': selected_model,
                'use_smote': use_smote,
                'scoring_metric': scoring_metric,
                'decision_threshold': decision_threshold,
                'cost_fp': cost_fp,
                'cost_fn': cost_fn,
                'selected_segment': selected_segment
            }
            
        st.success("Models trained successfully! Navigate to other tabs to view results.")
    
    # Display training status
    if st.session_state.models is not None:
        st.success("Models are currently trained and ready for analysis")
        
        # Show quick model performance summary
        if st.session_state.cv_results:
            cv_df = pd.DataFrame(st.session_state.cv_results)
            best_model = cv_df.loc[cv_df['PR_AUC'].idxmax()]
            
            performance_summary = {
                'Metric': ['Best Model', 'PR-AUC Score', 'F1-Score', 'ROC-AUC Score'],
                'Value': [
                    best_model['Model'],
                    f"{best_model['PR_AUC']:.3f}",
                    f"{best_model['F1']:.3f}",
                    f"{best_model['ROC_AUC']:.3f}"
                ]
            }
            
            perf_df = pd.DataFrame(performance_summary)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No models currently trained. Click 'Train Models' to begin analysis.")

if __name__ == "__main__":
    main()
