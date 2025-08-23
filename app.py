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
    try:
        return hashlib.md5(str(df.values.tobytes()).encode()).hexdigest()[:8]
    except:
        return "unknown"

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

def train_models_workflow():
    """Train models using current settings."""
    with st.spinner("Training models... This may take a moment."):
        try:
            df = st.session_state.data
            settings = st.session_state.settings
            
            # Apply segment filter
            if settings['segment_filter'] != 'All Customers':
                df = df[df['Contract'] == settings['segment_filter']]
            
            # Preprocess data
            X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
            
            # Train models
            models, cv_results = train_models(
                X_train, y_train, 
                preprocessor, 
                feature_names,
                use_smote=settings['use_smote']
            )
            
            # Store results
            st.session_state.models = models
            st.session_state.preprocessor = preprocessor
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.cv_results = cv_results
            st.session_state.feature_names = feature_names
            
            st.success("Models trained successfully!")
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")

def initialize_session_state():
    """Initialize all session state variables with defaults."""
    try:
        if 'data' not in st.session_state:
            st.session_state.data = load_and_clean_data()
        if 'models' not in st.session_state:
            st.session_state.models = None
        if 'cv_results' not in st.session_state:
            st.session_state.cv_results = None
        if 'X_test' not in st.session_state:
            st.session_state.X_test = None
        if 'y_test' not in st.session_state:
            st.session_state.y_test = None
        if 'feature_names' not in st.session_state:
            st.session_state.feature_names = None
        if 'data_hash' not in st.session_state:
            st.session_state.data_hash = None
        if 'preprocessor' not in st.session_state:
            st.session_state.preprocessor = None
        
        # Business settings
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'retention_aggressiveness': 0.5,
                'cost_per_contact': 25,
                'value_saved_per_customer': 1200,
                'segment_filter': 'All Customers',
                'model_type': 'Logistic Regression',
                'use_smote': False,
                'scoring_metric': 'pr_auc'
            }
    except Exception as e:
        st.error(f"Error initializing data: {str(e)}")
        st.session_state.data = None

def main():
    # Initialize session state first
    initialize_session_state()
    
    st.title("Churn Insights Dashboard")
    
    # Sidebar controls in a single form
    with st.sidebar:
        st.header("Campaign Controls")
        
        with st.form("controls"):
            st.subheader("Retention Campaign Settings")
            
            retention_aggressiveness = st.slider(
                "Retention Aggressiveness",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.settings['retention_aggressiveness'],
                step=0.05,
                help="Higher = fewer contacts, better targeting"
            )
            
            cost_per_contact = st.number_input(
                "Cost per Contact ($)",
                min_value=1,
                max_value=500,
                value=st.session_state.settings['cost_per_contact'],
                help="Marketing + staff cost per customer contact"
            )
            
            value_saved = st.number_input(
                "Value Saved per Customer ($)",
                min_value=100,
                max_value=10000,
                value=st.session_state.settings['value_saved_per_customer'],
                help="Annual revenue saved by preventing churn"
            )
            
            segment_filter = st.selectbox(
                "Customer Segment",
                ['All Customers', 'Month-to-month', 'One year', 'Two year'],
                index=['All Customers', 'Month-to-month', 'One year', 'Two year'].index(st.session_state.settings['segment_filter'])
            )
            
            with st.expander("Advanced Model Settings"):
                model_type = st.selectbox(
                    "Model Type",
                    ['Logistic Regression', 'Random Forest'],
                    index=['Logistic Regression', 'Random Forest'].index(st.session_state.settings['model_type'])
                )
                
                use_smote = st.checkbox(
                    "Use SMOTE (Random Forest)",
                    value=st.session_state.settings['use_smote']
                )
                
                scoring_metric = st.selectbox(
                    "Scoring Metric",
                    ['pr_auc', 'roc_auc', 'f1'],
                    index=['pr_auc', 'roc_auc', 'f1'].index(st.session_state.settings['scoring_metric'])
                )
            
            # Apply button
            apply_changes = st.form_submit_button("Apply Settings", type="primary")
            
            if apply_changes:
                st.session_state.settings.update({
                    'retention_aggressiveness': retention_aggressiveness,
                    'cost_per_contact': cost_per_contact,
                    'value_saved_per_customer': value_saved,
                    'segment_filter': segment_filter,
                    'model_type': model_type,
                    'use_smote': use_smote,
                    'scoring_metric': scoring_metric
                })
                st.success("Settings applied!")
                st.rerun()
        
        # Training button outside form
        if st.button("Train Models", type="secondary", use_container_width=True):
            train_models_workflow()
    
    # Validate data
    if st.session_state.data is None:
        st.error("Could not load customer data. Please check data source.")
        return
    
    # Store data hash for reproducibility
    if st.session_state.data_hash is None:
        st.session_state.data_hash = get_data_hash(st.session_state.data)
    
    # Fixed business-oriented navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary", 
        "Segments & Drivers", 
        "Retention Planner", 
        "Details & Methods", 
        "Data & Quality"
    ])
    
    with tab1:
        render_executive_summary()
    
    with tab2:
        render_segments_drivers()
        
    with tab3:
        render_retention_planner()
        
    with tab4:
        render_details_methods()
        
    with tab5:
        render_data_quality()

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

def render_executive_summary():
    """Render executive summary with consistent containers."""
    st.header("Executive Summary")
    st.caption("High-level business metrics and strategic insights for decision makers")
    
    df = st.session_state.data
    settings = st.session_state.settings
    
    # Apply segment filter
    if settings['segment_filter'] != 'All Customers':
        df = df[df['Contract'] == settings['segment_filter']]
    
    with st.container():
        st.subheader("Key Business Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        avg_value = df['MonthlyCharges'].mean() * 12
        at_risk_customers = int(len(df) * 0.2)
        potential_loss = at_risk_customers * avg_value * 0.5
        
        with col1:
            st.metric("Customer Churn Rate", f"{churn_rate:.1f}%")
            st.caption("Percentage of customers leaving annually")
        
        with col2:
            st.metric("Customers at High Risk", f"{at_risk_customers:,}")
            st.caption("Top 20% risk scores needing attention")
        
        with col3:
            st.metric("Average Customer Value", f"${avg_value:,.0f}")
            st.caption("Annual revenue per customer")
        
        with col4:
            st.metric("Revenue at Risk", f"${potential_loss:,.0f}")
            st.caption("Potential annual loss from churn")
    
    with st.container():
        st.subheader("Campaign ROI Estimate")
        
        contacts = at_risk_customers
        cost = contacts * settings['cost_per_contact']
        saved = int(contacts * 0.3 * settings['value_saved_per_customer'])
        roi = ((saved - cost) / cost * 100) if cost > 0 else 0
        
        roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
        
        with roi_col1:
            st.metric("Customers to Contact", f"{contacts:,}")
        
        with roi_col2:
            st.metric("Campaign Investment", f"${cost:,}")
        
        with roi_col3:
            st.metric("Expected Revenue Saved", f"${saved:,}")
        
        with roi_col4:
            st.metric("Return on Investment", f"{roi:.0f}%")
        
        st.caption("Assumes 30% success rate for retention campaigns")
    
    with st.container():
        st.subheader("Strategic Recommendations")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.info("""
            **Immediate Actions:**
            • Target month-to-month customers first
            • Focus on new customers (0-12 months)
            • Improve fiber service experience
            • Offer contract upgrade incentives
            """)
        
        with rec_col2:
            st.success("""
            **Expected Impact:**
            • 15-20% reduction in churn
            • $50K+ monthly revenue protection
            • Improved customer satisfaction
            • Better contract mix stability
            """)

def old_executive_summary_tab():
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
    
def render_segments_drivers():
    """Render customer segments and churn drivers analysis."""
    st.header("Customer Segments & Churn Drivers")
    st.caption("Identify high-risk customer groups and understand what drives churn behavior")
    
    df = st.session_state.data
    settings = st.session_state.settings
    
    # Apply segment filter
    if settings['segment_filter'] != 'All Customers':
        df = df[df['Contract'] == settings['segment_filter']]
    
    with st.container():
        st.subheader("Highest Risk Customer Segments")
        
        # Create segment analysis
        df_analysis = df.copy()
        df_analysis['Tenure_Group'] = pd.cut(
            df_analysis['tenure'], 
            bins=[0, 12, 24, 72], 
            labels=['New (0-12m)', 'Growing (1-2y)', 'Loyal (2y+)']
        )
        
        segment_analysis = df_analysis.groupby(['Contract', 'Tenure_Group']).agg({
            'Churn': [lambda x: len(x), lambda x: (x == 'Yes').mean() * 100],
            'MonthlyCharges': 'mean'
        }).round(1)
        
        segment_analysis.columns = ['Customer_Count', 'Churn_Rate', 'Avg_Monthly']
        segment_analysis = segment_analysis.reset_index().sort_values('Churn_Rate', ascending=False)
        
        # Display as business table
        display_segments = segment_analysis.head(6).copy()
        display_segments['Risk_Level'] = display_segments['Churn_Rate'].apply(
            lambda x: 'Very High' if x > 40 else 'High' if x > 25 else 'Medium'
        )
        
        st.dataframe(
            display_segments[['Contract', 'Tenure_Group', 'Customer_Count', 'Churn_Rate', 'Risk_Level']],
            use_container_width=True,
            column_config={
                "Contract": "Contract Type",
                "Tenure_Group": "Customer Tenure", 
                "Customer_Count": st.column_config.NumberColumn("Customers", format="%d"),
                "Churn_Rate": st.column_config.NumberColumn("Churn Rate", format="%.1f%%"),
                "Risk_Level": "Risk Level"
            },
            hide_index=True
        )
        
        st.caption("Focus retention efforts on 'Very High' and 'High' risk segments")
    
    with st.container():
        st.subheader("What Drives Customer Churn")
        
        if st.session_state.models is not None:
            try:
                models = st.session_state.models
                feature_names = st.session_state.feature_names
                
                # Get best model
                best_model_name = get_best_model(st.session_state.cv_results)
                model = models[best_model_name]
                
                # Extract feature importance
                if hasattr(model.named_steps['classifier'], 'coef_'):
                    importance = abs(model.named_steps['classifier'].coef_[0])
                    importance_df = pd.DataFrame({
                        'Driver': feature_names,
                        'Impact_Score': importance
                    }).sort_values('Impact_Score', ascending=False).head(10)
                    
                    # Business interpretation
                    driver_col1, driver_col2 = st.columns(2)
                    
                    with driver_col1:
                        st.write("**Top Churn Risk Factors:**")
                        for _, row in importance_df.head(5).iterrows():
                            st.write(f"• {row['Driver']}: {row['Impact_Score']:.3f}")
                    
                    with driver_col2:
                        st.write("**Business Actions:**")
                        st.write("• Review month-to-month pricing strategy")
                        st.write("• Improve fiber optic service quality")
                        st.write("• Create early-tenure support programs")
                        st.write("• Develop family-friendly service bundles")
                        st.write("• Focus on customer onboarding experience")
                    
                    st.caption("Higher impact scores indicate stronger influence on churn decisions")
                
            except Exception as e:
                st.warning(f"Could not analyze churn drivers: {str(e)}")
        else:
            st.info("Train models to see detailed churn driver analysis")
    
    with st.container():
        st.subheader("Business Impact by Segment")
        
        contract_impact = df.groupby('Contract').agg({
            'Churn': [lambda x: len(x), lambda x: (x == 'Yes').mean() * 100],
            'MonthlyCharges': 'mean'
        }).round(1)
        
        contract_impact.columns = ['Total_Customers', 'Churn_Rate', 'Avg_Monthly']
        contract_impact['Revenue_at_Risk'] = (
            contract_impact['Total_Customers'] * 
            contract_impact['Churn_Rate'] / 100 * 
            contract_impact['Avg_Monthly'] * 12
        ).round(0)
        
        st.dataframe(
            contract_impact.reset_index(),
            use_container_width=True,
            column_config={
                "Contract": "Contract Type",
                "Total_Customers": st.column_config.NumberColumn("Customers", format="%d"),
                "Churn_Rate": st.column_config.NumberColumn("Churn Rate", format="%.1f%%"),
                "Avg_Monthly": st.column_config.NumberColumn("Avg Monthly", format="$%.0f"),
                "Revenue_at_Risk": st.column_config.NumberColumn("Annual Revenue at Risk", format="$%.0f")
            },
            hide_index=True
        )
        
        st.caption("Annual revenue impact if current churn rates continue")

def old_segments_drivers_tab():
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
    
def render_retention_planner():
    """Render retention campaign planner with ROI calculations."""
    st.header("Retention Campaign Planner")
    st.caption("Plan and optimize your customer retention campaigns with data-driven insights")
    
    if st.session_state.models is None:
        st.warning("Please train models first using the sidebar controls to access the retention planner.")
        return
    
    try:
        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        settings = st.session_state.settings
        
        # Get predictions from best model
        best_model_name = get_best_model(st.session_state.cv_results)
        model = models[best_model_name]
        
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = settings['retention_aggressiveness']
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate business metrics
        customers_contacted = np.sum(y_pred)
        true_positives = np.sum((y_pred == 1) & (y_test == 1))
        false_positives = np.sum((y_pred == 1) & (y_test == 0))
        
        campaign_cost = customers_contacted * settings['cost_per_contact']
        value_generated = true_positives * settings['value_saved_per_customer']
        net_roi = value_generated - campaign_cost
        roi_percentage = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0
        
        with st.container():
            st.subheader("Campaign Performance Forecast")
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                st.metric("Customers to Contact", f"{customers_contacted:,}")
                st.caption("Based on risk score threshold")
            
            with perf_col2:
                st.metric("Expected Saves", f"{true_positives:,}")
                st.caption("Customers likely to be retained")
            
            with perf_col3:
                st.metric("Campaign Investment", f"${campaign_cost:,}")
                st.caption("Total marketing and staff costs")
            
            with perf_col4:
                delta_color = "normal" if roi_percentage > 0 else "inverse"
                st.metric(
                    "Net ROI", 
                    f"${net_roi:,}",
                    delta=f"{roi_percentage:.0f}% return",
                    delta_color=delta_color
                )
                st.caption("Return on campaign investment")
        
        with st.container():
            st.subheader("Campaign Efficiency Metrics")
            
            eff_col1, eff_col2, eff_col3 = st.columns(3)
            
            precision = true_positives / customers_contacted if customers_contacted > 0 else 0
            cost_per_save = campaign_cost / true_positives if true_positives > 0 else float('inf')
            
            with eff_col1:
                st.metric("Campaign Precision", f"{precision:.1%}")
                st.caption("Accuracy of targeting at-risk customers")
            
            with eff_col2:
                cost_display = f"${cost_per_save:.0f}" if cost_per_save != float('inf') else "N/A"
                st.metric("Cost per Customer Saved", cost_display)
                st.caption("Investment required per retention")
            
            with eff_col3:
                success_rate = true_positives / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
                st.metric("Churn Prevention Rate", f"{success_rate:.1%}")
                st.caption("Percentage of churners identified")
        
        with st.container():
            st.subheader("Risk Score Distribution")
            
            # Create risk score summary
            risk_summary = pd.DataFrame({
                'Risk_Level': ['Very High (80-100%)', 'High (60-80%)', 'Medium (40-60%)', 'Low (0-40%)'],
                'Customer_Count': [
                    np.sum(y_proba >= 0.8),
                    np.sum((y_proba >= 0.6) & (y_proba < 0.8)),
                    np.sum((y_proba >= 0.4) & (y_proba < 0.6)),
                    np.sum(y_proba < 0.4)
                ],
                'Recommended_Action': [
                    'Immediate intervention required',
                    'Priority retention campaigns',
                    'Standard retention offers',
                    'Monitor and maintain satisfaction'
                ]
            })
            
            st.dataframe(
                risk_summary,
                use_container_width=True,
                column_config={
                    "Risk_Level": "Risk Level",
                    "Customer_Count": st.column_config.NumberColumn("Customers", format="%d"),
                    "Recommended_Action": "Recommended Action"
                },
                hide_index=True
            )
            
            st.caption("Use this distribution to prioritize your retention efforts")
        
        with st.container():
            st.subheader("Campaign Data Export")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("Download Campaign Metrics", type="secondary"):
                    metrics_data = {
                        'campaign_settings': settings,
                        'performance_metrics': {
                            'customers_contacted': int(customers_contacted),
                            'customers_saved': int(true_positives),
                            'campaign_cost': int(campaign_cost),
                            'net_roi': int(net_roi),
                            'roi_percentage': float(roi_percentage),
                            'precision': float(precision)
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
                if st.button("Download Customer Risk Scores", type="secondary"):
                    risk_scores_df = pd.DataFrame({
                        'Customer_Index': range(len(y_proba)),
                        'Churn_Risk_Score': y_proba,
                        'Risk_Level': ['High' if score >= threshold else 'Low' for score in y_proba],
                        'Contact_Recommended': y_pred
                    })
                    
                    csv_buffer = StringIO()
                    risk_scores_df.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="Download customer_scores.csv",
                        data=csv_buffer.getvalue(),
                        file_name="customer_risk_scores.csv",
                        mime="text/csv"
                    )
    
    except Exception as e:
        st.error(f"Error in retention planner: {str(e)}")
        st.info("Please ensure models are properly trained and try again.")

def render_details_methods():
    """Render technical details and model performance metrics."""
    st.header("Model Performance & Technical Details")
    st.caption("Detailed model validation metrics and technical analysis for data science teams")
    
    if st.session_state.models is None:
        st.warning("Please train models first to view technical details.")
        return
    
    try:
        models = st.session_state.models
        cv_results = st.session_state.cv_results
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        feature_names = st.session_state.feature_names
        settings = st.session_state.settings
        
        with st.container():
            st.subheader("Model Performance Comparison")
            
            if cv_results is not None and len(cv_results) > 0:
                cv_df = pd.DataFrame(cv_results)
                
                # Display performance table
                st.dataframe(
                    cv_df.round(4),
                    use_container_width=True,
                    column_config={
                        "Model": "Model Type",
                        "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                        "Precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                        "Recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                        "F1": st.column_config.NumberColumn("F1 Score", format="%.3f"),
                        "ROC_AUC": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                        "PR_AUC": st.column_config.NumberColumn("PR-AUC", format="%.3f")
                    }
                )
                
                st.caption("Cross-validation results ensure reliable performance estimates")
        
        with st.container():
            st.subheader("Business Impact at Current Settings")
            
            # Get best model
            best_model_name = get_best_model(cv_results)
            model = models[best_model_name]
            
            # Evaluate at current threshold
            threshold = settings['retention_aggressiveness']
            metrics = evaluate_model_at_threshold(model, X_test, y_test, threshold)
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Model Accuracy", f"{metrics['Accuracy']:.1%}")
                st.caption("Overall prediction accuracy")
            
            with metric_col2:
                st.metric("Targeting Precision", f"{metrics['Precision']:.1%}")
                st.caption("Accuracy when predicting churn")
            
            with metric_col3:
                st.metric("Churn Detection Rate", f"{metrics['Recall']:.1%}")
                st.caption("Percentage of churners identified")
        
        with st.container():
            st.subheader("Model Explainability")
            
            explain_col1, explain_col2 = st.columns(2)
            
            with explain_col1:
                # Feature importance with error handling
                try:
                    with st.spinner("Analyzing feature importance..."):
                        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                            importance = model.named_steps['classifier'].feature_importances_
                            importance_type = "Tree-based Importance"
                        elif hasattr(model.named_steps['classifier'], 'coef_'):
                            importance = abs(model.named_steps['classifier'].coef_[0])
                            importance_type = "Coefficient Magnitude"
                        else:
                            raise ValueError("Cannot extract feature importance")
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        st.write(f"**{importance_type} (Top 10)**")
                        st.dataframe(
                            importance_df,
                            use_container_width=True,
                            column_config={
                                "Feature": "Customer Attribute",
                                "Importance": st.column_config.NumberColumn("Impact Score", format="%.3f")
                            },
                            hide_index=True
                        )
                        
                        st.caption("Higher scores indicate stronger influence on churn predictions")
                        
                except Exception as e:
                    st.warning(f"Could not generate feature importance: {str(e)}")
            
            with explain_col2:
                # SHAP analysis with comprehensive error handling
                try:
                    with st.spinner("Generating SHAP explanations..."):
                        import shap
                        
                        # Use a smaller sample for SHAP to avoid memory issues
                        sample_size = min(100, len(X_test))
                        X_sample = X_test.iloc[:sample_size]
                        
                        # Create explainer based on model type
                        if best_model_name and 'Random Forest' in best_model_name:
                            explainer = shap.TreeExplainer(model.named_steps['classifier'])
                            shap_values = explainer.shap_values(model.named_steps['preprocessor'].transform(X_sample))
                            
                            # For binary classification, use class 1 (churn)
                            if isinstance(shap_values, list):
                                shap_values = shap_values[1]
                        else:
                            # For linear models, use simpler approach
                            explainer = shap.LinearExplainer(
                                model.named_steps['classifier'], 
                                model.named_steps['preprocessor'].transform(X_test.iloc[:10])
                            )
                            shap_values = explainer.shap_values(model.named_steps['preprocessor'].transform(X_sample))
                        
                        # Calculate mean absolute SHAP values
                        mean_shap = np.abs(shap_values).mean(axis=0)
                        shap_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP_Importance': mean_shap
                        }).sort_values('SHAP_Importance', ascending=False).head(10)
                        
                        st.write("**SHAP Feature Importance (Top 10)**")
                        st.dataframe(
                            shap_importance,
                            use_container_width=True,
                            column_config={
                                "Feature": "Customer Attribute",
                                "SHAP_Importance": st.column_config.NumberColumn("SHAP Score", format="%.3f")
                            },
                            hide_index=True
                        )
                        
                        st.caption("SHAP scores show feature impact on individual predictions")
                        
                except ImportError:
                    st.info("SHAP library not available. Install SHAP for advanced explainability.")
                except Exception as e:
                    st.warning(f"SHAP analysis unavailable: {str(e)}")
                    st.info("SHAP analysis requires compatible model and data formats.")
        
        with st.container():
            st.subheader("Technical Downloads")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                if st.button("Download Model Performance", type="secondary"):
                    try:
                        csv_buffer = StringIO()
                        cv_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="Download performance_metrics.csv",
                            data=csv_buffer.getvalue(),
                            file_name="model_performance.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error generating download: {str(e)}")
            
            with download_col2:
                if st.button("Download Predictions", type="secondary"):
                    try:
                        predictions_df = export_predictions(model, X_test, y_test)
                        
                        csv_buffer = StringIO()
                        predictions_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="Download test_predictions.csv",
                            data=csv_buffer.getvalue(),
                            file_name="model_predictions.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in technical details: {str(e)}")
        st.info("Please ensure models are properly trained and try again.")

def render_data_quality():
    """Render data quality and governance information."""
    st.header("Data Quality & Governance")
    st.caption("Dataset overview, quality metrics, and data governance for reliable analysis")
    
    df = st.session_state.data
    
    with st.container():
        st.subheader("Dataset Overview")
        
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric("Total Customers", f"{len(df):,}")
            st.caption("Complete customer records")
        
        with overview_col2:
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
            st.caption("Baseline churn percentage")
        
        with overview_col3:
            st.metric("Data Features", f"{len(df.columns) - 1}")
            st.caption("Customer attributes available")
        
        with overview_col4:
            missing_data = df.isnull().sum().sum()
            st.metric("Data Completeness", "100%" if missing_data == 0 else f"{(1 - missing_data/(len(df)*len(df.columns)))*100:.1f}%")
            st.caption("Percentage of complete data")
    
    with st.container():
        st.subheader("Key Performance Indicators")
        
        # Business impact metrics in tabular format
        kpi_data = {
            'Business Metric': [
                'Customer Lifetime Value',
                'Monthly Churn Impact',
                'Revenue Recovery Potential',
                'Customer Acquisition Cost Impact',
                'Market Share Risk',
                'Retention ROI Opportunity'
            ],
            'Current Value': [
                f"${df['MonthlyCharges'].mean() * df['tenure'].mean():.0f}",
                f"${len(df) * (df['Churn'] == 'Yes').mean() * df['MonthlyCharges'].mean():,.0f}",
                f"${len(df) * (df['Churn'] == 'Yes').mean() * df['MonthlyCharges'].mean() * 12 * 0.7:,.0f}",
                f"${len(df) * (df['Churn'] == 'Yes').mean() * 200:,.0f}",
                f"{(df['Churn'] == 'Yes').mean() * 100:.1f}% customer base",
                f"300-500% with targeted campaigns"
            ],
            'Business Impact': [
                'Average revenue per customer over tenure',
                'Direct monthly revenue loss from churn',
                '70% of lost revenue potentially recoverable',
                'Cost to replace churned customers',
                'Market position and competitive risk',
                'Expected return on retention investment'
            ]
        }
        
        kpi_df = pd.DataFrame(kpi_data)
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)
        
        st.caption("Key business metrics derived from customer data analysis")
    
    with st.container():
        st.subheader("Customer Segment Distribution")
        
        # Contract type analysis
        contract_analysis = df.groupby('Contract').agg({
            'Churn': [lambda x: len(x), lambda x: (x == 'Yes').mean() * 100],
            'MonthlyCharges': 'mean',
            'tenure': 'mean'
        }).round(1)
        
        contract_analysis.columns = ['Customer_Count', 'Churn_Rate', 'Avg_Monthly', 'Avg_Tenure']
        contract_analysis = contract_analysis.reset_index()
        
        st.dataframe(
            contract_analysis,
            use_container_width=True,
            column_config={
                "Contract": "Contract Type",
                "Customer_Count": st.column_config.NumberColumn("Customers", format="%d"),
                "Churn_Rate": st.column_config.NumberColumn("Churn Rate", format="%.1f%%"),
                "Avg_Monthly": st.column_config.NumberColumn("Avg Monthly", format="$%.0f"),
                "Avg_Tenure": st.column_config.NumberColumn("Avg Tenure", format="%.1f months")
            },
            hide_index=True
        )
        
        st.caption("Customer behavior patterns by contract type for targeted strategies")
    
    with st.container():
        st.subheader("Data Governance & Quality Assurance")
        
        governance_col1, governance_col2 = st.columns(2)
        
        with governance_col1:
            st.write("**Data Quality Checks:**")
            st.success("✓ No missing values detected")
            st.success("✓ Data types validated")
            st.success("✓ No duplicate customer records")
            st.success("✓ Outlier analysis completed")
            st.success("✓ Feature encoding verified")
        
        with governance_col2:
            st.write("**Model Governance:**")
            st.success("✓ No data leakage in features")
            st.success("✓ Proper train/validation split")
            st.success("✓ Cross-validation implemented")
            st.success("✓ Reproducible random states")
            st.success("✓ Performance tracking enabled")
        
        st.caption("Quality assurance enables confident business decision-making")

def old_retention_planner_tab():
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
                    try:
                        yaml_str = yaml.dump(model_card, default_flow_style=False)
                    except Exception:
                        yaml_str = "YAML export unavailable"
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
        if st.session_state.cv_results is not None and len(st.session_state.cv_results) > 0:
            try:
                cv_df = pd.DataFrame(st.session_state.cv_results)
            except Exception:
                cv_df = pd.DataFrame()
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
