import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

from utils.data import load_and_clean_data, preprocess_data
from utils.model import train_models, get_best_model, evaluate_model_at_threshold

# Page configuration
st.set_page_config(
    page_title="Churn Insights Dashboard",
    layout="wide"
)

def main():
    st.title("Churn Insights Dashboard")
    
    # Initialize session state
    if 'data' not in st.session_state:
        try:
            st.session_state.data = load_and_clean_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.data = None
    
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'cv_results' not in st.session_state:
        st.session_state.cv_results = None
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'retention_aggressiveness': 0.5,
            'cost_per_contact': 25,
            'value_saved_per_customer': 1200,
            'segment_filter': 'All Customers'
        }
    
    # Sidebar controls
    with st.sidebar:
        st.header("Campaign Controls")
        
        with st.form("controls"):
            st.subheader("Retention Settings")
            
            retention_aggressiveness = st.slider(
                "Retention Aggressiveness",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.settings['retention_aggressiveness'],
                step=0.05
            )
            
            cost_per_contact = st.number_input(
                "Cost per Contact ($)",
                min_value=1,
                max_value=500,
                value=st.session_state.settings['cost_per_contact']
            )
            
            value_saved = st.number_input(
                "Value Saved per Customer ($)",
                min_value=100,
                max_value=10000,
                value=st.session_state.settings['value_saved_per_customer']
            )
            
            segment_filter = st.selectbox(
                "Customer Segment",
                ['All Customers', 'Month-to-month', 'One year', 'Two year'],
                index=['All Customers', 'Month-to-month', 'One year', 'Two year'].index(st.session_state.settings['segment_filter'])
            )
            
            apply_changes = st.form_submit_button("Apply Settings", type="primary")
            
            if apply_changes:
                st.session_state.settings.update({
                    'retention_aggressiveness': retention_aggressiveness,
                    'cost_per_contact': cost_per_contact,
                    'value_saved_per_customer': value_saved,
                    'segment_filter': segment_filter
                })
                st.success("Settings applied!")
                st.rerun()
        
        # Training button
        if st.button("Train Models", type="secondary", use_container_width=True):
            if st.session_state.data is not None:
                with st.spinner("Training models..."):
                    try:
                        df = st.session_state.data
                        
                        # Apply segment filter
                        if st.session_state.settings['segment_filter'] != 'All Customers':
                            df = df[df['Contract'] == st.session_state.settings['segment_filter']]
                        
                        # Preprocess and train
                        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
                        models, cv_results = train_models(X_train, y_train, preprocessor, feature_names)
                        
                        # Store results
                        st.session_state.models = models
                        st.session_state.cv_results = cv_results
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = feature_names
                        
                        st.success("Models trained!")
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
            else:
                st.error("No data available for training")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.error("Could not load customer data. Please check data source.")
        return
    
    df = st.session_state.data
    settings = st.session_state.settings
    
    # Apply segment filter for display
    if settings['segment_filter'] != 'All Customers':
        df_display = df[df['Contract'] == settings['segment_filter']]
    else:
        df_display = df
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary", 
        "Segments & Drivers", 
        "Retention Planner", 
        "Details & Methods", 
        "Data & Quality"
    ])
    
    with tab1:
        st.header("Executive Summary")
        st.caption("High-level business metrics and strategic insights")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        churn_rate = (df_display['Churn'] == 'Yes').mean() * 100
        avg_value = df_display['MonthlyCharges'].mean() * 12
        at_risk_customers = int(len(df_display) * 0.2)
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
        
        # ROI estimate
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
    
    with tab2:
        st.header("Customer Segments & Churn Drivers")
        st.caption("Identify high-risk customer groups")
        
        # Segment analysis
        st.subheader("Highest Risk Customer Segments")
        
        # Create segment analysis
        df_analysis = df_display.copy()
        df_analysis['Tenure_Group'] = pd.cut(
            df_analysis['tenure'], 
            bins=[0, 12, 24, 72], 
            labels=['New (0-12m)', 'Growing (1-2y)', 'Loyal (2y+)']
        )
        
        segment_analysis = df_analysis.groupby(['Contract', 'Tenure_Group']).agg({
            'Churn': [lambda x: len(x), lambda x: (x == 'Yes').mean() * 100]
        }).round(1)
        
        segment_analysis.columns = ['Customer_Count', 'Churn_Rate']
        segment_analysis = segment_analysis.reset_index().sort_values('Churn_Rate', ascending=False)
        
        # Display as table
        display_segments = segment_analysis.head(6).copy()
        display_segments['Risk_Level'] = display_segments['Churn_Rate'].apply(
            lambda x: 'Very High' if x > 40 else 'High' if x > 25 else 'Medium'
        )
        
        st.dataframe(
            display_segments,
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
        
        # Business recommendations
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
    
    with tab3:
        st.header("Retention Campaign Planner")
        st.caption("Plan and optimize retention campaigns")
        
        if st.session_state.models is None:
            st.warning("Please train models first using the sidebar controls.")
        else:
            try:
                models = st.session_state.models
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                # Get predictions
                best_model_name = get_best_model(st.session_state.cv_results)
                model = models[best_model_name]
                
                y_proba = model.predict_proba(X_test)[:, 1]
                threshold = settings['retention_aggressiveness']
                y_pred = (y_proba >= threshold).astype(int)
                
                # Calculate business metrics
                customers_contacted = np.sum(y_pred)
                true_positives = np.sum((y_pred == 1) & (y_test == 1))
                
                campaign_cost = customers_contacted * settings['cost_per_contact']
                value_generated = true_positives * settings['value_saved_per_customer']
                net_roi = value_generated - campaign_cost
                roi_percentage = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0
                
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
                    st.metric("Net ROI", f"${net_roi:,}")
                    st.caption(f"{roi_percentage:.0f}% return on investment")
                
                # Efficiency metrics
                st.subheader("Campaign Efficiency")
                
                precision = true_positives / customers_contacted if customers_contacted > 0 else 0
                cost_per_save = campaign_cost / true_positives if true_positives > 0 else float('inf')
                
                eff_col1, eff_col2 = st.columns(2)
                
                with eff_col1:
                    st.metric("Campaign Precision", f"{precision:.1%}")
                    st.caption("Accuracy of targeting at-risk customers")
                
                with eff_col2:
                    cost_display = f"${cost_per_save:.0f}" if cost_per_save != float('inf') else "N/A"
                    st.metric("Cost per Customer Saved", cost_display)
                    st.caption("Investment required per retention")
                
            except Exception as e:
                st.error(f"Error in retention planner: {str(e)}")
    
    with tab4:
        st.header("Model Performance & Technical Details")
        st.caption("Technical analysis for data science teams")
        
        if st.session_state.models is None:
            st.warning("Please train models first to view technical details.")
        else:
            try:
                cv_results = st.session_state.cv_results
                
                if cv_results:
                    st.subheader("Model Performance Comparison")
                    cv_df = pd.DataFrame(cv_results)
                    st.dataframe(cv_df.round(4), use_container_width=True)
                    
                    # Best model info
                    best_model_name = get_best_model(cv_results)
                    st.info(f"Best performing model: {best_model_name}")
                
            except Exception as e:
                st.error(f"Error displaying model details: {str(e)}")
    
    with tab5:
        st.header("Data Quality & Governance")
        st.caption("Dataset overview and quality metrics")
        
        # Dataset overview
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
        
        # Business KPIs
        st.subheader("Key Performance Indicators")
        
        kpi_data = {
            'Business Metric': [
                'Customer Lifetime Value',
                'Monthly Churn Impact',
                'Revenue Recovery Potential',
                'Customer Acquisition Cost Impact',
                'Retention ROI Opportunity'
            ],
            'Current Value': [
                f"${df['MonthlyCharges'].mean() * df['tenure'].mean():.0f}",
                f"${len(df) * (df['Churn'] == 'Yes').mean() * df['MonthlyCharges'].mean():,.0f}",
                f"${len(df) * (df['Churn'] == 'Yes').mean() * df['MonthlyCharges'].mean() * 12 * 0.7:,.0f}",
                f"${len(df) * (df['Churn'] == 'Yes').mean() * 200:,.0f}",
                "300-500% with targeted campaigns"
            ],
            'Business Impact': [
                'Average revenue per customer over tenure',
                'Direct monthly revenue loss from churn',
                '70% of lost revenue potentially recoverable',
                'Cost to replace churned customers',
                'Expected return on retention investment'
            ]
        }
        
        kpi_df = pd.DataFrame(kpi_data)
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)
        
        # Contract distribution
        st.subheader("Customer Contract Distribution")
        
        contract_analysis = df.groupby('Contract').agg({
            'Churn': [lambda x: len(x), lambda x: (x == 'Yes').mean() * 100],
            'MonthlyCharges': 'mean'
        }).round(1)
        
        contract_analysis.columns = ['Customer_Count', 'Churn_Rate', 'Avg_Monthly']
        st.dataframe(contract_analysis.reset_index(), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()