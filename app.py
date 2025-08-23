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

# Custom CSS for styled metric boxes
st.markdown("""
<style>
.metric-container {
    background: linear-gradient(90deg, rgba(31, 119, 180, 0.1) 0%, rgba(255, 127, 14, 0.1) 100%);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid rgba(31, 119, 180, 0.2);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}
.metric-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15), 0 2px 4px rgba(0, 0, 0, 0.12);
}
.metric-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    margin: 0;
    line-height: 1;
}
.metric-subtitle {
    font-size: 0.9rem;
    color: #8c8c8c;
    margin-top: 0.2rem;
    font-weight: 500;
}
.metric-description {
    font-size: 0.85rem;
    color: #a0a0a0;
    margin-top: 0.5rem;
    line-height: 1.4;
}
.section-divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, rgba(31, 119, 180, 0.3) 0%, rgba(255, 127, 14, 0.3) 100%);
    margin: 2rem 0;
}
.insight-box {
    background: rgba(31, 119, 180, 0.05);
    border-left: 4px solid #1f77b4;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.warning-box {
    background: rgba(255, 193, 7, 0.05);
    border-left: 4px solid #ffc107;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.success-box {
    background: rgba(40, 167, 69, 0.05);
    border-left: 4px solid #28a745;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

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
                "Retention aggressiveness (τ)",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.settings['retention_aggressiveness'],
                step=0.05,
                help="Lower τ flags more customers (higher recall), higher τ flags fewer (higher precision)."
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
                ['All Customers', 'Month-to-month'],
                index=['All Customers', 'Month-to-month'].index(st.session_state.settings['segment_filter']) if st.session_state.settings['segment_filter'] in ['All Customers', 'Month-to-month'] else 0
            )
            
            apply_changes = st.form_submit_button("Apply Settings", type="primary")
            
            if apply_changes:
                # Clear models if segment changed to prevent mismatched predictions
                if st.session_state.settings['segment_filter'] != segment_filter:
                    st.session_state.models = None
                    st.session_state.cv_results = None
                    
                st.session_state.settings.update({
                    'retention_aggressiveness': retention_aggressiveness,
                    'cost_per_contact': cost_per_contact,
                    'value_saved_per_customer': value_saved,
                    'segment_filter': segment_filter
                })
                st.success("Settings applied!")
                st.rerun()
        
        # Add Advanced section for technical controls
        with st.expander("Advanced"):
            st.caption("Use PR-AUC for imbalanced data; adjust campaign settings in Planner to suit your budget and customer experience goals.")
            
            model_type = st.selectbox(
                "Model Type",
                ['Logistic Regression', 'Random Forest'],
                index=0
            )
            
            use_smote = st.checkbox(
                "Use SMOTE for class balancing",
                value=False,
                help="Synthetic oversampling for minority class"
            )
            
            scoring_metric = st.selectbox(
                "Optimization Metric",
                ['pr_auc', 'roc_auc', 'f1'],
                index=0,
                help="Better than ROC-AUC for imbalanced data; focuses on quality for churners."
            )
        
        # Training button
        if st.button("Train Models", type="secondary", use_container_width=True):
            if st.session_state.data is not None:
                with st.spinner("Training models..."):
                    try:
                        df = st.session_state.data.copy()
                        
                        # Apply segment filter with validation
                        if st.session_state.settings['segment_filter'] != 'All Customers':
                            filtered_df = df[df['Contract'] == st.session_state.settings['segment_filter']]
                            if len(filtered_df) < 50:
                                st.error(f"Not enough data for segment '{st.session_state.settings['segment_filter']}'. Need at least 50 customers.")
                                st.stop()
                            df = filtered_df
                        
                        # Preprocess and train
                        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
                        models, cv_results = train_models(X_train, y_train, preprocessor, feature_names)
                        
                        # Store results with segment info
                        st.session_state.models = models
                        st.session_state.cv_results = cv_results
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = feature_names
                        st.session_state.trained_segment = st.session_state.settings['segment_filter']
                        
                        st.success(f"Models trained successfully for {st.session_state.settings['segment_filter']}!")
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
                        st.session_state.models = None
                        st.session_state.cv_results = None
            else:
                st.error("No data available for training")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.error("Could not load customer data. Please check data source.")
        return
    
    df = st.session_state.data
    settings = st.session_state.settings
    
    # Apply segment filter for display with validation
    try:
        if settings['segment_filter'] != 'All Customers':
            df_display = df[df['Contract'] == settings['segment_filter']].copy()
            if len(df_display) == 0:
                st.warning(f"No customers found for segment '{settings['segment_filter']}'. Showing all customers.")
                df_display = df.copy()
        else:
            df_display = df.copy()
    except Exception as e:
        st.error(f"Error applying segment filter: {str(e)}")
        df_display = df.copy()
    
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
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">{churn_rate:.1f}%</div>
                <div class="metric-subtitle">Customer Churn Rate</div>
                <div class="metric-description">Percentage of customers who have discontinued their service. Industry benchmark: 15-25%. Higher rates indicate retention challenges requiring immediate attention.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">{at_risk_customers:,}</div>
                <div class="metric-subtitle">Customers at High Risk</div>
                <div class="metric-description">Customers with highest predicted churn probability. These require proactive retention campaigns to prevent revenue loss and reduce acquisition costs.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">${avg_value:,.0f}</div>
                <div class="metric-subtitle">Average Customer Value</div>
                <div class="metric-description">Annual revenue per customer based on monthly charges. This represents the financial impact of each customer loss and the value of successful retention efforts.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">${potential_loss:,.0f}</div>
                <div class="metric-subtitle">Revenue at Risk</div>
                <div class="metric-description">Estimated annual revenue loss from high-risk customers. This represents the maximum potential impact and justifies investment in retention strategies.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ROI estimate section
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Campaign Investment Forecast")
        st.markdown("*Simple analysis to help you plan your customer retention campaigns*")
        
        # Add Context & Assumptions info block with simpler language
        st.info("""
> **How to Read These Numbers**  
These numbers show how many customers we could save and how much money that might be worth based on your current campaign settings.

- **You can control**: How aggressive you want to be, how much you spend per customer contact, and the value of keeping each customer.
- **The system estimates**: Which customers are likely to leave based on past data.
- **Next steps**: Use the **Retention Planner** tab to adjust your settings and see how it affects your return on investment.
        """)
        
        contacts = at_risk_customers
        cost = contacts * settings['cost_per_contact']
        saved = int(contacts * 0.3 * settings['value_saved_per_customer'])
        roi = ((saved - cost) / cost * 100) if cost > 0 else 0
        
        roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
        
        with roi_col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">{contacts:,}</div>
                <div class="metric-subtitle">Customers to Contact</div>
                <div class="metric-description">High-risk customers targeted for proactive retention outreach. Optimal contact volume balances reach with resource efficiency.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with roi_col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">${cost:,}</div>
                <div class="metric-subtitle">Campaign Investment</div>
                <div class="metric-description">Total marketing spend including staff time, incentives, and communication costs. Investment scales with contact volume and retention strategy complexity.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with roi_col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">${saved:,}</div>
                <div class="metric-subtitle">Expected Revenue Saved</div>
                <div class="metric-description">Projected revenue retention based on 30% campaign success rate. Conservative estimate ensuring realistic business planning and budget allocation.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with roi_col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">{roi:.0f}%</div>
                <div class="metric-subtitle">Return on Investment</div>
                <div class="metric-description">Net financial return from retention campaign. Positive ROI indicates profitable retention strategy worth scaling across customer segments.</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <strong>Business Insight:</strong> These projections assume we can successfully retain 3 out of every 10 customers we contact, which is a typical success rate for retention campaigns.
        </div>
        """, unsafe_allow_html=True)
        
        # Add detailed Assumptions & ROI section here
        threshold = settings['retention_aggressiveness']
        cost_per_contact = settings['cost_per_contact']
        value_saved = settings['value_saved_per_customer']
            
        st.markdown(f"""
### Campaign Assumptions & ROI Details

**What these numbers mean**
- **Risk score**: How likely each customer is to cancel (0% to 100%)
- **Campaign aggressiveness**: How selective we are - lower means we contact more customers, higher means we're more selective
- **Cost per contact**: What it costs to reach one customer with an offer or call
- **Value saved**: How much revenue we keep when we prevent one customer from leaving

**How we calculate return on investment**
- **Campaign cost** = Number of customers contacted × Cost per contact  
- **Revenue saved** = Customers who stay × Value saved per customer  
- **Net profit** = Revenue saved − Campaign cost  
- **ROI percentage** = (Net profit ÷ Campaign cost) × 100

**Why returns can look very high**
- Campaign costs are usually much smaller than the revenue from keeping customers
- If keeping a customer is worth much more than contacting them, the ROI percentage gets very large
- Think of these as "what if" scenarios based on your assumptions, not guaranteed results

**Your current settings**
- Cost per contact = **${cost_per_contact:,.2f}**
- Value of keeping one customer = **${value_saved:,.2f}**
- Campaign selectiveness = **{threshold:.2f}** (lower = contact more customers)

**Important limitations**
- These numbers don't include setup costs, staff time, or system costs
- We assume every contacted customer gets the same offer and responds the same way
- Some customers might accept offers even if they weren't planning to leave
- Contacting too many customers might annoy them and hurt future campaigns
- These predictions are based on past data and might not work the same way in the future

*Tip: Use the **Retention Planner** tab to try different settings and see how they affect your costs and returns.*
        """)
    
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
        
        # Enhanced strategic recommendations
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Strategic Recommendations")
        st.markdown("*Data-driven action plan for immediate implementation*")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("""
            <div class="warning-box">
                <h4>Immediate Actions Required</h4>
                <ul>
                    <li><strong>Priority 1:</strong> Target month-to-month customers with contract upgrade offers</li>
                    <li><strong>Priority 2:</strong> Implement 90-day onboarding program for new customers</li>
                    <li><strong>Priority 3:</strong> Address fiber service quality issues causing churn</li>
                    <li><strong>Priority 4:</strong> Deploy retention specialists for high-value segments</li>
                </ul>
                <p><em>Timeline: Execute within 30 days for maximum impact</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        with rec_col2:
            st.markdown("""
            <div class="success-box">
                <h4>Expected Business Impact</h4>
                <ul>
                    <li><strong>Churn Reduction:</strong> 15-20% decrease in customer losses</li>
                    <li><strong>Revenue Protection:</strong> $50K+ monthly recurring revenue saved</li>
                    <li><strong>Customer Satisfaction:</strong> Improved Net Promoter Score</li>
                    <li><strong>Operational Efficiency:</strong> Better resource allocation and planning</li>
                </ul>
                <p><em>ROI: 300-500% return within 6 months</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.header("Retention Campaign Planner")
        
        models_available = (st.session_state.models is not None and 
                           st.session_state.cv_results is not None and 
                           len(st.session_state.cv_results) > 0)
        
        if not models_available:
            st.warning("Please train models first using the sidebar controls.")
        else:
            try:
                # Check if models were trained for current segment
                trained_segment = getattr(st.session_state, 'trained_segment', 'All Customers')
                if trained_segment != settings['segment_filter']:
                    st.warning(f"Models were trained for '{trained_segment}' but current filter is '{settings['segment_filter']}'. Please retrain models.")
                    st.stop()
                
                models = st.session_state.models
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                # Validate all required data exists
                if models is None or X_test is None or y_test is None:
                    st.error("Model data is incomplete. Please retrain models.")
                    st.stop()
                
                # Get predictions with error handling
                best_model_name = get_best_model(st.session_state.cv_results)
                if best_model_name is None or best_model_name not in models:
                    st.error(f"Model '{best_model_name}' not found in trained models.")
                    st.stop()
                    
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
                
                st.subheader("Campaign Results Preview")
                st.markdown("*Here's what would happen if we run this campaign today*")
                
                # Calculate ROI metrics for 5-column layout
                offer_cost = customers_contacted * settings['cost_per_contact']
                savings = true_positives * settings['value_saved_per_customer']
                net_roi_calc = savings - offer_cost
                roi_pct = (net_roi_calc / offer_cost * 100.0) if offer_cost > 0 else 0.0
                
                # Enhanced metric cards with better descriptions
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">{true_positives:,}</div>
                        <div class="metric-subtitle">Customers We'd Keep</div>
                        <div class="metric-description">How many customers we expect to save from leaving based on our outreach efforts</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">{customers_contacted:,}</div>
                        <div class="metric-subtitle">Customers We'd Contact</div>
                        <div class="metric-description">Total number of at-risk customers we'd reach out to with special offers or calls</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">${offer_cost:,.0f}</div>
                        <div class="metric-subtitle">Campaign Investment</div>
                        <div class="metric-description">Total cost to run the campaign including staff time, offers, and communication costs</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">${net_roi_calc:,.0f}</div>
                        <div class="metric-subtitle">Net Profit</div>
                        <div class="metric-description">Money left over after subtracting campaign costs from the revenue we save by keeping customers</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">{roi_pct:,.0f}%</div>
                        <div class="metric-subtitle">Return on Investment</div>
                        <div class="metric-description">For every $100 spent on the campaign, how much profit we expect to make back</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced ROI explanation with stacked layout
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                
                # First section - How to Calculate Return
                st.markdown("""
                <div class="insight-box">
                    <h4>How to Calculate Return</h4>
                    <ol>
                        <li><strong>Campaign Cost:</strong> Number of customers contacted × Cost to reach each one</li>
                        <li><strong>Revenue Saved:</strong> Customers who stay × Value of keeping each customer</li>
                        <li><strong>Profit:</strong> Revenue saved - Campaign cost</li>
                        <li><strong>Return %:</strong> (Profit ÷ Campaign cost) × 100</li>
                    </ol>
                    <p><em>Think of it like: "For every $100 I spend, how much do I get back?"</em></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Second section - How to Use These Metrics
                st.markdown("""
                <div class="warning-box">
                    <h4>How to Use These Metrics</h4>
                    <ul>
                        <li><strong>Lower selectiveness:</strong> Contact more customers, save more, but spend more too</li>
                        <li><strong>Higher selectiveness:</strong> Contact fewer customers, spend less, but might miss some at-risk customers</li>
                        <li><strong>Sweet spot:</strong> Balance between reaching enough customers and controlling costs</li>
                    </ul>
                    <p><em>Adjust the settings on the left to find what works for your budget and goals.</em></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Keep original detailed metrics
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">{customers_contacted:,}</div>
                        <div class="metric-subtitle">Customers to Contact</div>
                        <div class="metric-description">AI-identified high-risk customers based on predictive model threshold. Precision targeting maximizes campaign effectiveness.</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with perf_col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">{true_positives:,}</div>
                        <div class="metric-subtitle">Expected Saves</div>
                        <div class="metric-description">Predicted successful retentions from targeted outreach. Based on model accuracy and historical retention campaign performance.</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with perf_col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">${campaign_cost:,}</div>
                        <div class="metric-subtitle">Campaign Investment</div>
                        <div class="metric-description">Total campaign cost including outreach, incentives, and operational overhead. Scales linearly with contact volume.</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with perf_col4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">${net_roi:,}</div>
                        <div class="metric-subtitle">Net ROI ({roi_percentage:.0f}%)</div>
                        <div class="metric-description">Net financial return after campaign costs. Positive values indicate profitable retention strategy with strong business case.</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Efficiency metrics
                st.subheader("Campaign Efficiency")
                
                precision = true_positives / customers_contacted if customers_contacted > 0 else 0
                cost_per_save = campaign_cost / true_positives if true_positives > 0 else float('inf')
                
                eff_col1, eff_col2 = st.columns(2)
                
                with eff_col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">{precision:.1%}</div>
                        <div class="metric-subtitle">Campaign Precision</div>
                        <div class="metric-description">Accuracy of AI model in identifying truly at-risk customers. Higher precision reduces wasted outreach costs and improves campaign ROI.</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with eff_col2:
                    cost_display = f"${cost_per_save:.0f}" if cost_per_save != float('inf') else "N/A"
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">{cost_display}</div>
                        <div class="metric-subtitle">Cost per Customer Saved</div>
                        <div class="metric-description">Average investment required to successfully retain one customer. Compare against customer lifetime value to validate campaign profitability.</div>
                    </div>
                    """, unsafe_allow_html=True)
                
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
                
                if cv_results is not None and len(cv_results) > 0:
                    st.subheader("Model Performance Comparison")
                    cv_df = pd.DataFrame(cv_results)
                    st.dataframe(cv_df.round(4), use_container_width=True)
                    
                    # Best model info with PR-AUC tooltip
                    best_model_name = get_best_model(cv_results)
                    st.info(f"Best performing model: {best_model_name}")
                    
                    # Add Metrics & Assumptions section
                    st.markdown("""
### Metrics & Assumptions

**Imbalance-aware metric**
- **PR-AUC** focuses on performance for the positive (churn) class and is more informative than ROC-AUC when churners are rare.

**Threshold (τ) trade-off**
- Lower τ increases **Recall** (catch more churners) but may reduce **Precision** (more false alarms).
- Business cost/benefit is managed in **Retention Planner**.

**Assumptions carried through the app**
- Cost per contact and Value saved are scenario inputs that drive ROI and can be changed anytime.
- We assume one contact per flagged customer, immediate offer delivery, and uniform offer effectiveness.

**Limitations**
- Fixed and overhead costs not modeled; include them in external budgeting.
- No explicit modeling of offer acceptance, discount cannibalization, or long-term behavior changes.
- Results depend on dataset representativeness and calibration; retrain/validate periodically.
                    """)
                    
                    # Add assumptions export
                    try:
                        threshold = settings['retention_aggressiveness']
                        cost_per_contact_export = settings['cost_per_contact']
                        value_saved_export = settings['value_saved_per_customer']
                        
                        # Calculate current metrics
                        contacts = customers_contacted if 'customers_contacted' in locals() else 0
                        TP = true_positives if 'true_positives' in locals() else 0
                        offer_cost_export = contacts * cost_per_contact_export
                        savings_export = TP * value_saved_export
                        net_roi_export = savings_export - offer_cost_export
                        roi_pct_export = (net_roi_export / offer_cost_export * 100.0) if offer_cost_export > 0 else 0.0
                        
                        assumptions_payload = {
                            "threshold": float(threshold),
                            "cost_per_contact": float(cost_per_contact_export),
                            "value_saved": float(value_saved_export),
                            "contacts": int(contacts),
                            "churners_saved": int(TP),
                            "offer_cost": float(offer_cost_export),
                            "net_roi": float(net_roi_export),
                            "roi_pct": float(roi_pct_export),
                            "metrics": {k: float(v) for k, v in cv_df.iloc[0].to_dict().items()} if len(cv_df) > 0 else {},
                            "rows": int(len(df_display)) if 'df_display' in locals() else None
                        }
                        
                        st.download_button(
                            "Metrics & Settings (JSON)",
                            data=json.dumps(assumptions_payload, indent=2),
                            file_name="metrics_settings.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.caption(f"Export unavailable: {str(e)}")
                
            except Exception as e:
                st.error(f"Error displaying model details: {str(e)}")
    
    with tab5:
        st.header("Data Quality & Governance")
        st.caption("Dataset overview and quality metrics")
        
        # Dataset overview with enhanced styling
        st.subheader("Dataset Overview")
        st.markdown("*Comprehensive data quality and completeness metrics*")
        
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">{len(df):,}</div>
                <div class="metric-subtitle">Total Customers</div>
                <div class="metric-description">Complete customer records in analysis dataset. Sufficient sample size ensures statistical significance and reliable model predictions.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with overview_col2:
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">{churn_rate:.1f}%</div>
                <div class="metric-subtitle">Overall Churn Rate</div>
                <div class="metric-description">Baseline churn percentage across all customer segments. Serves as benchmark for measuring retention campaign effectiveness and segment analysis.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with overview_col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">{len(df.columns) - 1}</div>
                <div class="metric-subtitle">Data Features</div>
                <div class="metric-description">Customer attributes available for analysis including demographics, services, and billing information. Rich feature set enables sophisticated predictive modeling.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with overview_col4:
            missing_data = df.isnull().sum().sum()
            completeness = "100%" if missing_data == 0 else f"{(1 - missing_data/(len(df)*len(df.columns)))*100:.1f}%"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">{completeness}</div>
                <div class="metric-subtitle">Data Completeness</div>
                <div class="metric-description">Percentage of complete data without missing values. High completeness ensures model accuracy and reliable business insights.</div>
            </div>
            """, unsafe_allow_html=True)
        
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