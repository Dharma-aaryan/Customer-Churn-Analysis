import streamlit as st
import pandas as pd
import numpy as np
from utils.data import load_and_clean_data, preprocess_data
from utils.model import train_models, get_best_model

# Page configuration
st.set_page_config(
    page_title="Churn Insights Dashboard",
    layout="wide"
)

def main():
    st.title("Churn Insights Dashboard")
    
    # Load data
    try:
        if 'data' not in st.session_state:
            st.session_state.data = load_and_clean_data()
        
        df = st.session_state.data
        
        if df is None:
            st.error("Could not load data")
            return
            
        st.success(f"Data loaded: {len(df):,} customers")
        
        # Simple tabs
        tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Models"])
        
        with tab1:
            st.header("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{len(df):,}")
            
            with col2:
                churn_rate = (df['Churn'] == 'Yes').mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            with col3:
                avg_monthly = df['MonthlyCharges'].mean()
                st.metric("Avg Monthly", f"${avg_monthly:.2f}")
            
            with col4:
                avg_tenure = df['tenure'].mean()
                st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
            
            st.subheader("Data Sample")
            st.dataframe(df.head())
        
        with tab2:
            st.header("Churn Analysis")
            
            # Contract analysis
            contract_analysis = df.groupby('Contract').agg({
                'Churn': lambda x: (x == 'Yes').mean() * 100
            }).round(1)
            
            st.subheader("Churn Rate by Contract Type")
            st.dataframe(contract_analysis)
            
            # Basic statistics
            st.subheader("Key Statistics")
            stats_data = {
                'Metric': ['Total Customers', 'Churned Customers', 'Retained Customers', 'Average Monthly Revenue'],
                'Value': [
                    len(df),
                    len(df[df['Churn'] == 'Yes']),
                    len(df[df['Churn'] == 'No']),
                    f"${df['MonthlyCharges'].mean():.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data))
        
        with tab3:
            st.header("Model Training")
            
            if st.button("Train Simple Model"):
                with st.spinner("Training models..."):
                    try:
                        # Preprocess data
                        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
                        
                        # Train models
                        models, cv_results = train_models(X_train, y_train, preprocessor, feature_names)
                        
                        # Store results
                        st.session_state.models = models
                        st.session_state.cv_results = cv_results
                        
                        st.success("Models trained successfully!")
                        
                        # Show results
                        if cv_results:
                            st.subheader("Model Performance")
                            cv_df = pd.DataFrame(cv_results)
                            st.dataframe(cv_df)
                        
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
            
            # Show model status
            if 'models' in st.session_state and st.session_state.models:
                st.success("Models are trained and ready!")
                
                if 'cv_results' in st.session_state and st.session_state.cv_results:
                    best_model = get_best_model(st.session_state.cv_results)
                    st.info(f"Best model: {best_model}")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check the data files and try refreshing the page.")

if __name__ == "__main__":
    main()