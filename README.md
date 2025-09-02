# 📊 Customer Churn Insights Dashboard  

**Customer Churn Insights** is an interactive analytics and machine learning dashboard built with **Python**, **Streamlit**, **Scikit-learn**, and **Plotly**, designed to analyze telecom customer churn and provide actionable business insights. It combines predictive modeling with business-focused ROI calculations to support decision-making for retention strategies.  

---

## 🚀 Features  

- 🔍 Executive Summary with KPIs (Churn Rate, At-Risk Customers, Potential Customers Saved, Estimated Savings, ROI %)  
- 🧩 Segments & Drivers (high-risk cohorts, top churn drivers with plain-English explanations)  
- 🛠 Retention Planner (threshold slider, cost/value inputs, Net ROI & ROI % calculations)  
- 📈 Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC/PR curves)  
- 📂 Data & Quality checks (dataset snapshot, missing values, contract distributions)  

---

## 🛠️ Tech Stack  

| Layer              | Tech                                    |  
|--------------------|-----------------------------------------|  
| Frontend / UI      | Streamlit                               |  
| ML Models          | Scikit-learn (Logistic Regression, Random Forest) |  
| Imbalance Handling | Imbalanced-learn (SMOTE)                |  
| Data Wrangling     | Pandas, NumPy                           |  
| Visualization      | Matplotlib, Plotly                      |  
| Explainability     | SHAP                                    |

---

The Churn Insights Dashboard runs as a **Streamlit web application** that integrates data preprocessing, machine learning models, explainability, and ROI-focused retention planning.  
The app ingests the Telco Customer Churn dataset, preprocesses categorical and numerical features, trains and evaluates models with cross-validation, and surfaces insights in a **five-tab business-oriented interface**:  

1. **Executive Summary** – KPIs and quick business impact snapshot  
2. **Segments & Drivers** – high-risk groups and churn drivers  
3. **Retention Planner** – scenario planning with ROI calculations  
4. **Details & Methods** – metrics, confusion matrix, ROC & PR curves  
5. **Data & Quality** – dataset overview and health checks  

All metrics and ROI values are calculated based on user-adjustable assumptions (threshold, cost per contact, value saved). Exports are available for metrics, feature importance, and scored customers.  
