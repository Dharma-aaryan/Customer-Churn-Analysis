Customer Churn Insights Dashboard

An interactive Streamlit-based Machine Learning dashboard designed to analyze telecom customer churn, surface actionable business insights, and simulate retention strategies with ROI calculations. This project transforms raw customer data into a decision-support tool that speaks directly to both business leaders and data practitioners, combining statistical rigor with clear storytelling.

🚀 Key Features
🔍 Executive Summary 
KPI cards for Churn Rate, Customers at Risk, Potential Customers Saved, Estimated Savings, ROI% 
One-click snapshot of overall churn health and business impact

🧩 Segments & Drivers
Identifies high-risk customer groups (e.g., contract type × tenure bins) Highlights top churn drivers in plain English (↑ increases risk, ↓ decreases risk)
Explains “why churn happens” with model coefficients and feature importance

🛠 Retention Planner
Retention aggressiveness slider (τ): adjust how many customers are flagged at risk
Business inputs: cost per contact, value saved per churn prevented
Outputs: Churners Saved, Contacts Sent, Offer Cost, Net ROI, ROI %
ROI explainer panel clarifies assumptions and why ROI % can appear very high
Advanced controls (model choice, SMOTE) available in an optional expander

📈 Details & Methods
Full model evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
Confusion matrix, ROC curve, Precision–Recall curve
Glossary of metrics explained in simple terms

📂 Data & Quality
Dataset overview: size, missing values, and contract type distribution
Ensures transparency and trust in the inputs driving predictions

📊 ROI Assumptions
ROI is scenario-based and depends heavily on user inputs:

Cost per contact: expense of reaching one customer
Value saved per churn prevented: revenue preserved if a churner stays
Retention aggressiveness (τ): decision threshold for flagging at-risk customers

Formulas:
Offer cost = Contacts × Cost per contact
Savings = Churners Saved × Value saved
Net ROI = Savings − Offer cost
ROI % = (Net ROI ÷ Offer cost) × 100

⚠️ ROI % can look very high when value saved ≫ contact cost. This is mathematically correct but should be treated as directional guidance rather than a guarantee. Real campaigns also consider overhead, offer uptake rates, and long-term customer behavior.

📈 Example Insights
Month-to-month contracts with short tenure and higher monthly charges drive the highest churn.
Transitioning customers to longer-term contracts significantly reduces churn risk.
Even conservative assumptions show positive ROI for targeted retention campaigns.

⚙️ Tech Stack
Streamlit – interactive app framework
Pandas / NumPy – data wrangling
Scikit-learn – ML models (Logistic Regression, Random Forest)
Imbalanced-learn (SMOTE) – class imbalance handling
Matplotlib / Plotly – charts & visualizations
SHAP – model explainability (optional)

🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.
