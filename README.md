📊 Churn Insights Dashboard

An interactive Streamlit-based Machine Learning dashboard that analyzes telecom customer churn, provides business-focused insights, and supports retention decision-making with ROI calculations.

This project takes the popular Telco Customer Churn dataset and demonstrates how to build a complete analytics and modeling pipeline — from data preprocessing and ML training to explainability and business ROI simulation — all wrapped in a clean, user-friendly interface.

🚀 Features
🔍 Executive Summary

KPI cards: Churn rate, Customers at risk, Potential customers saved, Estimated savings, ROI %

High-level snapshot of dataset size, quality, and churn baseline

🧩 Segments & Drivers

Identifies high-risk customer segments (e.g., contract type × tenure bins)

Highlights top churn drivers with plain-English explanations (↑ increases churn risk, ↓ lowers churn risk)

Visual segment-level churn rates for targeted retention strategies

🛠 Retention Planner

Retention aggressiveness slider (τ) → adjust decision threshold

Business inputs: cost per contact, value saved per customer

Outputs: Churners saved, Contacts sent, Offer cost, Net ROI, ROI %

Advanced controls (model choice, SMOTE) hidden in an Advanced panel

Clear ROI explanation and assumptions surfaced directly in the UI

📈 Details & Methods

Model evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

Confusion matrix, ROC curve, Precision–Recall curve

Glossary of metrics in simple terms for non-technical users

📂 Data & Quality

Dataset overview: row/column counts, missing values

Distribution of contract types and other key categorical features

⚙️ Tech Stack

Python 3.10+

Streamlit
 – interactive UI framework

Pandas / NumPy – data wrangling

Scikit-learn – preprocessing & ML models (Logistic Regression, Random Forest)

Imbalanced-learn (SMOTE) – class imbalance handling

Matplotlib / Plotly – visualizations

SHAP – model explainability (optional)

📊 ROI Assumptions

ROI metrics in the dashboard are scenario-based and depend heavily on assumptions:

Cost per contact: cost of reaching one customer with an offer

Value saved per churn prevented: estimated value if one churner stays

Retention aggressiveness (τ): threshold for flagging customers at risk

Formula:

Offer cost = Contacts × Cost per contact

Savings = Churners saved × Value saved

Net ROI = Savings − Offer cost

ROI % = (Net ROI ÷ Offer cost) × 100

⚠️ ROI % can look very high if value saved ≫ contact cost. Treat ROI as directional — real-world campaigns include fixed costs, acceptance rates, and long-term effects.

📈 Example Insights

Customers on month-to-month contracts with short tenure and higher monthly charges are most at risk.

Moving at-risk customers to longer-term contracts may significantly reduce churn.

With current assumptions, retention campaigns show positive ROI even with moderate false positives.

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
