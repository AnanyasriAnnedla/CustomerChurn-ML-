# Customer Churn Predictor + Explainer

Predicts whether a telecom customer will churn — and explains *why* using SHAP values.

## Live Demo
👉 [Try it here]()

## Project Overview
This project predicts whether a customer will leave a telecom service (called churn)
using Machine Learning. It helps companies identify high-risk customers and take
preventive action. Unlike a standard churn predictor, this app explains *why* a
customer is predicted to churn using SHAP waterfall charts.

## Objective
Build a classification model that predicts customer churn based on customer data
such as tenure, services used, and billing information — with visual explainability.

## Dataset
Telco Customer Churn — Kaggle

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- SHAP
- Streamlit
- Matplotlib, Seaborn

## Project Workflow
1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Encoding
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. SHAP Explainability
8. Streamlit Web App

## Model
Logistic Regression
- Accuracy: 81%
- Confusion Matrix: [[933, 103], [151, 222]]

## Known Limitations
- TotalCharges is derived from Monthly Charges × Tenure in the UI, whereas in
  reality these are independent variables. This may slightly affect predictions
  for edge case inputs.
- Logistic Regression (81% accuracy) is used for interpretability. A tree-based
  model like XGBoost would likely perform better but is less transparent.

## How to Run
1. Clone the repository
   git clone https://github.com/AnanyasriAnnedla/CustomerChurn-ML-

2. Navigate to the project folder
   cd CustomerChurn-ML-

3. Install dependencies
   pip install -r requirements.txt

4. Train and save the model
   python train.py

5. Launch the app
   streamlit run app.py