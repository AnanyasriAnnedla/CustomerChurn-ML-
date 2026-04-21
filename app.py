import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

#Load artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Why Churn...", layout="wide")
st.title("Customer Churn Predictor ")
st.markdown("Fill in customer details below to predict churn and understand *why*.")

#Sidebar Inputs
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0, 120, 50)
total_charges = monthly_charges * tenure  #derived

senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

#Build raw row
raw = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": float(total_charges),
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "gender": "Male",  #neutral default
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
}

input_df = pd.DataFrame([raw])

#Encode to match training
input_df = pd.get_dummies(input_df, drop_first=True)

#Align columns with training features
input_df = input_df.reindex(columns=feature_names, fill_value=0)

#Scale
input_scaled = scaler.transform(input_df)

#Predict
pred = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.metric("Prediction", "🔴 Will Churn" if pred == 1 else "🟢 Will Stay")
with col2:
    st.metric("Churn Probability", f"{prob*100:.1f}%")

#SHAP Explanation
st.markdown("### 🔍 Why this prediction?")

explainer = shap.LinearExplainer(model, scaler.transform(
    pd.DataFrame([dict(zip(feature_names, [0]*len(feature_names)))])
), feature_perturbation="interventional")

shap_values = explainer.shap_values(input_scaled)

fig, ax = plt.subplots(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_scaled[0],
        feature_names=feature_names
    ),
    show=False
)
st.pyplot(fig)

st.markdown("---")
st.caption("Built on Logistic Regression + SHAP | Telco Customer Churn Dataset")