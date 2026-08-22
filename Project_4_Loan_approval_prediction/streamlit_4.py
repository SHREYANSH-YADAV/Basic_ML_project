import streamlit as st
import pandas as pd
import joblib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "Model", "loan_approval_model.pkl")
META_PATH = os.path.join(HERE, "Model", "loan_approval_metadata.json")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

st.title("🏦 Loan Approval Prediction")
st.divider()

with st.form("input_form"):
    st.subheader("Enter Details")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    with col2:
        applicant_income = st.number_input("Applicant Income (monthly)", 0, 100000, 5000)
        coapplicant_income = st.number_input("Co-applicant Income (monthly)", 0, 50000, 0)
        loan_amount = st.number_input("Loan Amount (in thousands)", 9, 700, 140)
        loan_term = st.selectbox("Loan Term (days)", [360, 180, 240, 120, 300])
    credit_history = st.selectbox("Credit History Meets Guidelines?", [1.0, 0.0], format_func=lambda x: "Yes" if x==1.0 else "No")
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }
    input_dict["Total_Income"] = input_dict["ApplicantIncome"] + input_dict["CoapplicantIncome"]
    input_dict["Loan_Income_Ratio"] = input_dict["LoanAmount"] / (input_dict["Total_Income"] / 1000 + 1)
    X = pd.DataFrame([input_dict])
    X = X[meta["numeric_cols"] + meta["categorical_cols"]]
    pred = model.predict(X)[0]
    label = meta["classes"][pred]
    proba = model.predict_proba(X)[0][pred]
    if label == "Y":
        st.success(f"### ✅ Loan Approved (confidence {proba*100:.1f}%)")
    else:
        st.error(f"### ❌ Loan Not Approved (confidence {proba*100:.1f}%)")

with st.expander("Model performance (test set)"):
    st.json(meta["metrics"][meta["best_model"]])