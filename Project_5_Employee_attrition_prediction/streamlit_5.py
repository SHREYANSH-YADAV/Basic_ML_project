import streamlit as st
import pandas as pd
import joblib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "Model", "employee_attrition_model.pkl")
META_PATH = os.path.join(HERE, "Model", "employee_attrition_metadata.json")

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="🧑‍💼", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

st.title("🧑‍💼 Employee Attrition Prediction")
st.divider()

with st.form("input_form"):
    st.subheader("Enter Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        income = st.number_input("Monthly Income", 10000, 200000, 50000)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager", "HR Executive", "Manufacturing Director"])
    with col2:
        distance = st.slider("Distance From Home (km)", 1, 30, 10)
        years = st.slider("Years at Company", 0, 37, 3)
        job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        wlb = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    training = st.slider("Training Times Last Year", 0, 6, 2)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_dict = {
        "Age": age,
        "Gender": gender,
        "MonthlyIncome": income,
        "Department": department,
        "JobRole": job_role,
        "DistanceFromHome_KM": distance,
        "YearsAtCompany": years,
        "JobSatisfaction": job_sat,
        "WorkLifeBalance": wlb,
        "EnvironmentSatisfaction": env_sat,
        "TrainingTimesLastYear": training,
        "OverTime": overtime,
        "MaritalStatus": marital,
    }
    input_dict["Income_per_YearAtCompany"] = input_dict["MonthlyIncome"] / (input_dict["YearsAtCompany"] + 1)
    input_dict["Satisfaction_Index"] = (input_dict["JobSatisfaction"] + input_dict["WorkLifeBalance"] + input_dict["EnvironmentSatisfaction"]) / 3
    X = pd.DataFrame([input_dict])
    X = X[meta["numeric_cols"] + meta["categorical_cols"]]
    pred = model.predict(X)[0]
    label = meta["classes"][pred]
    proba = model.predict_proba(X)[0][pred]
    if label == "Yes":
        st.error(f"### ⚠️ High Attrition Risk (confidence {proba*100:.1f}%)")
    else:
        st.success(f"### ✅ Likely to Stay (confidence {proba*100:.1f}%)")

with st.expander("Model performance (test set)"):
    st.json(meta["metrics"][meta["best_model"]])