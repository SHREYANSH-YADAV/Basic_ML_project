import streamlit as st
import pandas as pd
import joblib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "Model", "student_performance_model.pkl")
META_PATH = os.path.join(HERE, "Model", "student_performance_metadata.json")

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

st.title("🎓 Student Performance Prediction")
st.divider()

with st.form("input_form"):
    st.subheader("Enter Details")
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 30.0, 100.0, 80.0)
        study_hours = st.slider("Weekly Study Hours", 0.0, 20.0, 4.0)
        online_hours = st.slider("Online Study Hours/week", 0.0, 10.0, 1.5)
        assignment = st.slider("Assignment Score", 0.0, 100.0, 70.0)
    with col2:
        internal1 = st.slider("Internal Assessment 1", 0.0, 100.0, 65.0)
        internal2 = st.slider("Internal Assessment 2", 0.0, 100.0, 65.0)
        gpa = st.slider("Previous GPA", 0.0, 10.0, 7.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    parent_edu = st.selectbox("Parental Education", ["High School", "Graduate", "Postgraduate"])
    internet = st.selectbox("Internet Access", ["Yes", "No"])
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_dict = {
        "Attendance_Percent": attendance,
        "Weekly_Study_Hours": study_hours,
        "Assignment_Score": assignment,
        "Internal_Assessment_1": internal1,
        "Internal_Assessment_2": internal2,
        "Previous_GPA": gpa,
        "Online_Study_Hours": online_hours,
        "Gender": gender,
        "Extracurricular_Activities": extracurricular,
        "Parental_Education": parent_edu,
        "Internet_Access": internet,
    }
    input_dict["Avg_Internal_Assessment"] = (input_dict["Internal_Assessment_1"] + input_dict["Internal_Assessment_2"]) / 2
    input_dict["Total_Study_Hours"] = input_dict["Weekly_Study_Hours"] + input_dict["Online_Study_Hours"]
    X = pd.DataFrame([input_dict])
    X = X[meta["numeric_cols"] + meta["categorical_cols"]]
    pred = model.predict(X)[0]
    st.success(f"### Predicted Final Exam Score: {pred:.1f} / 100")
    st.progress(min(max(pred/100, 0.0), 1.0))

with st.expander("Model performance (test set)"):
    st.json(meta["metrics"][meta["best_model"]])