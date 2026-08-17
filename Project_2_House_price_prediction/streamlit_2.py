import streamlit as st
import pandas as pd
import joblib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "Model", "house_price_model.pkl")
META_PATH = os.path.join(HERE, "Model", "house_price_metadata.json")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

st.title("🏠 House Price Prediction")
st.divider()

with st.form("input_form"):
    st.subheader("Enter Details")
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (Sq Ft)", 300, 5000, 1200)
        bedrooms = st.number_input("Bedrooms", 1, 6, 3)
        bathrooms = st.number_input("Bathrooms", 1, 6, 2)
        age = st.slider("Age of Property (years)", 0.0, 45.0, 5.0)
    with col2:
        amenities = st.slider("Amenities Count", 0, 8, 3)
        distance = st.slider("Distance from City Center (km)", 0.2, 40.0, 6.0)
        parking = st.number_input("Parking Spaces", 0, 3, 1)
        location = st.selectbox("Location Tier", ["Tier1_City", "Tier2_City", "Tier3_City"])
    furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_dict = {
        "Area_SqFt": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_Years": age,
        "Amenities_Count": amenities,
        "Distance_From_City_Center_KM": distance,
        "Parking_Spaces": parking,
        "Location_Tier": location,
        "Furnishing_Status": furnishing,
    }
    input_dict["Price_per_SqFt_Proxy"] = input_dict["Amenities_Count"] / (input_dict["Area_SqFt"] + 1) * 1000
    input_dict["Room_Density"] = (input_dict["Bedrooms"] + input_dict["Bathrooms"]) / (input_dict["Area_SqFt"] / 1000 + 0.1)
    X = pd.DataFrame([input_dict])
    X = X[meta["numeric_cols"] + meta["categorical_cols"]]
    pred = model.predict(X)[0]
    st.success(f"### Predicted House Price: ₹ {pred:.2f} Lakhs")

with st.expander("Model performance (test set)"):
    st.json(meta["metrics"][meta["best_model"]])