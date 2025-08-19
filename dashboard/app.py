import streamlit as st
import pandas as pd
import requests
import os
from datetime import date

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AlphaCare Insurance Dashboard", layout="wide")
st.title("🚗 AlphaCare Insurance Analytics Dashboard")
st.markdown("A platform for **insurance risk analytics & predictions**.")

# Sidebar inputs
st.sidebar.header("📋 Enter Policy Details")
age = st.sidebar.number_input("Age", 18, 100, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Other"])
province = st.sidebar.text_input("Province", "Gauteng")
custom_value = st.sidebar.number_input("Custom Value Estimate", 1000.0, 100000.0, 25000.0, 500.0)
premium = st.sidebar.number_input("Total Premium", 100.0, 50000.0, 1800.0, 50.0)
zip_code = st.sidebar.number_input("Zip Code", 0, 9999, 4122)
vehicle_intro = st.sidebar.date_input("Vehicle Introduction Date", date(2020, 1, 1))

# Prediction button
if st.sidebar.button("🔮 Predict Risk"):
    payload = {
        "Age": age,
        "Gender": gender,
        "VehicleType": vehicle_type,
        "Province": province,
        "CustomValueEstimate": custom_value,
        "TotalPremium": premium,
        "ZipCode": zip_code,
        "VehicleIntroDate": vehicle_intro.isoformat()
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result = response.json()
        if result.get("error"):
            st.error(f"Prediction error: {result['error']}")
        else:
            st.success(f"✅ Predicted Risk Level: {result['predicted_risk']}")
            if result.get("probability"):
                st.subheader("🔹 Probability Breakdown")
                st.dataframe(pd.DataFrame(result["probability"], columns=["Probability"]))
    except Exception as e:
        st.error(f"Connection error: {e}")

# Sample analytics
st.subheader("📊 Example Portfolio Overview")
sample_path = os.path.join("data", "sample_claims.csv")
if os.path.exists(sample_path):
    df = pd.read_csv(sample_path)
    st.dataframe(df.head(20))
    st.bar_chart(df["Province"].value_counts())
else:
    st.info(f"No sample dataset found at `{sample_path}`. Add it to display analytics.")
