import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Load trained model
model = joblib.load("models/xgboost.pkl")

# Page configuration
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.markdown("<h1 style='text-align: center; color: crimson;'>🩺 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)

# Sidebar Inputs
with st.sidebar:
    st.header("Input Features")
    with st.expander("👤 Personal Info"):
        pregnancies = st.slider("Pregnancies", 0, 20, step=1)
        age = st.slider("Age (years)", 1, 100, step=1)

    with st.expander("🧪 Medical Parameters (Realistic Ranges)"):
        glucose = st.number_input("Glucose", min_value=50, step=1, help="Normal fasting: 70-99")
        blood_pressure = st.number_input("Blood Pressure", min_value=60, step=1, help="Normal: 80/120")
        skin_thickness = st.number_input("Skin Thickness", min_value=10, step=1, help="Normal: ~20")
        insulin = st.number_input("Insulin", min_value=15, step=1, help="Normal: ~16-166")
        bmi = st.number_input("BMI", min_value=10.0, step=0.1, help="Normal: 18.5–24.9")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, step=0.01, help="Genetic risk factor")

# Collect inputs
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

# Predict button
if st.button("🔍 Predict Diabetes Risk"):
    # Predict
    prediction = model.predict(input_data)[0]

    st.subheader("🔎 Result")
    if prediction == 1:
        st.error("⚠️ **You have Diabetes!**")
        st.markdown("🔬 *Please consult a doctor for further diagnosis.*")
    else:
        st.success("✅ **You don't have Diabetes**")
        st.markdown("💪 *Keep maintaining a healthy lifestyle!*")

    # Feature importance
    st.subheader("📊 Feature Importance")
    fig, ax = plt.subplots()
    plot_importance(model, ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("###### 👩‍💻 Built by *Arshiya K* ")
