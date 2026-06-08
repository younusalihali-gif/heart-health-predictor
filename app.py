# ============================================================
# Heart Health Predictor — Streamlit Web App
# ============================================================

import streamlit as st
import joblib
import numpy as np

# --- Load the saved model ---
model = joblib.load('heart_disease_model.pkl')

# --- Page config ---
st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="❤️",
    layout="centered"
)

# --- Header ---
st.title("❤️ Heart Health Predictor")
st.markdown("Enter patient details below to predict heart disease risk.")
st.markdown("---")

# --- Input fields ---
st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age",              min_value=1,   max_value=120, value=45)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol     = st.number_input("Cholesterol",      min_value=100, max_value=600, value=200)
    thalach  = st.number_input("Max Heart Rate",   min_value=60,  max_value=220, value=150)
    oldpeak  = st.number_input("ST Depression",    min_value=0.0, max_value=10.0,value=1.0, step=0.1)

with col2:
    sex     = st.selectbox("Sex",             options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp      = st.selectbox("Chest Pain Type", options=[0,1,2,3], format_func=lambda x: {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-Anginal", 3:"Asymptomatic"}[x])
    fbs     = st.selectbox("Fasting Blood Sugar > 120", options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
    restecg = st.selectbox("Resting ECG",     options=[0,1,2], format_func=lambda x: {0:"Normal", 1:"ST Abnormality", 2:"LV Hypertrophy"}[x])
    exang   = st.selectbox("Exercise Angina", options=[1,0],   format_func=lambda x: "Yes" if x==1 else "No")
    slope   = st.selectbox("ST Slope",        options=[0,1,2], format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x])
    ca      = st.selectbox("Major Vessels (0-3)", options=[0,1,2,3])
    thal    = st.selectbox("Thal",            options=[0,1,2,3], format_func=lambda x: {0:"Normal", 1:"Fixed Defect", 2:"Reversible Defect", 3:"Unknown"}[x])

st.markdown("---")

# --- Predict button ---
if st.button("🔍 Predict", use_container_width=True):

    # Arrange inputs in same order as training data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk — This patient shows signs of heart disease.")
    else:
        st.success(f"✅ Low Risk — This patient appears healthy.")

    st.markdown(f"**Confidence:** {max(probability)*100:.1f}%")

    with st.expander("See probability breakdown"):
        st.write(f"Probability of No Disease : {probability[0]*100:.1f}%")
        st.write(f"Probability of Disease    : {probability[1]*100:.1f}%")

st.markdown("---")
st.caption("Built with ❤️ using Logistic Regression | UCI Heart Disease Dataset | Accuracy: 84.21%")
