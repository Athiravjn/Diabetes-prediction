import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0.0, 200.0)
blood_pressure = st.number_input("Blood Pressure", 0.0, 150.0)
skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0)
insulin = st.number_input("Insulin", 0.0, 900.0)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is unlikely to have diabetes.")
