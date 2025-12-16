import pandas as pd
import numpy as np
import joblib
import streamlit as st

model = joblib.load('loan_approval_model.pkl')
state_encoder = joblib.load('state_encoder.pkl')

st.title("🏦 Loan Approval Predictor")
st.write("Enter your details to see if you qualify for a loan.")

amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=5000)
fico = st.slider("Credit Score (FICO)", 300, 850, 650)
dti = st.slider("Debt-to-Income Ratio (%)", 0, 100, 20)
emp_length = st.slider("Years of Employment", 0, 10, 2)
state = st.selectbox("State", state_encoder.classes_)

if st.button("Predict Approval"):
    # Prepare data for model
    # Convert state to number using the loaded encoder
    state_num = state_encoder.transform([state])[0]
    
    # Create row (Must match the order we trained on: amount, fico, dti, state, emp_length)
    # Check your training column order! In the code above it was: amount, fico, dti, state, emp_length
    features = np.array([[amount, fico, dti, state_num, emp_length]])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"✅ Approved! (Probability: {probability:.0%})")
        st.balloons()
    else:
        st.error(f"❌ Rejected. (Probability: {probability:.0%})")
        st.write("Tip: Try lowering the Loan Amount or improving your Credit Score.")
