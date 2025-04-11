
import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
try:
    with open("best_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except pickle.UnpicklingError:
    st.error("Failed to load the model file. Please check the file format.")

try:
    with open("best_model.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except pickle.UnpicklingError:
    st.error("Failed to load the scaler file. Please check the file format.")

# Streamlit app title
st.title("Email Open Prediction App")

# Input fields for features
customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
emails_opened = st.number_input("Emails Opened in the Past", min_value=0, max_value=50, value=5)
emails_clicked = st.number_input("Emails Clicked After Opening", min_value=0, max_value=50, value=2)
purchase_history = st.number_input("Total Purchase History ($)", min_value=0, value=1500)
time_spent = st.number_input("Average Time Spent on Website (min)", min_value=0.0, value=5.0)
days_since_last_open = st.number_input("Days Since Last Open", min_value=0, max_value=365, value=30)
engagement_score = st.number_input("Customer Engagement Score", min_value=0.0, value=70.0)
clicked_previous = st.selectbox("Clicked Previous Emails", [0, 1])
device_type = st.selectbox("Device Type (0: Desktop, 1: Mobile)", [0, 1])

# Prediction button
if st.button("Predict"):
    # Preprocess input data
    input_data = np.array([[customer_age, emails_opened, emails_clicked, purchase_history, 
                            time_spent, days_since_last_open, engagement_score, 
                            clicked_previous, device_type]])
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]
    
    # Display result
    if prediction == 1:
        st.success(f"The customer is likely to open the email! (Probability: {probability:.2f})")
    else:
        st.warning(f"The customer is unlikely to open the email. (Probability: {probability:.2f})")


