import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ------------------------------------------------------
# Safe model loading
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_time_path = os.path.join(BASE_DIR, "../Models/flowcast_travel_time_model.pkl")
model_fare_path = os.path.join(BASE_DIR, "../Models/flowcast_fare_model.pkl")

try:
    model_time = joblib.load(model_time_path)
    model_fare = joblib.load(model_fare_path)
except FileNotFoundError:
    st.error("❌ Model files not found. Check your folder structure and capitalization ('Models').")
    st.stop()

# ------------------------------------------------------
# App config
# ------------------------------------------------------
st.set_page_config(page_title="FlowCast : Bangalore Travel Time & Fare Predictor", layout="centered")
st.title("🚗 FlowCast : Bangalore Travel Time & Fare Predictor")
st.write("Estimate **average inter-ward travel time and fare** in Bangalore using Uber Movement data.")

# ------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------
st.sidebar.header("Input Parameters")

# Distance input
distance_km = st.sidebar.slider("Distance (km)", 0.5, 25.0, 6.0, 0.5)

# Time input
selected_time = st.sidebar.time_input("Select Time of Day", datetime.strptime("09:00", "%H:%M").time())
hod = selected_time.hour

# Manual Rush Hour toggle
rush_hour_option = st.sidebar.radio("Rush Hour", ["Yes", "No"])
rush_hour = 1 if rush_hour_option == "Yes" else 0

# ------------------------------------------------------
# Predict button
# ------------------------------------------------------
if st.sidebar.button("Predict"):
    X_input = pd.DataFrame([[distance_km, hod, rush_hour]], columns=["distance_km", "hod", "rush_hour"])
    
    pred_time = model_time.predict(X_input)[0]
    pred_fare = model_fare.predict(X_input)[0]

    # If not rush hour, reduce travel time by 5 minutes (300 seconds)
    if rush_hour == 0:
        pred_time = max(0, pred_time - 300)

    # Display results
    st.subheader("Predictions")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🕒 Estimated Travel Time", value=f"{pred_time/60:.2f} min")
    with col2:
        st.metric(label="💰 Estimated Fare", value=f"₹{pred_fare:.2f}")

# ------------------------------------------------------
# Footer
# ------------------------------------------------------
st.markdown("---")
st.caption("Built with Random Forest | Data: Uber Movement Bangalore | Model by Ayush Rai")
