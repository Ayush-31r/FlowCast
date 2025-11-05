import streamlit as st
import pandas as pd
import joblib
import os


# Get the directory of this file (app/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build safe absolute paths to the models
model_time_path = os.path.join(BASE_DIR, "../Models/flowcast_travel_time_model.pkl")
model_fare_path = os.path.join(BASE_DIR, "../Models/flowcast_fare_model.pkl")

# Load models safely
try:
    model_time = joblib.load(model_time_path)
    model_fare = joblib.load(model_fare_path)
except FileNotFoundError:
    import streamlit as st
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
distance_km = st.sidebar.slider("Distance (km)", 0.5, 25.0, 6.0, 0.5)
hod = st.sidebar.slider("Hour of Day (0–23)", 0, 23, 9)
rush_hour = 1 if hod in list(range(8,11)) + list(range(17,21)) else 0
st.sidebar.write(f"Rush Hour: {'Yes' if rush_hour else 'No'}")

# ------------------------------------------------------
# Predict button
# ------------------------------------------------------
if st.sidebar.button("Predict"):
    X_input = pd.DataFrame([[distance_km, hod, rush_hour]],
                           columns=["distance_km", "hod", "rush_hour"])
    
    pred_time = model_time.predict(X_input)[0]
    pred_fare = model_fare.predict(X_input)[0]

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
