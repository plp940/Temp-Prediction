# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import joblib

# Load model and scaler
model = keras.models.load_model("temperature.h5", compile=False)   # Must be a Keras model (.h5)
scaler = joblib.load("Scaler.pkl")                  # MinMaxScaler saved using joblib

# Streamlit UI setup
st.set_page_config(page_title="Temperature Prediction App", page_icon="🌡️", layout="wide")
st.title("🌡️ Temperature Prediction App")
st.write("This app predicts the next temperature based on the last 30 values.")

# Input from user
input_text = st.text_area("Enter the last 30 temperature readings (comma-separated):", 
                          "30, 32, 31, 29, 28, 30, 33, 32, 31, 30, 29, 28, 30, 29, 31, 33, 32, 30, 29, 30, 28, 27, 29, 31, 30, 32, 34, 33, 32, 30")

if st.button("Predict"):
    try:
        # Convert input to list of floats
        input_values = [float(x.strip()) for x in input_text.split(",")]

        if len(input_values) != 30:
            st.error("Please enter exactly 30 temperature values.")
        else:
            # Scale and reshape
            input_array = np.array(input_values).reshape(-1, 1)
            scaled_input = scaler.transform(input_array).reshape(1, 30, 1)

            # Make prediction
            scaled_output = model.predict(scaled_input)
            predicted_temp = scaler.inverse_transform(scaled_output)

            st.success(f"🌤️ Predicted Next Temperature: {predicted_temp[0][0]:.2f}°C")

    except ValueError:
        st.error("Invalid input! Please enter only numeric temperature values, separated by commas.")
