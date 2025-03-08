import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  # To load the saved scaler

# Load the pre-trained model
model = tf.keras.models.load_model("ddos_cnn_model.h5")

# Load the pre-trained scaler (optional if scaling was done)
scaler = joblib.load("scaler.pkl")

# Define prediction function
def predict_ddos(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)  # Scale input
    prediction = model.predict(features)[0][0]
    return "🚨 DDoS Attack Detected!" if prediction > 0.5 else "✅ Benign Traffic"

# Streamlit UI
st.title("DDoS Attack Detection with CNN")

# Input fields
feature_inputs = []
for i in range(10):  # Change 10 to the number of features in your dataset
    feature = st.number_input(f"Feature {i+1}", value=0.0)
    feature_inputs.append(feature)

# Predict button
if st.button("Predict"):
    result = predict_ddos(feature_inputs)
    st.subheader(result)
