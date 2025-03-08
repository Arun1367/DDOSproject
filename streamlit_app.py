import streamlit as st
import numpy as np
import tensorflow as tf
import joblib  # For loading the scaler

# Load the pre-trained model
model = tf.keras.models.load_model("ddos_cnn_model.h5")

# Load the pre-trained scaler
scaler = joblib.load("scaler.pkl")

# Define feature names based on your dataset
feature_names = [
    "Destination", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean"
]

# Streamlit UI
st.title("🚀 DDoS Attack Detection with CNN")

# Collect user inputs for all features
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Convert inputs to a NumPy array
X_new = np.array([list(inputs.values())]).reshape(1, -1)

st.write("🔍 **Input Data:**", X_new)

# Prediction Function
def predict_ddos(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)  # Scale input
    prediction = model.predict(features)[0][0]
    return "🚨 DDoS Attack Detected!" if prediction > 0.5 else "✅ Benign Traffic"

# Predict button
if st.button("🔍 Predict"):
    result = predict_ddos(X_new)
    st.subheader(result)
