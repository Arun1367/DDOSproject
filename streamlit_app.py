import streamlit as st
import numpy as np
import time
import tensorflow as tf
import pickle

# Load the CNN model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ddos_cnn_model.h5")  # Ensure this path is correct
    return model

cnn_model = load_model()

# Load the saved scaler
@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

scaler = load_scaler()

st.title("🔄 Live DDoS Attack Prediction")

# Function to generate random test data
def generate_random_data():
    destination_port = np.random.choice([80, 443, 22, 53])
    flow_duration = np.random.randint(500, 5000)
    fwd_packet_length_mean = np.random.randint(100, 1000)
    bwd_packet_length_mean = np.random.randint(100, 1000)
    flow_bytes_per_s = np.random.uniform(1000, 1000000)
    flow_packets_per_s = np.random.uniform(10, 200)
    flow_iat_mean = np.random.uniform(0.1, 500)
    
    data = np.array([[destination_port, flow_duration, fwd_packet_length_mean, 
                      bwd_packet_length_mean, flow_bytes_per_s, flow_packets_per_s, flow_iat_mean]], dtype=np.float32)
    return data

# Continuous Prediction Button
if st.button("Start Continuous Prediction"):
    st.write("Generating live network traffic data and predicting...")

    for _ in range(5):  # Prevent infinite loop for testing
        input_data = generate_random_data()
        
        # Scale the input data
        scaled_input_data = scaler.transform(input_data)
        
        # **Fixing the input shape issue**
        model_input_shape = cnn_model.input_shape
        
        if len(model_input_shape) == 3:  # If model expects 3D input
            scaled_input_data = scaled_input_data.reshape(1, 7, 1)
        elif len(model_input_shape) == 2:  # If model expects 2D input
            scaled_input_data = scaled_input_data.reshape(1, 7)
        
        # Predict using the CNN model
        prediction = cnn_model.predict(scaled_input_data)

        # Interpret prediction (0 = Normal, 1 = DDoS)
        result = "🚀 **DDoS Attack Detected!**" if prediction[0][0] > 0.5 else "✅ **Normal Traffic**"

        # Display data and prediction
        st.write("### **Generated Input Data:**")
        st.write(f"Destination Port: {input_data[0][0]}")
        st.write(f"Flow Duration: {input_data[0][1]}")
        st.write(f"Fwd Packet Length Mean: {input_data[0][2]}")
        st.write(f"Bwd Packet Length Mean: {input_data[0][3]}")
        st.write(f"Flow Bytes/s: {input_data[0][4]}")
        st.write(f"Flow Packets/s: {input_data[0][5]}")
        st.write(f"Flow IAT Mean: {input_data[0][6]}")
        st.write("### **Prediction:**", result)

        # Wait before generating new data
        time.sleep(2)
