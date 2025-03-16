import streamlit as st
import numpy as np
import time
import tensorflow as tf  

# Load the CNN model (Ensure the model file is in the same directory)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ddos_cnn_model (1).h5")  # Local model file
    return model

cnn_model = load_model()

# Function to generate random test data based on your feature set
def generate_random_data():
    destination_port = np.random.choice([80, 443, 22, 53])
    flow_duration = np.random.randint(500, 5000)
    fwd_packet_length_mean = np.random.randint(100, 1000)
    bwd_packet_length_mean = np.random.randint(100, 1000)
    flow_bytes_per_s = np.random.uniform(1000, 1000000)
    flow_packets_per_s = np.random.uniform(10, 200)
    flow_iat_mean = np.random.uniform(0.1, 500)

    # Return data as a NumPy array with correct shape (1 sample, 7 features)
    return np.array([[destination_port, flow_duration, fwd_packet_length_mean, 
                      bwd_packet_length_mean, flow_bytes_per_s, flow_packets_per_s, flow_iat_mean]])

st.title("🔄 Live DDoS Attack Prediction")

# Continuous Prediction Button
if st.button("Start Continuous Prediction"):
    st.write("Generating live network traffic data and predicting...")

    for _ in range(5):  # Prevent infinite loop by limiting predictions
        # Generate new data
        input_data = generate_random_data()
        
        # Make a prediction using the CNN model
        prediction = cnn_model.predict(input_data)
        
        # Interpret prediction (assuming binary classification: 0 = Normal, 1 = DDoS Attack)
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
        time.sleep(2)  # Adjust time interval (in seconds)
