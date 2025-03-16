import streamlit as st
import numpy as np
import time
import tensorflow as tf  # Assuming your CNN model is built with TensorFlow/Keras

# Load your pre-trained CNN model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ddos_cnn_model.h5")  # Change with your model path
    return model

cnn_model = load_model()

# Function to generate random test data
def generate_random_data():
    flow_duration = np.random.randint(500, 5000)
    packet_length_mean = np.random.randint(100, 1000)
    flow_packets_per_s = np.random.uniform(10, 200)
    destination_port = np.random.choice([80, 443, 22, 53])
    
    return np.array([[flow_duration, packet_length_mean, flow_packets_per_s, destination_port]])

st.title("🔄 Live DDoS Attack Prediction")

# Button to start continuous prediction
if st.button("Start Continuous Prediction"):
    st.write("Generating live network traffic data and predicting continuously...")
    
    while True:
        # Generate new data
        input_data = generate_random_data()
        
        # Make a prediction using the CNN model
        prediction = cnn_model.predict(input_data)
        
        # Interpret prediction (assuming binary classification: 0 = Normal, 1 = DDoS Attack)
        result = "🚀 **DDoS Attack Detected!**" if prediction[0][0] > 0.5 else "✅ **Normal Traffic**"
        
        # Display data and prediction
        st.write("**Generated Input Data:**", input_data)
        st.write("**Prediction:**", result)
        
        # Wait before generating new data
        time.sleep(2)  # Adjust time interval (in seconds)
