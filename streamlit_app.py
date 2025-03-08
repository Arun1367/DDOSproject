import streamlit as st

st.title('DDOS ML PREDICTION🛜')

st.write('This is an app for predicting DDOS attack or Normal ')
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load Pretrained Models
def load_model(model_name):
    if model_name == "CNN":
        return tf.keras.models.load_model("cnn_model.h5")
    elif model_name == "DNN":
        return tf.keras.models.load_model("dnn_model.h5")
    return None

# Feature Inputs
st.title("DDoS Attack Prediction App")
st.sidebar.header("User Input Features")

features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets',
            'Total Backward Packets', 'Total Length of Fwd Packets',
            'Total Length of Bwd Packets', 'Fwd Packet Length Max',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Fwd Packet Length Std', 'Bwd Packet Length Max',
            'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min']

user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.number_input(feature, value=0.0)

user_data = np.array(list(user_input.values())).reshape(1, -1)

# Choose Model
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression", "CNN", "RNN", "GAN"])
model = load_model(model_choice)

# Preprocess Input (Standardization)
scaler = pickle.load(open("scaler.pkl", "rb"))
scaled_input = scaler.transform(user_data)

# Make Prediction
if st.button("Predict"):
    if model_choice in ["Random Forest", "Logistic Regression"]:
        prediction = model.predict(scaled_input)
    else:
        prediction = model.predict(scaled_input)
        prediction = np.argmax(prediction, axis=1)  # Convert softmax output to class
    
    result = "DDoS Attack Detected!" if prediction[0] == 1 else "Traffic is Normal."
    st.write(f"Prediction: {result}")

st.write("App is ready for testing!")
