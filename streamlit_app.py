import streamlit as st

st.title('DDOS ML PREDICTION🛜')

st.write('This is an app for predicting DDOS attack or Normal ')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Sample Data (Replace with your actual data loading)
def create_dummy_data():
    np.random.seed(42)  # For reproducibility
    num_samples = 1000
    data = {
        'Destination Port': np.random.randint(1, 65536, num_samples),
        'Flow Duration': np.random.randint(1000, 1000000, num_samples),
        'Total Fwd Packets': np.random.randint(1, 100, num_samples),
        'Total Backward Packets': np.random.randint(1, 100, num_samples),
        'Total Length of Fwd Packets': np.random.randint(0, 10000, num_samples),
        'Total Length of Bwd Packets': np.random.randint(0, 10000, num_samples),
        'Fwd Packet Length Max': np.random.randint(0, 1000, num_samples),
        'Fwd Packet Length Min': np.random.randint(0, 500, num_samples),
        'Fwd Packet Length Mean': np.random.uniform(0, 750, num_samples),
        'Fwd Packet Length Std': np.random.uniform(0, 200, num_samples),
        'Bwd Packet Length Max': np.random.randint(0, 1000, num_samples),
        'Bwd Packet Length Min': np.random.randint(0, 500, num_samples),
        'Bwd Packet Length Mean': np.random.uniform(0, 750, num_samples),
        'Bwd Packet Length Std': np.random.uniform(0, 200, num_samples),
        'Flow Bytes/s': np.random.uniform(0, 100000, num_samples),
        'Flow Packets/s': np.random.uniform(0, 10000, num_samples),
        'Flow IAT Mean': np.random.uniform(0, 500000, num_samples),
        'Flow IAT Std': np.random.uniform(0, 200000, num_samples),
        'Flow IAT Max': np.random.randint(0, 1000000, num_samples),
        'Flow IAT Min': np.random.randint(0, 500000, num_samples),
        'Label': np.random.choice(['Benign', 'Malicious'], num_samples),
    }
    return pd.DataFrame(data)

df = create_dummy_data()

# Preprocessing
df['Label'] = df['Label'].map({'Benign': 0, 'Malicious': 1})
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training Functions
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_cnn(X_train, y_train, X_test, y_test):
    X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = keras.Sequential([
        keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test), verbose=0)
    return model

def train_rnn(X_train, y_train, X_test, y_test):
    X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(X_train.shape[1], 1)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test), verbose=0)
    return model

# Streamlit App
def main():
    st.title("Network Intrusion Detection App")

    st.sidebar.header("Input Features")
    input_features = {}
    for col in X.columns:
        input_features[col] = st.sidebar.number_input(f"{col}", value=X[col].mean())

    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "CNN", "RNN"])

    if st.button("Predict"):
        input_df = pd.DataFrame([input_features])

        if model_choice == "Random Forest":
            model = train_random_forest(X_train, y_train)
            prediction = model.predict(input_df)
        elif model_choice == "Logistic Regression":
            model = train_logistic_regression(X_train, y_train)
            prediction = model.predict(input_df)
        elif model_choice == "CNN":
            model = train_cnn(X_train, y_train, X_test, y_test)
            input_reshaped = input_df.values.reshape(1, input_df.shape[1], 1)
            prediction = (model.predict(input_reshaped) > 0.5).astype("int32")
        elif model_choice == "RNN":
            model = train_rnn(X_train, y_train, X_test, y_test)
            input_reshaped = input_df.values.reshape(1, input_df.shape[1], 1)
            prediction = (model.predict(input_reshaped) > 0.5).astype("int32")

        if prediction[0] == 1:
            st.error("Malicious Traffic Detected!")
        else:
            st.success("Benign Traffic Detected!")

if __name__ == "__main__":
    main()
