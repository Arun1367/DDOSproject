import streamlit as st

st.title('DDOS ML PREDICTION🛜')

st.write('This is an app for predicting DDOS attack or Normal ')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Sample Data (Replace with your actual data loading)
def create_dummy_data():
    np.random.seed(42)
    num_samples = 1000
    data = {
        'Destination Port': np.random.randint(1, 65536, num_samples),
        'Flow Duration': np.random.randint(1000, 1000000, num_samples),
        'Total Fwd Packets': np.random.randint(1, 100, num_samples),
        'Total Backward Packets': np.random.randint(1, 100, num_samples),
        'Total Length of Fwd Packets': np.random.randint(0, 10000, num_samples),
        'Total Length of Bwd Packets': np.random.randint(0, 10000, num_samples),
        'Label': np.random.choice(['Benign', 'DDoS'], num_samples),
    }
    return pd.DataFrame(data)

df = create_dummy_data()

# Preprocessing
df['Label'] = df['Label'].map({'Benign': 0, 'DDoS': 1})
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model Training
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

# Train the CNN model
cnn_model = train_cnn(X_train, y_train, X_test, y_test)

# Streamlit App
def main():
    st.title("DDoS Attack Detection (CNN)")

    st.sidebar.header("Input Features")
    input_features = {}
    for col in X.columns:
        input_features[col] = st.sidebar.number_input(f"{col}", value=X[col].mean())

    if st.button("Predict"):
        input_df = pd.DataFrame([input_features])
        input_reshaped = input_df.values.reshape(1, input_df.shape[1], 1)
        prediction = (cnn_model.predict(input_reshaped) > 0.5).astype("int32")

        if prediction[0] == 1:
            st.error("DDoS Attack Detected!")
        else:
            st.success("Benign Traffic Detected!")

if __name__ == "__main__":
    main()
