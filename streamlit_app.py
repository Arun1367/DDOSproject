import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Load dataset from CSV
@st.cache_data
def load_data():
    file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        return df
    return None

df = load_data()

if df is not None:
    # Check if 'Label' column exists
    if "Label" not in df.columns:
        st.error("❌ No 'Label' column found in dataset. Please check your file!")
    else:
        # Preprocessing
        df['Label'] = df['Label'].map({'Benign': 0, 'DDoS': 1})
        X = df.drop(columns=['Label'])
        y = df['Label']

        # Normalize input features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # CNN Model
        def train_cnn(X_train, y_train, X_test, y_test):
            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = keras.Sequential([
                keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test), verbose=1)
            
            # Display accuracy
            loss, accuracy = model.evaluate(X_test_reshaped, y_test)
            st.write(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

            return model, scaler

        # Train model
        cnn_model, scaler = train_cnn(X_train, y_train, X_test, y_test)

        # Streamlit App Interface
        st.title("🔍 DDoS Attack Detection with CNN")

        # Custom Feature Selection
        selected_features = st.multiselect("Select Features for Prediction", X.columns.tolist(), default=X.columns.tolist())

        if selected_features:
            # User Inputs for Selected Features
            input_features = {}
            for feature in selected_features:
                input_features[feature] = st.sidebar.number_input(f"{feature}", value=float(X[feature].mean()))

            if st.button("🔎 Predict"):
                # Convert input into DataFrame
                input_df = pd.DataFrame([input_features])

                # Scale input
                input_scaled = scaler.transform(input_df)

                # Reshape input
                input_reshaped = input_scaled.reshape(1, input_scaled.shape[1], 1)

                # Make Prediction
                prediction = (cnn_model.predict(input_reshaped) > 0.5).astype("int32")

                # Display Result
                if prediction[0] == 1:
                    st.error("🚨 DDoS Attack Detected!")
                else:
                    st.success("✅ Benign Traffic Detected!")
