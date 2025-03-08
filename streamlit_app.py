import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

st.title("DDoS Attack Detection (CNN)")

# Load the dataset with debugging information
@st.cache_data
def load_real_data():
    file_path = "https://github.com/ABHISHEKSASA/DATA/blob/main/ddos_attack.csv"  # Make sure this file is uploaded
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

        # Debugging Step: Print Column Names
        st.write("Dataset Columns:", df.columns.tolist())

        # Remove unwanted spaces and special characters
        df.columns = df.columns.str.strip()

        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

df = load_real_data()

if df is not None:
    st.write("Dataset Loaded Successfully ✅")

    # Check if 'Label' exists
    if any(col.lower() == 'label' for col in df.columns):
        # Standardize column name (in case of variations like ' label ', 'LABEL', etc.)
        df.rename(columns={col: 'Label' for col in df.columns if col.lower() == 'label'}, inplace=True)

        # Convert labels to binary
        df['Label'] = df['Label'].map({'Benign': 0, 'DDoS': 1})

        # Keep only numerical features
        df = df.select_dtypes(include=[np.number]).dropna()

        X = df.drop(columns=['Label'])
        y = df['Label']

        # Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train CNN Model
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
            model.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test), verbose=0)
            return model

        cnn_model = train_cnn(X_train, y_train, X_test, y_test)

        # Streamlit UI
        def main():
            st.sidebar.header("Input Features")
            input_features = {}
            for col, mean_val in zip(X.columns, X.mean()):
                input_features[col] = st.sidebar.number_input(f"{col}", value=float(mean_val))

            if st.button("Predict"):
                input_df = pd.DataFrame([input_features])
                input_scaled = scaler.transform(input_df)
                input_reshaped = input_scaled.reshape(1, input_scaled.shape[1], 1)
                prediction = (cnn_model.predict(input_reshaped) > 0.5).astype("int32")

                if prediction[0] == 1:
                    st.error("🚨 DDoS Attack Detected! 🚨")
                else:
                    st.success("✅ Benign Traffic Detected!")

        if __name__ == "__main__":
            main()

    else:
        st.error("❌ No 'Label' column found in the dataset. Check column names above.")
