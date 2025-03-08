import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

st.title('DDOS ML PREDICTION🛜')
st.write('This is an app for predicting DDOS attack or Normal')
# Function to load data
def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='latin1', quoting=1, error_bad_lines=False)) #added quoting and error_bad_lines.
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error loading file: {e}")
        return None

# Replace with the correct raw file URL
file_path = 'https://github.com/ABHISHEKSASA/DATA/blob/main/ddos_attack%20(1).csv'  # <--- Replace with your raw URL, make sure it is a csv.
df = load_data(file_path) # or encoding='utf-8'
# rest of your code


if df is not None:
    # Preprocessing
    if 'Label' in df.columns:
        df['Label'] = df['Label'].map({'Benign': 0, 'DDoS': 1})
        X = df.drop('Label', axis=1)
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

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
            st.sidebar.header("Input Features")
            input_features = {col: st.sidebar.number_input(f"{col}", value=X[col].mean()) for col in X.columns}

            if st.button("Predict"):
                input_df = pd.DataFrame([input_features])
                input_reshaped = input_df.values.reshape(1, input_df.shape[1], 1)
                prediction = (cnn_model.predict(input_reshaped) > 0.5).astype("int32")

                st.success("Benign Traffic Detected!") if prediction[0] == 0 else st.error("DDoS Attack Detected!")

        if __name__ == "__main__":
            main()
    else:
        st.error("The loaded data does not contain a 'Label' column. Please ensure your data is correctly formatted.")
