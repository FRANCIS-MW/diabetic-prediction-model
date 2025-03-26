import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Title
st.title("Diabetes Predictor")
st.write("This app predicts whether a patient has diabetes based on input symptoms.")

# Sidebar - Input Features
st.sidebar.header("Input Features")

# Define the exact features the model expects
features = [
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia",
    "Genital thrush", "visual blurring", "Itching", "Irritability",
    "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"
]

# Create input widgets for each feature (Yes=1, No=0)
input_data = {feature: st.sidebar.radio(feature, ["No", "Yes"]) for feature in features}

# Convert Yes/No inputs to 0/1 format
input_features = [1 if input_data[feat] == "Yes" else 0 for feat in features]

# Predict button
if st.sidebar.button("Predict"):
    # Reshape input for prediction
    input_array = np.array(input_features).reshape(1, -1)

    # Debugging - Check input shape
    expected_features = model.coef_.shape[1]  # Expected number of features
    st.write(f"Input shape: {input_array.shape}, Model expects: {expected_features}")

    if input_array.shape[1] != expected_features:
        st.error(f"Feature mismatch! Expected {expected_features}, but got {input_array.shape[1]}.")
    else:
        prediction = model.predict(input_array)[0]

        # Display result
        if prediction == 1:
            st.error("The model predicts that the patient has diabetes.")
        else:
            st.success("The model predicts that the patient does not have diabetes.")
