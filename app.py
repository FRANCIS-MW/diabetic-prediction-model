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
# List only the 10 features used during training.
features = [
    "Polyuria", 
    "Polydipsia", 
    "weakness", 
    "Polyphagia", 
    "visual blurring", 
    "delayed healing", 
    "partial paresis", 
    "muscle stiffness", 
    "Alopecia", 
    "Obesity"
]

# Create input widgets for each feature (assume binary Yes/No inputs)
input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.radio(feature, ("No", "Yes"))

# Convert Yes/No inputs to binary (0/1)
input_features = [1 if input_data[feat] == "Yes" else 0 for feat in features]

# Predict when button is clicked
if st.sidebar.button("Predict"):
    # Reshape input for the model
    input_array = np.array(input_features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    
    # Display result
    if prediction == 1:
        st.error("The model predicts that the patient has diabetes.")
    else:
        st.success("The model predicts that the patient does not have diabetes.")
