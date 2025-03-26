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
# Define the features expected by your model.
# (Adjust the list if your training used different or additional features.)
features = [
    "Polyuria", 
    "Polydipsia", 
    "sudden weight loss", 
    "weakness", 
    "Polyphagia", 
    "Genital thrush", 
    "visual blurring", 
    "Itching", 
    "Irritability", 
    "delayed healing", 
    "partial paresis", 
    "muscle stiffness", 
    "Alopecia", 
    "Obesity"
]

# Create input widgets for each feature.
# We assume a binary input for each symptom: Yes (1) or No (0)
input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.radio(feature, ("No", "Yes"))

# Convert Yes/No inputs to 0/1 format in the correct order.
input_features = [1 if input_data[feat] == "Yes" else 0 for feat in features]

# (Optional) If you wish to include additional numerical inputs, for example Age:
# age = st.sidebar.slider("Age", 0, 100, 50)
# Then, if Age was used in your training, insert it at the appropriate position:
# input_features.insert(0, age)

# Predict button
if st.sidebar.button("Predict"):
    # Reshape input features for prediction
    input_array = np.array(input_features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    
    # (Optional) Get prediction probability
    # prob = model.predict_proba(input_array)[0][prediction]
    
    # Display result
    if prediction == 1:
        st.error("The model predicts that the patient has diabetes.")
    else:
        st.success("The model predicts that the patient does not have diabetes.")
