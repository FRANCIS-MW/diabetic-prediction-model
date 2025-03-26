import pickle
import streamlit as st
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model  # Fixed return statement

model = load_model()  # Call the function

st.title('Diabetes Predictor')
st.write('This app helps to check whether a patient has diabetes')

st.sidebar.header('Input Features')
