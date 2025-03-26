import pickle
import streamlit as st
import pandas as pd

#load model
@st.cache_resource
def load_model():
  with open('model.pkl','rb') as file:
    model= pickle.load(file)
return load_model


st.title('Diabetis Predictor')
st.write('This app help as to check whether a patient has diabetis')

st.sidebar.header('Input Features')
