import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

st.write("DEBUG - Files in this folder:", os.listdir("."))

@st.cache_resource
def load_artifacts():
    model = joblib.load("wine_quality_model.pkl")
    scaler = joblib.load("wine_quality_scaler.pkl")
    features = joblib.load("wine_quality_features.pkl")
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model files not found. Error: {e}")
    st.stop()

st.title("🍷 Wine Quality Predictor")
st.write("App loaded successfully!")