import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("wine_quality_model.pkl")
    scaler = joblib.load("wine_quality_scaler.pkl")
    features = joblib.load("wine_quality_features.pkl")
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found.")
    st.stop()

st.title("🍷 Wine Quality Predictor")
st.write(
    "Enter the chemical properties of a red wine sample to predict whether "
    "its quality is **GOOD** (quality score ≥ 7) or **BAD** (quality score < 7)."
)

defaults = {
    "fixed acidity": 8.3, "volatile acidity": 0.53, "citric acid": 0.27,
    "residual sugar": 2.5, "chlorides": 0.087, "free sulfur dioxide": 15.9,
    "total sulfur dioxide": 46.5, "density": 0.9967, "pH": 3.31,
    "sulphates": 0.66, "alcohol": 10.4,
}

slider_ranges = {
    "fixed acidity": (4.0, 16.0, 0.1), "volatile acidity": (0.05, 1.6, 0.01),
    "citric acid": (0.0, 1.0, 0.01), "residual sugar": (0.5, 16.0, 0.1),
    "chlorides": (0.01, 0.62, 0.001), "free sulfur dioxide": (1.0, 72.0, 1.0),
    "total sulfur dioxide": (6.0, 289.0, 1.0), "density": (0.9900, 1.0040, 0.0001),
    "pH": (2.7, 4.0, 0.01), "sulphates": (0.3, 2.0, 0.01), "alcohol": (8.0, 15.0, 0.1),
}

st.subheader("Input Features")
col1, col2 = st.columns(2)
user_input = {}
feature_list = list(feature_names)

for i, feat in enumerate(feature_list):
    lo, hi, step = slider_ranges.get(feat, (0.0, 100.0, 0.1))
    target_col = col1 if i % 2 == 0 else col2
    user_input[feat] = target_col.slider(
        feat.title(), min_value=float(lo), max_value=float(hi),
        value=float(defaults.get(feat, (lo + hi) / 2)), step=float(step)
    )

if st.button("Predict Quality", type="primary"):
    input_df = pd.DataFrame([user_input])[feature_list]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]

    st.subheader("Result")
    if prediction == 1:
        st.success("✅ Predicted quality: **GOOD** (quality ≥ 7)")
    else:
        st.warning("❌ Predicted quality: **BAD** (quality < 7)")

    if proba is not None:
        st.write(f"Confidence — GOOD: {proba[1]:.2%} | BAD: {proba[0]:.2%}")

    with st.expander("See input values sent to the model"):
        st.dataframe(input_df)

st.divider()
st.caption("Model: Decision Tree Classifier, tuned with GridSearchCV. Trained on the UCI Wine Quality (red wine) dataset.")