import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('xgb.pkl')

# Define feature names
feature_names = [
    "MACDmacd", "KDJK", "KDJD", "KDJj", "WR"
]

# Streamlit user interface
st.title("Stock State Prediction")

# X1: input
MACDmacd = st.number_input("MACDmacd:", min_value=-200, max_value=200, value=50)

# X3: input
KDJK = st.number_input("KDJK:", min_value=0, max_value=100, value=10)

KDJD = st.number_input("KDJD:", min_value=0, max_value=100, value=10)

KDJj = st.number_input("KDJj:", min_value=-20, max_value=100, value=10)

WR = st.number_input("WR:", min_value=0, max_value=100, value=10)

# Process inputs and make predictions
feature_values = [MACDmacd,KDJK,KDJD,KDJj,WR]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"The model predicts that your probability of up is {probability:.1f}%. "
        )
    else:
        advice = (
            f"The model predicts that your probability of down is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
    
    
