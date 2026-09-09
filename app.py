"""
Skin Disease Classifier — Home Page
-------------------------------------
Entry point for the multi-page Streamlit app. Use the sidebar to navigate
to the Predict, Model Performance, and About pages.

Usage:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺")

st.title("🩺 Skin Disease Classifier")

st.markdown(
    """
    Welcome! This tool uses a deep learning model (MobileNetV2, fine-tuned)
    to classify skin images into one of 22 categories, including common
    conditions like Acne, Eczema, Psoriasis, and Vitiligo, as well as
    healthy/normal skin.

    ### How to use this app
    Use the sidebar on the left to navigate:

    - **Predict** — upload a skin image and get a prediction
    - **Model Performance** — see how accurate the model is, and which
      conditions it handles well vs. poorly
    - **About** — details about the project, dataset, and model

    ---

    **Disclaimer:** This is a student/educational project. Predictions
    are **not** a medical diagnosis. Always consult a qualified
    dermatologist or doctor for any skin health concerns.
    """
)
