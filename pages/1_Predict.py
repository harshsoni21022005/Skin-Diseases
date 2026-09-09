"""
Predict page — upload an image and get a disease prediction.
"""

import sys
import os
import tempfile
from io import BytesIO

# Make sure predict.py (in the project root) is importable from this pages/ subfolder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image, ImageOps

from predict import load_trained_model, load_class_mapping, predict_image

MODEL_PATH = "skin_disease_final.keras"
CLASSES_PATH = "class_indices.json"


@st.cache_resource
def get_model_and_classes():
    model = load_trained_model(MODEL_PATH)
    idx_to_class = load_class_mapping(CLASSES_PATH)
    return model, idx_to_class


st.set_page_config(page_title="Predict - Skin Disease Classifier", page_icon="🔍")

st.title("🔍 Predict")
st.write("Upload a skin image to get a prediction.")

model, idx_to_class = get_model_and_classes()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        temp_path = os.path.join(tempfile.gettempdir(), "temp_upload.jpg")
        image.save(temp_path)

        with st.spinner("Classifying..."):
            results = predict_image(temp_path, model, idx_to_class, top_k=5)
    except Exception as exc:
        st.error(f"Could not process the uploaded image. Please upload a valid JPG, JPEG, or PNG file.\n\nDetails: {exc}")
        st.stop()

    st.subheader("Predictions")
    for rank, (label, confidence) in enumerate(results, start=1):
        st.write(f"**{rank}. {label}** — {confidence * 100:.2f}%")
        st.progress(min(confidence, 1.0))

    best_label, best_confidence = results[0]
    if best_confidence < 0.4:
        st.warning(
            f"Confidence is low ({best_confidence * 100:.1f}%) — this prediction is uncertain. "
            "Please consult a dermatologist for an accurate diagnosis."
        )
    else:
        st.success(f"Most likely: **{best_label}** ({best_confidence * 100:.1f}% confidence)")
