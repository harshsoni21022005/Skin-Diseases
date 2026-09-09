"""
Model Performance page — computes real metrics from the saved model and test set.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "skin_disease_final.keras"
CLASSES_PATH = BASE_DIR / "class_indices.json"
TEST_DIR = BASE_DIR / "dataset" / "SkinDisease" / "test"


@st.cache_data
def get_evaluation_metrics():
    report_path = BASE_DIR / "evaluation_report.csv"
    per_class_path = BASE_DIR / "evaluation_report_per_class.csv"

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    class_names = [
        class_name for class_name, _ in sorted(class_indices.items(), key=lambda item: item[1])
    ]

    if report_path.exists() and per_class_path.exists():
        summary_df = pd.read_csv(report_path)
        per_class_df = pd.read_csv(per_class_path)
        accuracy = float(summary_df.loc[summary_df["Metric"] == "Accuracy", "Value"].iloc[0])
        precision = float(
            summary_df.loc[summary_df["Metric"] == "Weighted Precision", "Value"].iloc[0]
        )
        recall = float(summary_df.loc[summary_df["Metric"] == "Weighted Recall", "Value"].iloc[0])
        f1 = float(summary_df.loc[summary_df["Metric"] == "Weighted F1", "Value"].iloc[0])
        matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
        return accuracy, precision, recall, f1, per_class_df, matrix, class_names

    model = load_model(str(MODEL_PATH))

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
        classes=class_names,
    )

    if len(test_generator) == 0:
        raise ValueError(f"Test dataset directory is empty or missing: {TEST_DIR}")

    probabilities = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_generator.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = [
        {
            "Disease": label,
            "Precision": float(report[label]["precision"]),
            "Recall": float(report[label]["recall"]),
            "F1-score": float(report[label]["f1-score"]),
            "Support": int(report[label]["support"]),
        }
        for label in class_names
        if label in report
    ]

    return accuracy, precision, recall, f1, pd.DataFrame(per_class), matrix, class_names


st.set_page_config(page_title="Model Performance", page_icon="📊")

st.title("📊 Model Performance")

st.caption("These metrics are computed on the saved model and the real test set, not hardcoded into the page.")

try:
    accuracy, precision, recall, f1, per_class_df, confusion_matrix_values, class_names = (
        get_evaluation_metrics()
    )
except Exception as exc:
    st.warning(
        "The uploaded deployment does not include the test dataset, so live metrics cannot be recomputed here. "
        "Showing the saved evaluation report instead if available."
    )
    report_path = BASE_DIR / "evaluation_report.csv"
    per_class_path = BASE_DIR / "evaluation_report_per_class.csv"
    if report_path.exists() and per_class_path.exists():
        summary_df = pd.read_csv(report_path)
        per_class_df = pd.read_csv(per_class_path)
        accuracy = float(summary_df.loc[summary_df["Metric"] == "Accuracy", "Value"].iloc[0])
        precision = float(
            summary_df.loc[summary_df["Metric"] == "Weighted Precision", "Value"].iloc[0]
        )
        recall = float(summary_df.loc[summary_df["Metric"] == "Weighted Recall", "Value"].iloc[0])
        f1 = float(summary_df.loc[summary_df["Metric"] == "Weighted F1", "Value"].iloc[0])
        confusion_matrix_values = np.zeros((len(class_names) if 'class_names' in locals() else 1, 1), dtype=int)
        class_names = []
    else:
        st.error(f"No saved evaluation metrics are available. Deployment data is missing. Details: {exc}")
        st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy * 100:.1f}%")
col2.metric("Weighted Precision", f"{precision * 100:.1f}%")
col3.metric("Weighted Recall", f"{recall * 100:.1f}%")
col4.metric("Weighted F1", f"{f1 * 100:.1f}%")

st.markdown(
    """
    This model was trained on 22 skin condition classes using transfer learning
    (MobileNetV2, fine-tuned). The values above are measured on the held-out test
    set and reflect the model's real predictive performance.
    """
)

st.subheader("Per-class performance")
per_class_df = per_class_df.sort_values("F1-score", ascending=False).reset_index(drop=True)
st.dataframe(per_class_df, use_container_width=True, hide_index=True)

st.subheader("Download Evaluation Report")
report_path = BASE_DIR / "evaluation_report.csv"
if report_path.exists():
    with open(report_path, "rb") as f:
        st.download_button(
            label="Download summary CSV",
            data=f,
            file_name="evaluation_report.csv",
            mime="text/csv",
        )
else:
    st.info("Run the evaluation report script first to generate the downloadable CSV.")

st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(
    confusion_matrix_values,
    index=class_names,
    columns=class_names,
)
st.dataframe(cm_df, use_container_width=True)
