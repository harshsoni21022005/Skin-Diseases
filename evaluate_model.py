"""
Evaluate the saved skin disease model on the real test set.
Outputs accuracy, precision, recall, F1-score, and confusion matrix.
"""

from pathlib import Path
import json

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "skin_disease_final.keras"
CLASSES_PATH = BASE_DIR / "class_indices.json"
TEST_DIR = BASE_DIR / "dataset" / "SkinDisease" / "test"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found: {CLASSES_PATH}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)

    # Sort class names in the same order used in training / label mapping
    class_names = [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

    print("Loading model...")
    model = load_model(str(MODEL_PATH))

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
        classes=class_names,
    )

    print("Predicting on test set...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print("\nConfusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
