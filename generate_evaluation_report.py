import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "skin_disease_final.keras"
CLASSES_PATH = BASE_DIR / "class_indices.json"
TEST_DIR = BASE_DIR / "dataset" / "SkinDisease" / "test"
REPORT_PATH = BASE_DIR / "evaluation_report.csv"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    class_names = [
        class_name for class_name, _ in sorted(class_indices.items(), key=lambda item: item[1])
    ]

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

    probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(probs, axis=1)
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

    summary = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Value": accuracy,
                "Percentage": accuracy * 100,
            },
            {
                "Metric": "Weighted Precision",
                "Value": precision,
                "Percentage": precision * 100,
            },
            {
                "Metric": "Weighted Recall",
                "Value": recall,
                "Percentage": recall * 100,
            },
            {
                "Metric": "Weighted F1",
                "Value": f1,
                "Percentage": f1 * 100,
            },
        ]
    )

    per_class_rows = []
    for label in class_names:
        metrics = report[label]
        per_class_rows.append(
            {
                "Class": label,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1-score"],
                "Support": metrics["support"],
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)
    summary.to_csv(REPORT_PATH, index=False)
    per_class_df.to_csv(REPORT_PATH.with_name("evaluation_report_per_class.csv"), index=False)

    print("=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("\nSaved summary report to:", REPORT_PATH)
    print("Saved per-class report to:", REPORT_PATH.with_name("evaluation_report_per_class.csv"))
    print("\nConfusion matrix shape:", matrix.shape)
    print("\nTop-level metrics are saved in CSV format for download.")


if __name__ == "__main__":
    main()
