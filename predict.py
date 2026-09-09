"""
Skin Disease Prediction Script
-------------------------------
Loads the trained MobileNetV2-based classifier and predicts the disease
class for a single image.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --top 5
    python predict.py path/to/image.jpg --model path/to/model.keras --classes path/to/class_indices.json

Requirements (same environment used for training):
    pip install tensorflow numpy pillow
"""

import argparse
import json
import os
import sys

import numpy as np


def load_class_mapping(classes_path):
    if not os.path.exists(classes_path):
        sys.exit(f"ERROR: class mapping file not found: {classes_path}")
    with open(classes_path, "r") as f:
        class_indices = json.load(f)
    # class_indices.json maps {"ClassName": index}; we need the reverse for prediction
    idx_to_class = {int(v): k for k, v in class_indices.items()}
    return idx_to_class


def load_trained_model(model_path):
    if not os.path.exists(model_path):
        sys.exit(f"ERROR: model file not found: {model_path}")
    # Imported here so --help works even without tensorflow installed
    from tensorflow.keras.models import load_model
    return load_model(model_path)


def predict_image(img_path, model, idx_to_class, img_size=(224, 224), top_k=3):
    if not os.path.exists(img_path):
        sys.exit(f"ERROR: image not found: {img_path}")

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image

    img = keras_image.load_img(img_path, target_size=img_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    results = [(idx_to_class[i], float(preds[i])) for i in top_indices]
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict skin disease from an image.")
    parser.add_argument("image", help="Path to the image file to classify")
    parser.add_argument(
        "--model",
        default="skin_disease_final.keras",
        help="Path to the trained .keras model file (default: skin_disease_final.keras)",
    )
    parser.add_argument(
        "--classes",
        default="class_indices.json",
        help="Path to the class_indices.json file (default: class_indices.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of top predictions to show (default: 3)",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = load_trained_model(args.model)

    print(f"Loading class mapping from: {args.classes}")
    idx_to_class = load_class_mapping(args.classes)

    print(f"Classifying: {args.image}\n")
    results = predict_image(args.image, model, idx_to_class, top_k=args.top)

    print("Predictions:")
    print("-" * 40)
    for rank, (label, confidence) in enumerate(results, start=1):
        print(f"{rank}. {label:25s} {confidence * 100:6.2f}%")
    print("-" * 40)

    best_label, best_confidence = results[0]
    print(f"\nMost likely: {best_label} ({best_confidence * 100:.1f}% confidence)")

    if best_confidence < 0.4:
        print(
            "\nNote: confidence is low — this prediction is uncertain. "
            "This model is for educational/project purposes only and is not a "
            "substitute for professional medical diagnosis."
        )


if __name__ == "__main__":
    main()
