import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "SkinDisease" / "train"
TEST_DIR = BASE_DIR / "dataset" / "SkinDisease" / "test"
MODEL_PATH = BASE_DIR / "skin_disease_final.keras"
CLASS_MAP_PATH = BASE_DIR / "class_indices.json"

SEED = 42
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 22
EPOCHS_HEAD = 8
EPOCHS_FINE_TUNE = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


class_counts = {cls.name: len(list(cls.iterdir())) for cls in sorted(TRAIN_DIR.iterdir()) if cls.is_dir()}
class_weights = {idx: max(class_counts.values()) / count for idx, count in enumerate([class_counts[name] for name in sorted(class_counts)])}

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_indices = json.load(f)
class_names = [class_name for class_name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest",
    validation_split=0.15,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    classes=class_names,
    seed=SEED,
)

valid_generator = valid_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    classes=class_names,
    seed=SEED,
)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_callbacks = [
    ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

fine_tune_history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_HEAD + EPOCHS_FINE_TUNE,
    initial_epoch=len(history.history["loss"]),
    class_weight=class_weights,
    callbacks=fine_tune_callbacks,
    verbose=1,
)

print("\nTraining complete.")
print(f"Saved model to: {MODEL_PATH}")
