"""Train an InceptionV3 model for LemoScan AI.

This script is designed for the real fix to closed-set overconfidence: add a
negative class such as `Not lemon leaf` to dataset/train, dataset/val, and
dataset/test before training. Without that class, a neural classifier will still
be forced to choose one of the disease labels for humans, ferns, and other
out-of-distribution images.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


SEED = 42
IMG_SIZE = (299, 299)
BATCH_SIZE = 24
NEGATIVE_CLASS = "Not lemon leaf"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
TRAINING_DIR = PROJECT_ROOT / "training"
MODEL_DIR = PROJECT_ROOT / "model"
CLASS_INDICES_PATH = TRAINING_DIR / "inceptionv3_class_indices.json"
BEST_MODEL_PATH = MODEL_DIR / "inceptionv3_best_model.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
HISTORY_PATH = TRAINING_DIR / "inceptionv3_history.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train LemoScan InceptionV3 model.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument(
        "--allow-closed-set",
        action="store_true",
        help="Allow training without a Not lemon leaf class. This will not fix invalid-image overconfidence.",
    )
    return parser.parse_args()


def count_images(folder):
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_classes():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Missing training folder: {TRAIN_DIR}")

    return sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())


def validate_dataset(class_names, require_negative_class):
    if require_negative_class and NEGATIVE_CLASS not in class_names:
        raise ValueError(
            f"Missing negative class `{NEGATIVE_CLASS}` in {TRAIN_DIR}. "
            "Create matching folders under dataset/train, dataset/val, and dataset/test, "
            "then add non-lemon images such as humans, ferns, other leaves, soil, tools, and blurry photos."
        )

    missing = []
    empty = []
    counts = {}
    for split_name, split_dir in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing dataset split folder: {split_dir}")

        counts[split_name] = {}
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing.append(f"{split_name}/{class_name}")
                continue

            image_count = count_images(class_dir)
            counts[split_name][class_name] = image_count
            if image_count == 0:
                empty.append(f"{split_name}/{class_name}")

    if missing:
        raise ValueError("Missing class folders: " + ", ".join(missing))
    if empty:
        raise ValueError("Empty class folders: " + ", ".join(empty))

    return counts


def save_class_indices(class_names):
    class_indices = {class_name: index for index, class_name in enumerate(class_names)}
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    with CLASS_INDICES_PATH.open("w", encoding="utf-8") as file:
        json.dump(class_indices, file, indent=4)
    return class_indices


def build_datasets(class_names, batch_size):
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.15),
        ],
        name="training_augmentation",
    )

    def prepare_train(images, labels):
        images = augmentation(images, training=True)
        return preprocess_input(images), labels

    def prepare_eval(images, labels):
        return preprocess_input(images), labels

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.map(prepare_train, num_parallel_calls=autotune).prefetch(autotune),
        val_ds.map(prepare_eval, num_parallel_calls=autotune).prefetch(autotune),
        test_ds.map(prepare_eval, num_parallel_calls=autotune).prefetch(autotune),
    )


def build_model(class_count):
    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dropout(0.45, name="dropout")(x)
    outputs = layers.Dense(
        class_count,
        activation="softmax",
        kernel_regularizer=regularizers.l2(1e-4),
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="inceptionv3_lemoscan_classifier")
    return model, base_model


def class_weights_from_counts(train_counts, class_names):
    total = sum(train_counts[class_name] for class_name in class_names)
    return {
        index: total / (len(class_names) * train_counts[class_name])
        for index, class_name in enumerate(class_names)
    }


def combine_histories(*histories):
    combined = {}
    for history in histories:
        for key, values in history.history.items():
            combined.setdefault(key, []).extend(float(value) for value in values)
    return combined


def relative_path(path):
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def save_metadata(class_names, counts, test_accuracy):
    metadata = {
        "architecture": "inceptionv3",
        "model_path": relative_path(BEST_MODEL_PATH),
        "class_indices_path": relative_path(CLASS_INDICES_PATH),
        "image_size": list(IMG_SIZE),
        "classes": class_names,
        "negative_class": NEGATIVE_CLASS if NEGATIVE_CLASS in class_names else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_accuracy": float(test_accuracy),
        "dataset_counts": counts,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def main():
    args = parse_args()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    class_names = discover_classes()
    counts = validate_dataset(class_names, require_negative_class=not args.allow_closed_set)
    save_class_indices(class_names)

    print("Classes:", class_names)
    print("Dataset counts:", json.dumps(counts, indent=2))
    if NEGATIVE_CLASS not in class_names:
        print("WARNING: training a closed-set model without a negative class.")

    train_ds, val_ds, test_ds = build_datasets(class_names, args.batch_size)
    model, base_model = build_model(len(class_names))
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_stage_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage1_epochs,
        class_weight=class_weights_from_counts(counts["train"], class_names),
        callbacks=callbacks,
    )

    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - args.fine_tune_layers)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    total_epochs = len(history_stage_1.history["loss"]) + args.stage2_epochs
    history_stage_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(history_stage_1.history["loss"]),
        epochs=total_epochs,
        class_weight=class_weights_from_counts(counts["train"], class_names),
        callbacks=callbacks,
    )

    best_model = keras.models.load_model(BEST_MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(combine_histories(history_stage_1, history_stage_2), file, indent=4)
    save_metadata(class_names, counts, test_accuracy)

    print(f"Saved model: {BEST_MODEL_PATH}")
    print(f"Saved class indices: {CLASS_INDICES_PATH}")
    print(f"Saved active model metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
