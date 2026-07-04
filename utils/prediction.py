import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# Build paths relative to the project root so prediction works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "resnet_best_model.keras"
CLASS_INDICES_PATH = PROJECT_ROOT / "training" / "class_indices.json"
IMAGE_SIZE = (224, 224)

_model = None
_index_to_class = None


def get_model():
    """Load the trained model once, when the first prediction is requested."""
    global _model

    if _model is None:
        from tensorflow.keras.models import load_model

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        _model = load_model(MODEL_PATH)

    return _model


def get_index_to_class():
    """Load and reverse the saved class-to-index mapping."""
    global _index_to_class

    if _index_to_class is None:
        if not CLASS_INDICES_PATH.exists():
            raise FileNotFoundError(
                f"Class index file not found: {CLASS_INDICES_PATH}"
            )

        with CLASS_INDICES_PATH.open("r", encoding="utf-8") as file:
            class_indices = json.load(file)

        _index_to_class = {
            int(index): class_name for class_name, index in class_indices.items()
        }

    return _index_to_class


def preprocess_image(image_path):
    """Resize an image and apply the preprocessing expected by ResNet50."""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)


def predict_disease(image_path):
    """Return the predicted class, confidence, top three results, and warning."""
    processed_image = preprocess_image(image_path)
    probabilities = get_model().predict(processed_image, verbose=0)[0]
    index_to_class = get_index_to_class()

    if len(probabilities) != len(index_to_class):
        raise ValueError(
            "The number of model outputs does not match training/class_indices.json."
        )

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {
            "class": index_to_class[int(index)],
            "confidence": round(float(probabilities[index]) * 100, 2),
        }
        for index in top_indices
    ]

    predicted_class = top_predictions[0]["class"]
    confidence = top_predictions[0]["confidence"]
    warning = (
        "Low confidence prediction. Please upload a clearer image or consult an expert."
        if confidence < 70
        else None
    )

    return predicted_class, confidence, top_predictions, warning

