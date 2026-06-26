from pathlib import Path

import numpy as np
from PIL import Image


# Class names must match your training class order
CLASS_NAMES = [
    "Algae leaf spot",
    "Black Spot",
    "Citrus Canker",
    "Citrus Pest",
    "Greening",
    "Healthy Leaf"
]

# Image size used during model training
IMG_SIZE = (224, 224)
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "resnet_best_model.keras"

_model = None


def get_model():
    """Load the TensorFlow model once, only when prediction is requested."""
    global _model

    if _model is None:
        import tensorflow as tf

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        _model = tf.keras.models.load_model(MODEL_PATH)

    return _model


def preprocess_image(image_path):
    """
    Preprocess uploaded image for ResNet model prediction.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_disease(image_path):
    """
    Predict lemon leaf disease class and confidence.
    """
    processed_image = preprocess_image(image_path)
    predictions = get_model().predict(processed_image)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]) * 100)

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence
