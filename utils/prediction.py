import json

import numpy as np
from PIL import Image

from utils.model_loader import (
    get_class_indices_path,
    get_image_size,
    get_model,
    get_preprocess_function,
)


INVALID_IMAGE_CLASS = "Invalid image / Not a lemon leaf"
MIN_LEAF_LIKE_RATIO = 0.12
MIN_GREEN_RATIO = 0.06
MAX_LEAF_EDGE_DENSITY = 0.14
MIN_CONFIDENCE = 70.0
MIN_CONFIDENCE_MARGIN = 15.0

_index_to_class = None
_index_source = None


def get_index_to_class():
    """Load and reverse the active class-to-index mapping."""
    global _index_to_class, _index_source

    class_indices_path = get_class_indices_path()
    if _index_to_class is None or _index_source != class_indices_path:
        if not class_indices_path.exists():
            raise FileNotFoundError(
                f"Class index file not found: {class_indices_path}"
            )

        with class_indices_path.open("r", encoding="utf-8") as file:
            class_indices = json.load(file)

        _index_to_class = {
            int(index): class_name for class_name, index in class_indices.items()
        }
        _index_source = class_indices_path

    return _index_to_class


def preprocess_image(image_path):
    """Resize an image and apply preprocessing for the active model."""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(get_image_size())
        image_array = np.asarray(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)
    return get_preprocess_function()(image_array)


def _rgb_to_hue_saturation_value(image_array):
    """Return HSV components for an RGB image array in the 0-255 range."""
    rgb = image_array / 255.0
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    max_channel = np.max(rgb, axis=-1)
    min_channel = np.min(rgb, axis=-1)
    chroma = max_channel - min_channel

    hue = np.zeros_like(max_channel)
    non_gray = chroma > 1e-6

    red_is_max = (max_channel == red) & non_gray
    green_is_max = (max_channel == green) & non_gray
    blue_is_max = (max_channel == blue) & non_gray

    hue[red_is_max] = ((green[red_is_max] - blue[red_is_max]) / chroma[red_is_max]) % 6
    hue[green_is_max] = ((blue[green_is_max] - red[green_is_max]) / chroma[green_is_max]) + 2
    hue[blue_is_max] = ((red[blue_is_max] - green[blue_is_max]) / chroma[blue_is_max]) + 4
    hue *= 60.0

    saturation = np.zeros_like(max_channel)
    saturation[max_channel > 0] = chroma[max_channel > 0] / max_channel[max_channel > 0]

    return hue, saturation, max_channel


def _mask_edge_density(mask):
    """Measure how fragmented the green leaf-like mask is."""
    if not np.any(mask):
        return 1.0

    padded = np.pad(mask, 1, constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = mask & ~eroded
    return float(boundary.sum() / mask.sum())


def assess_leaf_likeness(image_path):
    """Estimate whether an image is close enough to a lemon leaf for classification."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if min(image.size) < 80:
            return False, "The image is too small for reliable leaf analysis."

        image.thumbnail((256, 256))
        image_array = np.asarray(image, dtype=np.float32)

    hue, saturation, value = _rgb_to_hue_saturation_value(image_array)

    green_pixels = (
        (hue >= 55)
        & (hue <= 170)
        & (saturation >= 0.18)
        & (value >= 0.15)
    )
    yellow_green_pixels = (
        (hue >= 35)
        & (hue < 55)
        & (saturation >= 0.22)
        & (value >= 0.18)
    )
    leaf_like_pixels = green_pixels | yellow_green_pixels

    green_ratio = float(np.mean(green_pixels))
    leaf_like_ratio = float(np.mean(leaf_like_pixels))
    edge_density = _mask_edge_density(leaf_like_pixels)

    if leaf_like_ratio < MIN_LEAF_LIKE_RATIO or green_ratio < MIN_GREEN_RATIO:
        return (
            False,
            "This image does not look like a lemon leaf. Please upload a clear lemon leaf photo.",
        )

    if edge_density > MAX_LEAF_EDGE_DENSITY:
        return (
            False,
            "This leaf shape looks too divided or fern-like for lemon leaf analysis. "
            "Please upload one clear lemon leaf.",
        )

    return True, None


def predict_disease(image_path):
    """Return the predicted class, confidence, top three results, and warning."""
    is_leaf_like, validation_warning = assess_leaf_likeness(image_path)
    if not is_leaf_like:
        return INVALID_IMAGE_CLASS, 0.0, [], validation_warning

    processed_image = preprocess_image(image_path)
    probabilities = get_model().predict(processed_image, verbose=0)[0]
    index_to_class = get_index_to_class()

    if len(probabilities) != len(index_to_class):
        raise ValueError(
            "The number of model outputs does not match the active class-index file."
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
    margin = confidence - top_predictions[1]["confidence"]

    if confidence < MIN_CONFIDENCE or margin < MIN_CONFIDENCE_MARGIN:
        return (
            "Uncertain lemon leaf condition",
            confidence,
            top_predictions,
            "The model is not confident enough to make a reliable disease call. "
            "Please upload a clearer lemon leaf image or consult an expert.",
        )

    return predicted_class, confidence, top_predictions, None
