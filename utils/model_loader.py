"""Load model artifacts and metadata shared by the application."""

import json
import os
from pathlib import Path
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = {
    "architecture": "resnet50",
    "model_path": "model/resnet_best_model.keras",
    "class_indices_path": "training/class_indices.json",
    "image_size": [224, 224],
}
METADATA_PATH = Path(
    os.environ.get("LEMOSCAN_MODEL_METADATA", PROJECT_ROOT / "model" / "model_metadata.json")
)

_model = None
_model_path = None
_model_lock = Lock()
_metadata = None


def _resolve_project_path(path_value):
    """Resolve a metadata path relative to the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_model_metadata():
    """Return model metadata, falling back to the original ResNet50 artifact."""
    global _metadata

    if _metadata is None:
        if METADATA_PATH.exists():
            with METADATA_PATH.open("r", encoding="utf-8") as file:
                loaded_metadata = json.load(file)
            _metadata = {**DEFAULT_METADATA, **loaded_metadata}
        else:
            _metadata = DEFAULT_METADATA.copy()

    return _metadata


def get_model_path():
    """Return the active model path from metadata."""
    return _resolve_project_path(get_model_metadata()["model_path"])


def get_class_indices_path():
    """Return the active class-index mapping path from metadata."""
    return _resolve_project_path(get_model_metadata()["class_indices_path"])


def get_image_size():
    """Return the active model image size as a width-height tuple."""
    image_size = get_model_metadata().get("image_size", DEFAULT_METADATA["image_size"])
    return tuple(int(value) for value in image_size)


def get_preprocess_function():
    """Return the preprocessing function expected by the active architecture."""
    architecture = get_model_metadata().get("architecture", "resnet50").lower()

    if architecture == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif architecture == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")

    return preprocess_input


def get_model():
    """Load the active trained model lazily, then reuse the same instance."""
    global _model, _model_path

    model_path = get_model_path()
    if _model is None or _model_path != model_path:
        with _model_lock:
            if _model is None or _model_path != model_path:
                from tensorflow.keras.models import load_model

                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                _model = load_model(model_path, compile=False)
                _model_path = model_path

    return _model
