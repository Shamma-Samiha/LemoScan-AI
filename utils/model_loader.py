"""Strict metadata-driven lazy loading of the frozen classifier."""
import json
import os
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = Path(os.environ.get("LEMOSCAN_MODEL_METADATA", PROJECT_ROOT / "model/model_metadata.json"))
EXPECTED_CLASSES = ["Algal leaf spot", "Black spot", "Citrus canker", "Citrus pest", "Greening", "Healthy leaf"]
_model = None
_model_lock = Lock()

def _resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def get_model_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    expected = {"architecture": "inceptionv3", "image_size": [299, 299], "input_size": [299, 299],
                "channels": 3, "number_of_classes": 6, "class_names": EXPECTED_CLASSES,
                "preprocessing": "inception_v3.preprocess_input", "preprocessing_location": "external_before_model"}
    for key, value in expected.items():
        actual = metadata.get(key)
        if key == "architecture" and isinstance(actual, str):
            actual = actual.lower()
        if actual != value:
            raise ValueError(f"Inconsistent metadata {key}: expected {value!r}, got {actual!r}")
    for key, relative in [("model_path", "model/inceptionv3_best_model.keras"),
                          ("class_indices_path", "training/inceptionv3_class_indices.json")]:
        if not metadata.get(key) or _resolve_project_path(metadata[key]).resolve() != (PROJECT_ROOT / relative).resolve():
            raise ValueError(f"Metadata {key} must point to frozen artifact {relative}")
    return metadata

def get_model_path():
    return _resolve_project_path(get_model_metadata()["model_path"])

def get_class_indices_path():
    return _resolve_project_path(get_model_metadata()["class_indices_path"])

def get_class_mapping():
    path = get_class_indices_path()
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {path}")
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping != dict(zip(EXPECTED_CLASSES, range(6))) or any(type(v) is not int for v in mapping.values()):
        raise ValueError("Class mapping does not match the frozen six-class output order.")
    return mapping

def get_image_size():
    return tuple(get_model_metadata()["image_size"])

def get_preprocess_function():
    get_model_metadata()
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    return preprocess_input

def get_model():
    global _model
    path = get_model_path()
    mapping = get_class_mapping()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    with _model_lock:
        if _model is None:
            from tensorflow.keras.models import load_model
            candidate = load_model(path, compile=False)
            if tuple(candidate.input_shape[1:]) != (299, 299, 3):
                raise ValueError(f"Unexpected model input shape: {candidate.input_shape}")
            if tuple(candidate.output_shape[1:]) != (len(mapping),):
                raise ValueError("Model output dimension does not match the class mapping.")
            _model = candidate
    return _model
