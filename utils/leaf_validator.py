"""Metadata-driven lazy binary gate; never falls back to disease confidence."""
import json
import math
from threading import Lock
import numpy as np
from utils.model_loader import PROJECT_ROOT, _resolve_project_path
METADATA_PATH = PROJECT_ROOT / "model/lemon_leaf_validator_metadata.json"
_model = None
_model_key = None
_lock = Lock()

def get_validator_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Validator metadata missing: {METADATA_PATH}")
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    size = meta.get("input_size")
    if not isinstance(size, list) or len(size)!=2 or any(type(v) is not int or v<=0 for v in size):
        raise ValueError("Invalid validator input_size.")
    if meta.get("architecture") != "MobileNetV3Small":
        raise ValueError("Unsupported validator architecture.")
    expected = f"RGB float32 bilinear resize to {size[0]}x{size[1]}; input range [0,255]; internal MobileNetV3 Rescaling to [-1,1]; no external preprocess_input"
    if meta.get("preprocessing") != expected:
        raise ValueError("Missing or unsupported validator preprocessing metadata.")
    threshold = meta.get("chosen_acceptance_threshold")
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<threshold<1:
        raise ValueError("Validator threshold must be finite and between zero and one.")
    if meta.get("labels") != {"NOT_LEMON_LEAF":0,"LEMON_LEAF":1}:
        raise ValueError("Validator label mapping mismatch.")
    if meta.get("acceptance_rule") != "lemon_probability >= threshold" or not meta.get("model_path"):
        raise ValueError("Validator model path or acceptance rule missing/inconsistent.")
    return meta

def get_validator_model(meta=None):
    global _model, _model_key
    meta = get_validator_metadata() if meta is None else meta
    path = _resolve_project_path(meta["model_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Validator model missing: {path}")
    key = (str(path.resolve()),tuple(meta["input_size"]),meta["preprocessing"])
    with _lock:
        if _model is None or _model_key != key:
            import tensorflow as tf
            model = tf.keras.models.load_model(path,compile=False)
            if tuple(model.input_shape[1:]) != (*meta["input_size"],3) or tuple(model.output_shape[1:]) != (1,):
                raise ValueError("Validator model input/output shape does not match metadata.")
            backbone = next((layer for layer in model.layers if isinstance(layer,tf.keras.Model)),None)
            rescalers = [] if backbone is None else [l for l in backbone.layers if isinstance(l,tf.keras.layers.Rescaling)]
            if not any(np.isclose(l.scale,1/127.5) and np.isclose(l.offset,-1) for l in rescalers):
                raise ValueError("Validator internal [-1,1] preprocessing is missing.")
            _model, _model_key = model,key
    return _model

def result_from_score(score, meta):
    score=float(score)
    if not math.isfinite(score) or not 0<=score<=1:
        raise ValueError("Validator returned an invalid probability.")
    threshold=float(meta["chosen_acceptance_threshold"])
    accepted=bool(score>=threshold)
    return {"lemon_score":score,"acceptance_threshold":threshold,"accepted_as_lemon_leaf":accepted,
            "validator_label":"LEMON_LEAF" if accepted else "NOT_LEMON_LEAF",
            "validator_model_name":meta["architecture"]}

def validate_rgb(rgb):
    from utils.preprocessing import prepare_validator_rgb
    meta=get_validator_metadata()
    prepared=prepare_validator_rgb(rgb,meta)
    values=np.asarray(get_validator_model(meta)(prepared.tensor,training=False))
    if values.shape!=(1,1):
        raise ValueError("Unexpected validator probability shape.")
    return result_from_score(values[0,0],meta)
