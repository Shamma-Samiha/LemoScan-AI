"""Structured inference with isolated, uncalibrated legacy uncertainty."""
import numpy as np
from utils.model_loader import get_class_mapping, get_model
from utils.preprocessing import prepare_image

INVALID_IMAGE_CLASS = "Invalid image / Not a lemon leaf"  # Legacy import only; never a model class.
MIN_CONFIDENCE = 70.0
MIN_CONFIDENCE_MARGIN = 15.0
SEMANTIC_WARNING = "Lemon-leaf identity is not verified; this six-class model cannot reject non-leaf images."

def get_index_to_class():
    return {index: name for name, index in get_class_mapping().items()}

def preprocess_image(source):
    """Compatibility entry point using the sole preprocessing implementation."""
    return prepare_image(source).tensor

def legacy_uncertainty(probabilities):
    """Original 70% / 15 percentage-point rule; provisional, not calibrated."""
    ordered = np.sort(probabilities)[::-1]
    return bool(100 * ordered[0] < MIN_CONFIDENCE or 100 * (ordered[0] - ordered[1]) < MIN_CONFIDENCE_MARGIN)

def predict_prepared(prepared):
    probabilities = np.asarray(get_model()(prepared.tensor, training=False))[0]
    mapping = get_index_to_class()
    if probabilities.shape != (6,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1, atol=1e-5):
        raise ValueError("Model returned an invalid six-class probability vector.")
    indices = np.argsort(-probabilities, kind="stable")[:3]
    top3 = [{"class": mapping[int(i)], "class_index": int(i), "probability": float(probabilities[i])} for i in indices]
    uncertain = legacy_uncertainty(probabilities)
    warnings = [SEMANTIC_WARNING]
    if uncertain:
        warnings.append("Legacy/provisional uncertainty rule triggered (70% confidence / 15-point margin); not statistically calibrated.")
    return {"predicted_class": top3[0]["class"], "predicted_class_index": top3[0]["class_index"],
            "confidence": top3[0]["probability"], "top_3": top3, "probabilities": probabilities.tolist(),
            "legacy_uncertain": uncertain, "warnings": warnings}

def predict_disease(source):
    """One classification call; confidence and probabilities are on the 0?1 scale."""
    return predict_prepared(prepare_image(source))

def analyze_image(source, gradcam_output_path=None):
    """Decode once, classify, explain that exact input and supplied class index."""
    prepared = prepare_image(source)
    result = predict_prepared(prepared)
    try:
        from utils.xai import generate_gradcam
        result["gradcam"] = generate_gradcam(prepared, result["predicted_class_index"], gradcam_output_path)
    except Exception as exc:
        result["gradcam"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        result["warnings"].append("Prediction completed, but explanation generation failed.")
    return result

def legacy_display_values(result):
    """Adapt probability units for existing UI fields without changing labels."""
    top = [{"class": item["class"], "confidence": item["probability"] * 100} for item in result["top_3"]]
    return result["predicted_class"], result["confidence"] * 100, top, " ".join(result["warnings"]) or None
