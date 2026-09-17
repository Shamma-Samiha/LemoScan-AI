"""Structured inference with isolated, uncalibrated legacy uncertainty."""
import numpy as np
from utils.model_loader import get_class_mapping, get_model
from utils.preprocessing import prepare_image, decode_rgb, prepare_disease_rgb, ImageValidationError
from utils.leaf_validator import validate_rgb

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
    """Decode once; fail closed at the validator; preserve successful stages."""
    result = {"status": None, "validator": None, "predicted_class": None,
              "predicted_class_index": None, "confidence": None, "top_3": [],
              "probabilities": [], "legacy_uncertain": None, "warnings": [],
              "gradcam": {"status": "not_run"}, "message": None, "error": None}
    try:
        rgb = decode_rgb(source)
    except (ImageValidationError, ValueError, TypeError) as exc:
        result.update(status="technical_error",message=str(exc),error={"stage":"technical_validation","detail":str(exc)})
        return result
    try:
        result["validator"] = validate_rgb(rgb)
    except Exception as exc:
        result.update(status="analysis_error",message="Lemon-leaf validation failed; disease analysis was not run.",
                      error={"stage":"validator","detail":f"{type(exc).__name__}: {exc}"})
        return result
    if not result["validator"]["accepted_as_lemon_leaf"]:
        result.update(status="not_lemon_leaf",
                      message="The uploaded image was not recognized as a lemon leaf with sufficient confidence.")
        return result
    result["warnings"].append("The input validator passed; broader real-world lemon-species recognition is not guaranteed.")
    try:
        prepared = prepare_disease_rgb(rgb)
        disease = predict_prepared(prepared)
        # The disease-only helper retains its warning for callers bypassing the gate.
        disease["warnings"] = [w for w in disease["warnings"] if w != SEMANTIC_WARNING]
        warnings = result["warnings"] + disease["warnings"]
        result.update(disease)
        result["warnings"] = warnings
        result["status"] = "lemon_leaf_analysis"
    except Exception as exc:
        result.update(status="analysis_error",message="The validator accepted the image, but disease classification failed.",
                      error={"stage":"disease_classifier","detail":f"{type(exc).__name__}: {exc}"})
        return result
    try:
        from utils.xai import generate_gradcam
        result["gradcam"] = generate_gradcam(prepared,result["predicted_class_index"],gradcam_output_path)
    except Exception as exc:
        result["gradcam"] = {"status":"unavailable","error":f"{type(exc).__name__}: {exc}"}
        result["warnings"].append("Prediction completed, but explanation generation failed.")
    return result

def legacy_display_values(result):
    """Adapt probability units for existing UI fields without changing labels."""
    if result.get("status") not in (None, "lemon_leaf_analysis"):
        raise ValueError(result.get("message") or "Image analysis did not produce a disease prediction.")
    top = [{"class": item["class"], "confidence": item["probability"] * 100} for item in result["top_3"]]
    return result["predicted_class"], result["confidence"] * 100, top, " ".join(result["warnings"]) or None
