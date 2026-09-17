"""Functionality smoke checks only: one deterministic TRAIN sample per class.
Run from any directory: python scripts/test_inception_inference.py
No accuracy computation, test-set inference, fitting, or checkpoint selection.
"""
import copy
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.keras.utils.set_random_seed(42)
from utils import model_loader, prediction, xai
from utils.preprocessing import prepare_image, ImageValidationError

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    out = ROOT / "reports/inference_integration"
    out.mkdir(parents=True, exist_ok=True)
    protected_paths = list((ROOT / "model").glob("*")) + list((ROOT / "reports/inceptionv3_test").glob("*"))
    protected_paths += [ROOT / "reports/final_split/final_split_manifest.csv", ROOT / "reports/dataset_audit/manual_review_decisions.csv"]
    protected = {str(p): digest(p) for p in protected_paths if p.is_file()}
    dataset_before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (ROOT / "dataset").rglob("*") if p.is_file()}
    training = json.loads((ROOT / "reports/inceptionv3_training/training_summary.json").read_text())
    assert digest(model_loader.get_model_path()) == training["final_model_sha256"]
    manifest = pd.read_csv(ROOT / "reports/final_split/final_split_manifest.csv")
    samples = manifest.loc[manifest.new_split.eq("train")].sort_values("filepath").groupby("class", sort=True).head(1)
    assert len(samples) == 6
    mapping = model_loader.get_class_mapping()
    outcomes = []
    for number, row in enumerate(samples.itertuples()):
        seen = {}
        real_predict, real_cam = prediction.predict_prepared, xai.generate_gradcam
        def capture_predict(prepared):
            seen["prepared"] = prepared
            return real_predict(prepared)
        def capture_cam(prepared, target, output_path):
            assert prepared is seen["prepared"], "Prediction and Grad-CAM must reuse the same object."
            seen["target"] = target
            return real_cam(prepared, target, output_path)
        with patch.object(prediction, "predict_prepared", side_effect=capture_predict) as classify, patch.object(xai, "generate_gradcam", side_effect=capture_cam):
            result = prediction.analyze_image(ROOT / row.filepath, out / f"smoke_{number}.png")
            assert classify.call_count == 1
        probabilities = np.asarray(result["probabilities"])
        assert probabilities.shape == (6,) and np.isfinite(probabilities).all()
        assert np.all((probabilities >= 0) & (probabilities <= 1)) and np.isclose(probabilities.sum(), 1, atol=1e-5)
        index = result["predicted_class_index"]
        assert mapping[result["predicted_class"]] == index == int(probabilities.argmax())
        assert len(result["top_3"]) == 3
        assert [p["probability"] for p in result["top_3"]] == sorted([p["probability"] for p in result["top_3"]], reverse=True)
        for item in result["top_3"]:
            assert mapping[item["class"]] == item["class_index"]
            assert item["probability"] == probabilities[item["class_index"]]
        cam = result["gradcam"]
        assert cam["status"] == "available", cam
        assert cam["target_class_index"] == index == seen["target"]
        assert np.isclose(cam["target_probability"], result["confidence"], atol=1e-6)
        assert cam["heatmap"].shape == (8, 8) and np.isfinite(cam["heatmap"]).all()
        assert cam["heatmap"].min() >= 0 and cam["heatmap"].max() <= 1
        assert Path(cam["output_path"]).is_file()
        outcomes.append({"source_class": row._asdict()["class"] if "class" in row._asdict() else samples.iloc[number]["class"],
                         "filepath": row.filepath, "split": "train", "passed": True,
                         "predicted_class": result["predicted_class"], "predicted_class_index": index,
                         "gradcam_target_class_index": cam["target_class_index"], "heatmap_shape": cam["heatmap_shape"],
                         "all_zero_heatmap": cam["all_zero"], "shared_prepared_object": True})
        print(f"PASS training sample {number+1}/6: structure, shared input, explicit target, Grad-CAM", flush=True)
    try:
        prepare_image(io.BytesIO(b"not a decodable image"))
    except ImageValidationError:
        corrupt_rejected = True
    else:
        raise AssertionError("Corrupt image was accepted")
    # Existing sample converted in memory for supported-format and RGB checks.
    with Image.open(ROOT / samples.iloc[0].filepath) as original:
        prepared = prepare_image(original.convert("L"))
        assert prepared.rgb.mode == "RGB" and tuple(prepared.tensor.shape) == (1, 299, 299, 3)
        unsupported = io.BytesIO()
        original.save(unsupported, format="BMP")
        unsupported.seek(0)
        try:
            prepare_image(unsupported)
        except ImageValidationError:
            pass
        else:
            raise AssertionError("Unsupported format accepted")
    retained = copy.deepcopy(result)
    retained.pop("gradcam")
    retained["warnings"].append("Existing warning must survive.")
    with patch.object(prediction, "predict_prepared", return_value=copy.deepcopy(retained)), patch.object(xai, "generate_gradcam", side_effect=RuntimeError("deliberate smoke-test failure")):
        failed_cam = prediction.analyze_image(ROOT / samples.iloc[0].filepath)
    assert failed_cam["gradcam"]["status"] == "unavailable"
    assert failed_cam["probabilities"] == retained["probabilities"]
    assert failed_cam["warnings"][:-1] == retained["warnings"]
    # Confirm malformed metadata/mapping fail clearly without loading another model.
    with patch.object(model_loader, "METADATA_PATH", out / "nonexistent_metadata.json"):
        try:
            model_loader.get_model_metadata()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing metadata accepted")
    with patch.object(model_loader, "get_class_indices_path", return_value=out / "nonexistent_mapping.json"):
        try:
            model_loader.get_class_mapping()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing mapping accepted")
    with patch.object(model_loader, "get_class_indices_path", return_value=ROOT / "model/model_metadata.json"):
        try:
            model_loader.get_class_mapping()
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched mapping accepted")
    assert all(digest(Path(p)) == value for p, value in protected.items())
    assert dataset_before == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (ROOT / "dataset").rglob("*") if p.is_file()}
    summary = {
        "active_model": "model/inceptionv3_best_model.keras", "model_sha256": digest(model_loader.get_model_path()),
        "architecture": "InceptionV3", "input_size": [299,299], "channels": 3,
        "preprocessing": "Pillow RGB decode; TensorFlow float32 bilinear resize (antialias=False); inception_v3.preprocess_input",
        "decoder_note": "Pillow JPEG decoding can differ slightly from the TensorFlow decoder used for training; prediction and Grad-CAM reuse exactly the same decoded image and tensor.",
        "class_mapping": mapping, "gradcam_layer": "inception_v3/mixed10", "gradcam_shape": [8,8],
        "gradcam_target": "Explicit predicted_class_index; no independent argmax",
        "smoke_test_results": outcomes, "all_smoke_tests_passed": True,
        "corrupt_image_rejected": corrupt_rejected, "unsupported_format_rejected": True, "grayscale_converted_to_rgb": True,
        "gradcam_failure_preserves_prediction_and_warnings": True, "missing_metadata_and_mapping_rejected": True,
        "mismatched_mapping_rejected": True, "protected_artifacts_unchanged": True, "dataset_stats_unchanged": True,
        "technical_validation_behavior": "Decode JPEG/PNG, force full pixel load, positive dimensions and <=40 million pixels; no semantic leaf rejection.",
        "legacy_uncertainty_status": "Original 70% confidence / 15 percentage-point margin isolated as provisional, uncalibrated flag; actual class and probabilities preserved.",
        "audit_findings": ["ResNet50/224 fallback removed; metadata now mandatory.",
                           "Prediction used Pillow default resize while XAI used Lanczos; replaced by a single shared path.",
                           "Blocking HSV/color/edge heuristic removed.", "Old uncertainty replaced class label; now a separate flag.",
                           "Grad-CAM independently chose argmax; now explicit prediction target.",
                           "Flask replaced warnings on XAI error; Streamlit lost prediction. Shared analysis now preserves both."],
        "application_changes": "Inference call-site adapters only; no frontend layout or design changes.",
        "remaining_limitations": ["No semantic lemon-leaf validator yet.", "Softmax probabilities and legacy uncertainty are not calibrated.",
                                  "Grad-CAM is an attribution visualization, not proof of biological diagnosis.",
                                  "410 unresolved C/D pairs cross dataset splits; complete biological-source independence remains unproven.",
                                  "Model metadata test_evaluated=false is historical training provenance; final test reports remain authoritative and unchanged."],
        "test_split_evaluated": False, "accuracy_computed": False, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    (out / "integration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("ALL INFERENCE SMOKE CHECKS PASSED", flush=True)

if __name__ == "__main__":
    main()

