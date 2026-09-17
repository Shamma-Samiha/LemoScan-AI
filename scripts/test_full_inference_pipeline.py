"""Full inference functionality tests; use training examples, never test accuracy."""
import copy,io,json,hashlib,sys
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
from utils import prediction,leaf_validator,xai,model_loader

def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    out=ROOT/"reports/full_pipeline"
    out.mkdir(parents=True,exist_ok=True)
    paths=list((ROOT/"model").glob("*"))
    for folder in ["reports/inceptionv3_test","reports/leaf_validator"]:
        paths+=list((ROOT/folder).glob("*"))
    paths += [ROOT/"reports/final_split/final_split_manifest.csv",ROOT/"reports/dataset_audit/manual_review_decisions.csv"]
    protected={str(p):sha(p) for p in paths if p.is_file()}
    stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in (ROOT/"dataset").rglob("*") if p.is_file()}
    meta=leaf_validator.get_validator_metadata()
    threshold=meta["chosen_acceptance_threshold"]
    assert threshold==0.397647500038147
    rows=[]
    def reject(source,case):
        with patch.object(prediction,"get_model",side_effect=AssertionError("Disease model must not be loaded")), patch.object(prediction,"predict_prepared",side_effect=AssertionError("Disease inference must not run")) as disease, patch.object(xai,"generate_gradcam",side_effect=AssertionError("Grad-CAM must not run")) as cam:
            result=prediction.analyze_image(source)
        assert result["status"]=="not_lemon_leaf",result
        assert not disease.called and not cam.called
        assert result["predicted_class"] is None and result["top_3"]==[] and result["probabilities"]==[]
        assert result["gradcam"]["status"]=="not_run"
        rows.append({"case":case,"passed":True,"status":result["status"],"validator":result["validator"],
                     "disease_calls":disease.call_count,"gradcam_calls":cam.call_count})
        return result
    green=reject(Image.new("RGB",(224,224),(0,255,0)),"solid_green")
    assert model_loader._model is None, "Rejected green must not load InceptionV3."
    manifest=pd.read_csv(ROOT/"reports/leaf_validator/validator_dataset_manifest.csv")
    for source in ["beans","caltech101"]:
        samples=manifest[(manifest.source==source)&(manifest.split=="train")].sort_values("filepath").head(3)
        assert len(samples)==3
        for i,row in enumerate(samples.itertuples()):
            reject(ROOT/row.filepath,f"{source}_{i+1}")
    print("PASS: solid green, 3 bean leaves, 3 objects; zero disease/Grad-CAM calls.",flush=True)
    final=pd.read_csv(ROOT/"reports/final_split/final_split_manifest.csv")
    positives=final[final.new_split.eq("train")].sort_values("filepath").groupby("class",sort=True).head(1)
    mapping=model_loader.get_class_mapping()
    for i,row in enumerate(positives.to_dict("records")):
        seen={}
        real_validate=prediction.validate_rgb
        real_prepare=prediction.prepare_disease_rgb
        def capture_validate(rgb):
            seen["rgb"]=rgb
            return real_validate(rgb)
        def capture_prepare(rgb):
            assert rgb is seen["rgb"],"Both models must share the same decoded RGB image."
            prepared=real_prepare(rgb)
            seen["prepared"]=prepared
            return prepared
        real_cam=xai.generate_gradcam
        def capture_cam(prepared,target,output):
            assert prepared is seen["prepared"]
            return real_cam(prepared,target,output)
        with patch.object(prediction,"decode_rgb",wraps=prediction.decode_rgb) as decode, patch.object(prediction,"validate_rgb",side_effect=capture_validate), patch.object(prediction,"prepare_disease_rgb",side_effect=capture_prepare), patch.object(prediction,"predict_prepared",wraps=prediction.predict_prepared) as disease, patch.object(xai,"generate_gradcam",side_effect=capture_cam) as cam:
            result=prediction.analyze_image(ROOT/row["filepath"],out/f"lemon_{i}.png")
        assert decode.call_count==disease.call_count==cam.call_count==1
        assert result["status"]=="lemon_leaf_analysis",result
        assert result["validator"]["accepted_as_lemon_leaf"]
        assert len(result["top_3"])==3 and np.isclose(sum(result["probabilities"]),1,atol=1e-5)
        probabilities=[p["probability"] for p in result["top_3"]]
        assert probabilities==sorted(probabilities,reverse=True)
        assert mapping[result["predicted_class"]]==result["predicted_class_index"]
        assert result["gradcam"]["status"]=="available"
        assert result["gradcam"]["target_class_index"]==result["predicted_class_index"]
        assert np.isclose(result["gradcam"]["target_probability"],result["confidence"],atol=1e-6)
        rows.append({"case":row["class"],"passed":True,"status":result["status"],"validator":result["validator"],
                     "disease_calls":disease.call_count,"gradcam_calls":cam.call_count,"decode_calls":decode.call_count,
                     "predicted_class":result["predicted_class"],"gradcam_target_class_index":result["gradcam"]["target_class_index"]})
        print("PASS:",row["class"],"shared RGB, disease structure, exact Grad-CAM target.",flush=True)
    with patch.object(prediction,"validate_rgb") as gate,patch.object(prediction,"predict_prepared") as disease,patch.object(xai,"generate_gradcam") as cam:
        corrupt=prediction.analyze_image(io.BytesIO(b"corrupt-image"))
    assert corrupt["status"]=="technical_error" and not gate.called and not disease.called and not cam.called
    for score,expected in [(np.nextafter(threshold,-np.inf),False),(threshold,True),(np.nextafter(threshold,np.inf),True)]:
        assert leaf_validator.result_from_score(score,meta)["accepted_as_lemon_leaf"] is expected
    assert leaf_validator.result_from_score(.45,meta)["accepted_as_lemon_leaf"], "Must not substitute a 0.5 threshold."
    # The actual pipeline is exercised below and exactly at the threshold.
    sample=ROOT/positives.iloc[0].filepath
    accepted=copy.deepcopy(result["validator"])
    disease_value={k:copy.deepcopy(result[k]) for k in ["predicted_class","predicted_class_index","confidence","top_3","probabilities","legacy_uncertain","warnings"]}
    for score,expected in [(np.nextafter(threshold,-np.inf),False),(threshold,True)]:
        mocked_gate=leaf_validator.result_from_score(score,meta)
        with patch.object(prediction,"validate_rgb",return_value=mocked_gate),patch.object(prediction,"predict_prepared",return_value=copy.deepcopy(disease_value)) as classify,patch.object(xai,"generate_gradcam",return_value={"status":"available"}) as explain:
            boundary=prediction.analyze_image(sample)
        assert classify.call_count==explain.call_count==int(expected)
        assert boundary["status"]==("lemon_leaf_analysis" if expected else "not_lemon_leaf")
    # Missing/unreadable model and inference failures both fail closed.
    for cause in [FileNotFoundError("simulated missing validator"),RuntimeError("simulated validator inference failure")]:
        with patch.object(leaf_validator,"get_validator_model",side_effect=cause),patch.object(prediction,"predict_prepared") as disease,patch.object(xai,"generate_gradcam") as cam:
            failed=prediction.analyze_image(sample)
        assert failed["status"]=="analysis_error" and failed["error"]["stage"]=="validator"
        assert not disease.called and not cam.called
    with patch.object(prediction,"validate_rgb",return_value=accepted),patch.object(prediction,"predict_prepared",side_effect=RuntimeError("simulated disease failure")),patch.object(xai,"generate_gradcam") as cam:
        failed=prediction.analyze_image(sample)
    assert failed["status"]=="analysis_error" and failed["validator"]==accepted and not cam.called
    disease_value["warnings"]=["Existing provisional warning."]
    with patch.object(prediction,"validate_rgb",return_value=accepted),patch.object(prediction,"predict_prepared",return_value=copy.deepcopy(disease_value)),patch.object(xai,"generate_gradcam",side_effect=RuntimeError("simulated explanation failure")):
        failed=prediction.analyze_image(sample)
    assert failed["status"]=="lemon_leaf_analysis" and failed["validator"]==accepted
    assert failed["gradcam"]["status"]=="unavailable" and "Existing provisional warning." in failed["warnings"]
    assert failed["probabilities"]==disease_value["probabilities"]
    # Flask receives rejection and structured errors without trying to render a disease.
    import app as flask_module
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        with patch.dict(flask_module.app.config,{"UPLOAD_FOLDER":temp,"RESULT_FOLDER":temp}),patch.object(flask_module,"render_template",return_value="handled") as render:
            for value in [green,corrupt,failed]:
                with patch.object(prediction,"analyze_image",return_value=value):
                    response=flask_module.app.test_client().post("/predict",data={"image":(io.BytesIO(b"mock-upload"),"sample.png")})
                    assert response.status_code==200
                    if value["status"]!="lemon_leaf_analysis":
                        assert render.call_args.kwargs["error"]==value["message"]
    assert all(sha(Path(p))==h for p,h in protected.items())
    assert stats=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in (ROOT/"dataset").rglob("*") if p.is_file()}
    summary={"validator_path":meta["model_path"],"validator_architecture":meta["architecture"],
             "validator_input_size":meta["input_size"],"validator_threshold":threshold,
             "validator_preprocessing":meta["preprocessing"],"disease_model_path":"model/inceptionv3_best_model.keras",
             "disease_architecture":"InceptionV3","disease_input_size":[299,299],"gradcam_layer":"inception_v3/mixed10",
             "smoke_test_results":rows,"all_tests_passed":True,"corrupt_image_status":corrupt["status"],
             "threshold_boundary_and_non_0_5_behavior_passed":True,
             "rejection_behavior":"No disease preparation, disease inference, model loading, or Grad-CAM; no disease label returned.",
             "failure_handling_behavior":{"validator":"analysis_error; fail closed","disease":"analysis_error; validator preserved",
                                          "gradcam":"lemon_leaf_analysis; prediction and previous warnings preserved"},
             "single_rgb_decode_verified":True,"flask_status_handling_passed":True,
             "protected_models_metadata_metrics_and_splits_unchanged":True,"original_dataset_stats_unchanged":True,
             "smoke_samples":"training manifest assignments only; no test evaluation or accuracy measurement",
             "limitations":["Validator benchmark does not establish broad real-world plant-species generalization.",
                            "Pillow versus training TensorFlow JPEG decoder can introduce small pixel differences.",
                            "Existing uncalibrated disease uncertainty and 410 unresolved cross-split similarity pairs remain."],
             "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    (out/"full_pipeline_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print("ALL FULL PIPELINE CHECKS PASSED",flush=True)

if __name__=="__main__": main()

