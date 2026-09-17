"""Verify saved validator outputs only; does not load or evaluate a model."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/leaf_validator"
def sha(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    summary=json.loads((OUT/"validator_summary.json").read_text())
    meta=json.loads((ROOT/"model/lemon_leaf_validator_metadata.json").read_text())
    freeze=json.loads((OUT/"frozen_threshold.json").read_text())
    model_hash=sha(ROOT/"model/lemon_leaf_validator.keras")
    assert model_hash==summary["model_sha256"]==meta["model_sha256"]==freeze["model_sha256"]
    assert summary["selected_threshold"]==meta["chosen_acceptance_threshold"]==freeze["threshold"]
    assert meta["test_evaluated_once"] and not summary["integrated_into_app"]
    manifest=pd.read_csv(OUT/"validator_dataset_manifest.csv",keep_default_na=False)
    test=pd.read_csv(OUT/"test_predictions.csv")
    assert len(manifest)==4100 and len(test)==614 and not manifest.filepath.duplicated().any()
    assert set(test.filepath)==set(manifest.loc[manifest.split.eq("test"),"filepath"])
    y=test.label.eq("LEMON_LEAF").to_numpy()
    pred=test.lemon_probability.to_numpy()>=freeze["threshold"]
    expected=summary["final_test_metrics"]
    assert int(np.sum(pred&~y))==expected["false_accepts"]
    assert int(np.sum(~pred&y))==expected["false_rejects"]
    assert np.isclose(np.mean(pred==y),expected["accuracy"])
    validation=pd.read_csv(OUT/"validation_predictions.csv")
    p=np.clip(validation.lemon_probability.to_numpy(),1e-7,1-1e-7)
    y=validation.label.eq("LEMON_LEAF").to_numpy(dtype=float)
    bce=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
    history=pd.read_csv(OUT/"training_history.csv")
    assert np.isclose(bce,history.val_loss.min(),atol=1e-5), (bce,history.val_loss.min())
    notebook=json.loads((ROOT/"notebooks/06_lemon_leaf_validator.ipynb").read_text(encoding="utf-8"))
    assert notebook["metadata"]["validator_execution"]["completed"]
    code=[c for c in notebook["cells"] if c["cell_type"]=="code"]
    assert all(c["execution_count"] and not any(o["output_type"]=="error" for o in c["outputs"]) for c in code)
    required=["validator_dataset_manifest.csv","training_history.csv","validation_metrics.json","test_metrics.json",
              "confusion_matrix.png","threshold_analysis.csv","error_examples.png","validator_summary.json"]
    assert all((OUT/f).is_file() for f in required)
    result={"all_required_artifacts_present":True,"all_notebook_code_cells_executed":len(code),
            "model_and_threshold_hashes_consistent":True,"test_counts_recomputed_from_saved_predictions":True,
            "selected_model_validation_loss_matches_best_training_checkpoint":True,
            "cached_validation_binary_crossentropy":bce,"test_inference_repeated":False}
    (OUT/"artifact_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()


