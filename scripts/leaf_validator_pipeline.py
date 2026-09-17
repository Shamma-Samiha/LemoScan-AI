"""Standalone validator training/evaluation implementation used by notebook 06."""
import os
os.environ.setdefault("TF_DETERMINISTIC_OPS","1")
import sys, json, time, hashlib, io
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from prepare_leaf_validator_data import build_dataset, sha, ROOT, DATA, OUT

MODEL = ROOT / "model/lemon_leaf_validator.keras"
META = ROOT / "model/lemon_leaf_validator_metadata.json"
SEED = 42
THRESHOLD_METHOD = ("Predeclared: among validation thresholds retaining at least 95% lemon recall and rejecting all validation synthetic negatives, "
                    "minimize false accepts; tie-break by highest lemon recall, then threshold closest to 0.5. "
                    "If infeasible, stop before test; never tune on test or final robustness results.")
LIMITATION = ("This is a dataset-specific binary gate, not a universal lemon-species verifier. "
              "Bean/object source domains differ from LemoScan. Other citrus species and broader field conditions are not covered. "
              "410 unresolved LemoScan C/D candidate pairs cross splits; complete biological-source independence is not proven.")

def save_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def setup():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    OUT.mkdir(parents=True,exist_ok=True)
    paths = [ROOT / name for name in ["model/inceptionv3_best_model.keras","model/inceptionv3_phase1_best.keras",
             "model/inceptionv3_finetune_best.keras","model/resnet_best_model.keras","model/model_metadata.json",
             "reports/final_split/final_split_manifest.csv","reports/dataset_audit/manual_review_decisions.csv","app.py","streamlit_app.py"]]
    paths += list((ROOT / "reports/inceptionv3_test").glob("*")) + list((ROOT / "utils").glob("*.py"))
    baseline = {str(p):sha(p) for p in paths if p.is_file()}
    stats = {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in (ROOT / "dataset").rglob("*") if p.is_file()}
    print("TensorFlow",tf.__version__,"Devices",tf.config.list_physical_devices())
    return baseline, stats

def dataset(frame, training=False):
    if training:
        assert frame.split.eq("train").all()
    labels = frame.label.eq("LEMON_LEAF").to_numpy(dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices(([(ROOT / p).as_posix() for p in frame.filepath],labels))
    def decode(path,label):
        image = tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        image.set_shape((None,None,3))
        return tf.image.resize(tf.cast(image,tf.float32),(224,224),method="bilinear"), label
    if training:
        ds = ds.shuffle(len(frame),seed=SEED,reshuffle_each_iteration=True)
    ds = ds.map(decode,num_parallel_calls=2,deterministic=True)
    ds = ds.batch(32)
    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal",seed=42),
            tf.keras.layers.RandomRotation(.04,seed=43,fill_mode="reflect"),
            tf.keras.layers.RandomZoom(.06,seed=44,fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(.04,.04,seed=45,fill_mode="reflect"),
            tf.keras.layers.RandomContrast(.08,seed=46)])
        ds = ds.map(lambda x,y:(tf.clip_by_value(aug(x,training=True),0,255),y),num_parallel_calls=1)
    options=tf.data.Options()
    options.experimental_deterministic=True
    options.threading.private_threadpool_size=2
    return ds.with_options(options).prefetch(2)

def train_model(manifest):
    history_path=OUT / "training_history.csv"
    completed=OUT / "training_completed.json"
    if MODEL.exists() and completed.exists():
        print("Reusing completed validator training; no retraining.")
        return pd.read_csv(history_path),json.loads(completed.read_text())
    assert not MODEL.exists(), "Partial/existing validator checkpoint found. Inspect before any retraining."
    assert not (OUT / "frozen_threshold.json").exists(), "Validator already frozen; do not retrain."
    with (OUT / "training_started.json").open("x",encoding="utf-8") as lock:
        json.dump({"pid":os.getpid(),"started":datetime.now(timezone.utc).isoformat()},lock)
    base=tf.keras.applications.MobileNetV3Small(input_shape=(224,224,3),include_top=False,weights="imagenet",include_preprocessing=True)
    base.trainable=False
    inputs=tf.keras.Input((224,224,3),name="rgb_0_255")
    features=base(inputs,training=False)
    x=tf.keras.layers.GlobalAveragePooling2D()(features)
    x=tf.keras.layers.Dropout(.25,seed=47)(x)
    outputs=tf.keras.layers.Dense(1,activation="sigmoid",name="lemon_probability")(x)
    model=tf.keras.Model(inputs,outputs,name="lemon_leaf_validator")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"])
    model.summary()
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(MODEL),monitor="val_loss",save_best_only=True,verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",patience=2,factor=.3,min_lr=1e-6,verbose=1),
        tf.keras.callbacks.CSVLogger(str(history_path)),
        tf.keras.callbacks.TerminateOnNaN()]
    started=time.perf_counter()
    history=model.fit(dataset(manifest[manifest.split.eq("train")],True),
                      validation_data=dataset(manifest[manifest.split.eq("val")]),epochs=12,callbacks=callbacks,verbose=2)
    elapsed=time.perf_counter()-started
    frame=pd.DataFrame(history.history)
    frame.insert(0,"epoch",np.arange(1,len(frame)+1))
    frame.to_csv(history_path,index=False)
    info={"training_seconds":elapsed,"epochs":len(frame),"best_epoch":int(frame.val_loss.idxmin()+1),
          "architecture":"MobileNetV3Small","backbone_frozen":True,"fine_tuning":False,"seed":SEED,
          "batch_size":32,"learning_rate":.001,"dropout":.25,
          "model_sha256":sha(MODEL),"selection":"minimum validation loss",
          "augmentation":{"flip":"horizontal","rotation":.04,"zoom":.06,"translation":.04,"contrast":.08}}
    save_json(completed,info)
    return frame,info

def plot_history(history):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,key in zip(axes,["loss","accuracy"]):
        ax.plot(history.epoch,history[key],label="train")
        ax.plot(history.epoch,history["val_"+key],label="validation")
        ax.set(xlabel="Epoch",ylabel=key,title="Validator "+key)
        ax.legend()
    fig.savefig(OUT / "training_curves.png",dpi=150)
    plt.show()

def infer_once(frame, name):
    cache=OUT / (name+"_inference.npz")
    fingerprint={"model_sha256":sha(MODEL),"manifest_sha256":sha(OUT/"validator_dataset_manifest.csv"),
                 "paths":frame.filepath.tolist()}
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:
            assert json.loads(str(data["fingerprint"]))==fingerprint
            return data["probabilities"]
    lock=OUT / (name+"_inference_started.json")
    assert not lock.exists(),"An earlier inference did not save its cache; inspect before repeating."
    if name=="test":
        freeze=json.loads((OUT / "frozen_threshold.json").read_text())
        assert freeze["model_sha256"]==sha(MODEL)
    model=tf.keras.models.load_model(MODEL,compile=False)
    with lock.open("x",encoding="utf-8") as f:
        json.dump(fingerprint,f)
    probabilities=np.concatenate([model(images,training=False).numpy().ravel() for images,_ in dataset(frame)])
    assert len(probabilities)==len(frame) and np.isfinite(probabilities).all()
    temp=cache.with_suffix(".tmp.npz")
    np.savez_compressed(temp,probabilities=probabilities,fingerprint=json.dumps(fingerprint))
    temp.replace(cache)
    return probabilities

def metrics(frame, probabilities, threshold):
    truth=frame.label.eq("LEMON_LEAF").to_numpy(dtype=int)
    pred=(probabilities>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
    return {"accuracy":float(accuracy_score(truth,pred)),"precision":float(precision_score(truth,pred,zero_division=0)),
            "recall":float(recall_score(truth,pred,zero_division=0)),"f1":float(f1_score(truth,pred,zero_division=0)),
            "false_accepts":int(fp),"false_rejects":int(fn),"true_accepts":int(tp),"true_rejects":int(tn),
            "false_positive_rate":float(fp/(fp+tn)),"false_negative_rate":float(fn/(fn+tp)),
            "images":len(frame),"threshold":float(threshold),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}

def select_threshold(manifest):
    frame=manifest[manifest.split.eq("val")].reset_index(drop=True)
    probabilities=infer_once(frame,"validation")
    frozen=OUT / "frozen_threshold.json"
    if frozen.exists():
        info=json.loads(frozen.read_text())
        assert info["model_sha256"]==sha(MODEL)
        return info
    # Each observed score defines a possible operating point under >= acceptance.
    thresholds=np.unique(np.r_[0.,.5,probabilities.astype(float),np.nextafter(float(probabilities.max()),np.inf)])
    synthetic=frame.source.eq("synthetic").to_numpy()
    rows=[]
    for threshold in thresholds:
        result=metrics(frame,probabilities,threshold)
        result.pop("confusion_matrix")
        result["synthetic_false_accepts"]=int(np.sum(probabilities[synthetic]>=threshold))
        rows.append(result)
    analysis=pd.DataFrame(rows)
    analysis.to_csv(OUT / "threshold_analysis.csv",index=False)
    feasible=analysis[(analysis.recall>=.95)&(analysis.synthetic_false_accepts==0)].copy()
    assert not feasible.empty,"No threshold meets predeclared validation constraints; test remains unopened."
    feasible["distance_to_half"]=abs(feasible.threshold-.5)
    chosen=feasible.sort_values(["false_accepts","recall","distance_to_half","threshold"],ascending=[True,False,True,True]).iloc[0]
    threshold=float(chosen.threshold)
    val_metrics=metrics(frame,probabilities,threshold)
    save_json(OUT / "validation_metrics.json",val_metrics)
    predictions=frame.copy()
    predictions["lemon_probability"]=probabilities
    predictions["accepted"]=probabilities>=threshold
    predictions.to_csv(OUT / "validation_predictions.csv",index=False)
    info={"threshold":threshold,"method":THRESHOLD_METHOD,"validation_metrics":val_metrics,
          "model_sha256":sha(MODEL),"manifest_sha256":sha(OUT/"validator_dataset_manifest.csv"),
          "frozen_before_test":True,"timestamp_utc":datetime.now(timezone.utc).isoformat()}
    save_json(frozen,info)
    metadata={"architecture":"MobileNetV3Small","model_path":"model/lemon_leaf_validator.keras","input_size":[224,224],
              "preprocessing":"RGB float32 bilinear resize to 224x224; input range [0,255]; internal MobileNetV3 Rescaling to [-1,1]; no external preprocess_input",
              "labels":{"NOT_LEMON_LEAF":0,"LEMON_LEAF":1},"chosen_acceptance_threshold":threshold,
              "acceptance_rule":"lemon_probability >= threshold","threshold_selection_method":THRESHOLD_METHOD,
              "validation_metrics":val_metrics,"final_test_metrics":None,"random_seed":SEED,
              "negative_dataset_sources":json.loads((OUT/"negative_provenance.json").read_text()),
              "model_sha256":sha(MODEL),"frozen_before_test":True,"limitation":LIMITATION}
    save_json(META,metadata)
    print("Threshold frozen BEFORE test:",threshold,flush=True)
    print(json.dumps(val_metrics,indent=2),flush=True)
    return info

def evaluate_test(manifest,freeze):
    frame=manifest[manifest.split.eq("test")].reset_index(drop=True)
    assert len(frame)==614
    assert freeze["model_sha256"]==sha(MODEL)
    probabilities=infer_once(frame,"test")
    result=metrics(frame,probabilities,freeze["threshold"])
    save_json(OUT/"test_metrics.json",result)
    frame["lemon_probability"]=probabilities
    frame["accepted"]=probabilities>=freeze["threshold"]
    frame["correct"]=frame.accepted.eq(frame.label.eq("LEMON_LEAF"))
    frame.to_csv(OUT/"test_predictions.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,5),layout="constrained")
    cm=np.asarray(result["confusion_matrix"])
    ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["NOT LEMON","LEMON"],yticklabels=["NOT LEMON","LEMON"],
           xlabel="Predicted",ylabel="True",title="Frozen validator — final test")
    fig.savefig(OUT/"confusion_matrix.png",dpi=150)
    plt.show()
    errors=frame[~frame.correct].copy()
    errors["error_confidence"]=np.where(errors.accepted,errors.lemon_probability,1-errors.lemon_probability)
    examples=errors.sort_values("error_confidence",ascending=False).head(8)
    fig,axes=plt.subplots(2,4,figsize=(14,7),layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax,row in zip(axes.flat,examples.itertuples()):
        with Image.open(ROOT/row.filepath) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"True: {row.label}\nP(lemon): {row.lemon_probability:.3f}\nSource: {row.source}",fontsize=9)
    if examples.empty:
        axes[0,0].text(.5,.5,"No test errors",ha="center")
    fig.suptitle("Final test errors — analysis only, no retuning")
    fig.savefig(OUT/"error_examples.png",dpi=140)
    plt.show()
    metadata=json.loads(META.read_text())
    metadata["final_test_metrics"]=result
    metadata["test_evaluated_once"]=True
    save_json(META,metadata)
    print(json.dumps(result,indent=2),flush=True)
    return result,frame

def robustness(manifest,freeze,test_predictions):
    path=OUT/"robustness_results.json"
    if path.exists():
        return json.loads(path.read_text())
    # Existing final-test probabilities reused; no extra test inference.
    rows=[]
    for source in ["beans","caltech101","lemoscan"]:
        for row in test_predictions[test_predictions.source.eq(source)].head(6).itertuples():
            rows.append({"case":source,"filepath":row.filepath,"lemon_probability":float(row.lemon_probability),
                         "accepted":bool(row.accepted),"expected_accept":source=="lemoscan"})
    green=np.zeros((1,224,224,3),dtype=np.float32)
    green[:,:,:,1]=255
    model=tf.keras.models.load_model(MODEL,compile=False)
    probability=float(model(green,training=False).numpy()[0,0])
    green_result={"case":"canonical_solid_green_RGB_0_255_0","lemon_probability":probability,
                  "accepted":bool(probability>=freeze["threshold"]),"expected_accept":False}
    rows.append(green_result)
    result={"cases":rows,"solid_green_rejected":not green_result["accepted"],
            "passed":all(row["accepted"]==row["expected_accept"] for row in rows),
            "policy":"Post-freeze robustness inspection only; never retune on these results."}
    save_json(path,result)
    Image.fromarray(green[0].astype(np.uint8)).save(OUT/"solid_green_robustness.png")
    print("Canonical solid green rejected:",result["solid_green_rejected"],"P(lemon):",probability,flush=True)
    return result

def finalize(manifest,train_info,freeze,test_results,robust,baseline,stats):
    assert all(sha(Path(p))==digest for p,digest in baseline.items())
    assert stats=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in (ROOT/"dataset").rglob("*") if p.is_file()}
    summary={"positive_count":2050,"negative_count":2050,"hard_leaf_negative_count":1000,"generic_object_negative_count":1000,
             "synthetic_negative_count":50,"split_counts":manifest.groupby(["split","label"]).size().unstack().to_dict(orient="index"),
             "training":train_info,"selected_threshold":freeze["threshold"],"threshold_selection_method":THRESHOLD_METHOD,
             "validation_metrics":freeze["validation_metrics"],"final_test_metrics":test_results,
             "robustness":robust,"model_path":"model/lemon_leaf_validator.keras","metadata_path":"model/lemon_leaf_validator_metadata.json",
             "model_sha256":sha(MODEL),"source_provenance":json.loads((OUT/"negative_provenance.json").read_text()),
             "protected_disease_artifacts_and_application_unchanged":True,"source_images_unchanged":True,
             "limitation":LIMITATION,"integrated_into_app":False,
             "deployment_status":"candidate_requires_real_world_validation" if robust["solid_green_rejected"] else "blocked_solid_green_false_accept",
             "test_integrity":"Model and acceptance threshold were frozen before one test inference pass. No post-test retuning.",
             "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    save_json(OUT/"validator_summary.json",summary)
    return summary

