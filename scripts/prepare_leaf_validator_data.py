"""Prepare independent validator data from official TFDS-referenced archives."""
from pathlib import Path
import hashlib, io, json, tarfile, zipfile, urllib.request
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/leaf_validator"
OUT = ROOT / "reports/leaf_validator"
SEED = 42
URLS = {
 "beans_train": "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip",
 "beans_validation": "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/validation.zip",
 "beans_test": "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/test.zip",
 "caltech101": "https://huggingface.co/datasets/HuggingFaceM4/Caltech-101/resolve/71d2394c09f83658fb6f6a1eb272a0d9dc71b331/caltech-101.zip"}
# Deliberately omit plants, flowers, fruit, background, and insects on foliage.
CATEGORIES = ["airplanes", "anchor", "barrel", "revolver", "buddha", "camera", "car_side",
              "ceiling_fan", "cellphone", "chair", "chandelier", "cup", "dollar_bill",
              "euphonium", "ewer", "ferry", "soccer_ball", "grand_piano", "gramophone",
              "headphone", "helicopter", "schooner", "laptop", "motorbikes", "watch"]

def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def download(key):
    folder = DATA / "archives"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (key + ".zip")
    if not path.exists():
        print("Downloading official archive:", key, flush=True)
        temp = path.with_suffix(".part")
        request = urllib.request.Request(URLS[key], headers={"User-Agent": "LemoScan-validator/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as target:
            while block := response.read(1024*1024):
                target.write(block)
        temp.replace(path)
    expected = {"beans_train":"284fe8456ce20687f4367ae7ad94a64577e7f9fde2c2c6b1c74340ab5dc82715",
                "beans_validation":"90b7aa1c26d91d9afff07a30bbc67a5ea34f1f1397f068d8675be09d7d0c602d",
                "beans_test":"ca67b15d960d1e2fd9d23fd8498ce86818ead90755c630a43baf19ec4af09312",
                "caltech101":"331234750fc7f77520e50d9565e8b6907b03565e30320da1401130db08c61f91"}
    assert sha(path) == expected[key], f"Archive checksum mismatch: {key}"
    assert zipfile.is_zipfile(path), f"Invalid archive: {path}"
    return path

def build_dataset():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "validator_dataset_manifest.csv"
    if manifest_path.exists():
        result = pd.read_csv(manifest_path, keep_default_na=False)
        assert len(result) == 4100
        assert all((ROOT / p).is_file() for p in result.filepath)
        return result
    archives = {key: download(key) for key in URLS}
    rng = np.random.default_rng(SEED)
    seen = set()
    def candidate(raw, source, identity, category):
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if pixel_hash in seen:
            return None
        seen.add(pixel_hash)
        return {"raw": raw, "source": source, "id": identity, "category": category, "pixel_sha256": pixel_hash}
    beans = []
    for key in ["beans_train", "beans_validation", "beans_test"]:
        with zipfile.ZipFile(archives[key]) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    item = candidate(archive.read(name), "beans", key + "/" + name, name.split("/")[-2])
                    if item:
                        beans.append(item)
    objects = {name: [] for name in CATEGORIES}
    object_records = []
    with zipfile.ZipFile(archives["caltech101"]) as archive:
        nested = next(name for name in archive.namelist() if name.endswith("101_ObjectCategories.tar.gz"))
        with tarfile.open(fileobj=io.BytesIO(archive.read(nested)), mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".jpg"):
                    continue
                category = member.name.split("/")[-2].lower()
                if category in objects:
                    object_records.append((member.name, category, tar.extractfile(member).read()))
    # Sequential archive I/O, canonical identity order for duplicate resolution.
    for name, category, raw in sorted(object_records):
        item = candidate(raw, "caltech101", name, category)
        if item:
            objects[category].append(item)
    selected = []
    def sample_split(items, counts, kind):
        items = sorted(items, key=lambda item: ({"beans_train":0,"beans_validation":1,"beans_test":2}.get(item["id"].split("/")[0],0), item["id"]))
        assert len(items) >= sum(counts), (kind, len(items), counts)
        chosen = [items[i] for i in rng.permutation(len(items))[:sum(counts)]]
        offset = 0
        for split, count in zip(["train", "val", "test"], counts):
            for item in chosen[offset:offset+count]:
                selected.append((item, split, kind))
            offset += count
    sample_split(beans, [700,150,150], "other_leaf")
    for category in CATEGORIES:
        sample_split(objects[category], [28,6,6], "generic_object")
    rows = []
    external = DATA / "external"
    external.mkdir(parents=True, exist_ok=True)
    for item, split, kind in selected:
        path = external / (item["pixel_sha256"] + ".jpg")
        path.write_bytes(item["raw"])
        rows.append({"source": item["source"], "source_id_or_filepath": item["id"],
                     "label": "NOT_LEMON_LEAF", "split": split, "negative_type": kind,
                     "notes": "Official TFDS-referenced archive; category=" + item["category"],
                     "filepath": path.relative_to(ROOT).as_posix(), "pixel_sha256": item["pixel_sha256"]})
    # 50 synthetic images = 2.44% of negatives, 1.22% of all training images.
    synthetic = DATA / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)
    for i in range(50):
        local = np.random.default_rng(SEED + 1000 + i)
        kind = ["green_solid", "solid_color", "gradient", "noise"][i % 4]
        if kind == "green_solid":
            color = [int(local.integers(0,70)), int(local.integers(100,250)), int(local.integers(0,70))]
            array = np.full((224,224,3), color, dtype=np.uint8)
        elif kind == "solid_color":
            array = np.full((224,224,3), local.integers(0,256,3), dtype=np.uint8)
        elif kind == "gradient":
            start, end = local.integers(0,256,(2,3))
            line = np.linspace(start,end,224).astype(np.uint8)
            array = np.broadcast_to(line[None,:,:],(224,224,3)).copy()
        else:
            array = local.integers(0,256,(224,224,3),dtype=np.uint8)
        path = synthetic / f"{i:03d}_{kind}.png"
        Image.fromarray(array).save(path)
        split = "train" if i < 35 else "val" if i < 43 else "test"
        rows.append({"source": "synthetic", "source_id_or_filepath": f"seed{SEED+1000+i}:{kind}",
                     "label": "NOT_LEMON_LEAF", "split": split, "negative_type": kind,
                     "notes": "Deterministic synthetic; canonical RGB(0,255,0) reserved for final robustness.",
                     "filepath": path.relative_to(ROOT).as_posix(),
                     "pixel_sha256": hashlib.sha256(str((224,224)).encode()+array.tobytes()).hexdigest()})
    disease = pd.read_csv(ROOT / "reports/final_split/final_split_manifest.csv")
    assert len(disease) == 2050 and not disease.filepath.duplicated().any()
    for row in disease.to_dict("records"):
        rows.append({"source": "lemoscan", "source_id_or_filepath": row["filepath"], "label": "LEMON_LEAF",
                     "split": row["new_split"], "negative_type": "", "notes": "Preserved final manifest; " + row["class"],
                     "filepath": row["filepath"], "pixel_sha256": ""})
    result = pd.DataFrame(rows).sort_values(["split","source","source_id_or_filepath"]).reset_index(drop=True)
    negatives = result[result.label.eq("NOT_LEMON_LEAF")]
    assert not negatives.pixel_sha256.duplicated().any()
    assert not result.filepath.duplicated().any()
    assert result.groupby(["split","label"]).size().to_dict() == {
        ("train","LEMON_LEAF"):1435, ("train","NOT_LEMON_LEAF"):1435,
        ("val","LEMON_LEAF"):308, ("val","NOT_LEMON_LEAF"):308,
        ("test","LEMON_LEAF"):307, ("test","NOT_LEMON_LEAF"):307}
    positive = result[result.label.eq("LEMON_LEAF")].set_index("filepath")
    assert all(positive.loc[row.filepath,"split"] == row.new_split for row in disease.itertuples())
    result.to_csv(manifest_path,index=False)
    provenance = {"seed":SEED, "source_urls":URLS, "archive_sha256":{k:sha(v) for k,v in archives.items()},
        "tfds_fallback":"TFDS 4.9.10 installed but download_and_prepare failed: missing importlib_resources. Original GCS/Caltech hosts returned HTTP 403; identical archives downloaded from AI-Lab-Makerere and HuggingFaceM4 mirrors with SHA256 checks; no packages installed.",
        "caltech_categories":CATEGORIES, "selection":"1000 Beans, 40 examples per each of 25 Caltech categories, 50 synthetic; exact decoded-RGB duplicates removed before sampling and splitting.",
        "counts":result.groupby(["source","split"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "caltech_catalog":"https://www.tensorflow.org/datasets/catalog/caltech101",
        "beans_catalog":"https://www.tensorflow.org/datasets/catalog/beans",
        "limitation":"Class-level semantic filtering only; object backgrounds not exhaustively audited. Bean source/session independence is not proven. LemoScan retains the 410 unresolved C/D cross-split candidate limitation."}
    (OUT / "negative_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(result.groupby(["source","split"]).size(), flush=True)
    return result

if __name__ == "__main__":
    build_dataset()

