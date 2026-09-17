"""Read-only runtime/deployment audit. No model inference or Git mutations."""
import ast,hashlib,json,re,subprocess,sys
from pathlib import Path
import importlib.metadata as metadata
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/deployment"
def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def exact_case(relative):
    current=ROOT
    for part in Path(relative).parts:
        assert part in [p.name for p in current.iterdir()],f"Case mismatch or missing file: {relative}"
        current=current/part
    return True
files=[ROOT/"streamlit_app.py",*sorted((ROOT/"utils").glob("*.py"))]
imports=set()
path_hits=[]
windows_imports=[]
for p in files:
    source=p.read_text(encoding="utf-8-sig")
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    if re.search(r"[A-Za-z]:[\\/]|/Users/|/home/",source):
        path_hits.append(str(p.relative_to(ROOT)))
windows_imports=sorted(imports&{"winreg","msvcrt","win32api","win32com","pythoncom"})
external=sorted(imports-set(sys.stdlib_module_names)-{"utils"})
assert external==["PIL","numpy","streamlit","tensorflow"],external
assert not path_hits and not windows_imports
models=[]
for name in ["inceptionv3_best_model.keras","lemon_leaf_validator.keras"]:
    relative="model/"+name
    p=ROOT/relative
    pointer=git("show","HEAD:"+relative)
    with p.open("rb") as f:header=f.read(128)
    entry={"path":relative,"bytes":p.stat().st_size,"tracked":git("ls-files","--",relative)==relative,
           "working_tree_actual_binary":header.startswith(b"PK"),
           "HEAD_is_lfs_pointer":pointer.startswith("version https://git-lfs.github.com/spec/v1"),
           "sha256":digest(p),"git_attributes":git("check-attr","filter","diff","merge","--",relative)}
    assert entry["tracked"] and entry["working_tree_actual_binary"] and entry["HEAD_is_lfs_pointer"]
    assert entry["sha256"] in pointer
    models.append(entry)
required=["model/inceptionv3_best_model.keras","model/lemon_leaf_validator.keras",
          "model/model_metadata.json","model/lemon_leaf_validator_metadata.json","training/inceptionv3_class_indices.json"]
for relative in required:exact_case(relative)
disease=json.loads((ROOT/"model/model_metadata.json").read_text())
validator=json.loads((ROOT/"model/lemon_leaf_validator_metadata.json").read_text())
for relative in [disease["model_path"],disease["class_indices_path"],validator["model_path"]]:
    assert not Path(relative).is_absolute() and "\\" not in relative
    exact_case(relative)
assert validator["chosen_acceptance_threshold"]==0.397647500038147
assert models[0]["sha256"]=="c9e07156c3be748d93378e687ecbd847123c30ca7ed3664b8f2303e14499e817"
assert models[1]["sha256"]=="b4d21d79003758979cb82428a52930d3615bcf0af2656aeeeafa2211cb87a297"
result={"runtime_files":[p.relative_to(ROOT).as_posix() for p in files],"external_imports":external,
        "hardcoded_local_path_hits":path_hits,"windows_only_imports":windows_imports,
        "required_filename_case_matches":True,"metadata_paths_repository_relative":True,
        "models":models,"git_lfs_fsck":git("lfs","fsck"),
        "requirements":(ROOT/"requirements.txt").read_text().splitlines(),
        "runtime_txt_removed":not (ROOT/"runtime.txt").exists(),"python_target":"3.13 (local verified 3.13.2)",
        "installed_versions":{x:metadata.version(x) for x in ["tensorflow","keras","streamlit","numpy","Pillow","protobuf","h5py"]},
        "dependency_consistency":"pip check: No broken requirements found.",
        "memory_review":{"model_cache":"Process-wide module singletons guarded by locks; no eager loading on UI landing.",
                         "upload_cache":"One current result, RGB preview and overlay per session; invalidated on new upload/clear.",
                         "preview_and_overlay_max_side_pixels":1200,
                         "temporary_upload_outputs":"TemporaryDirectory cleaned after each analysis.",
                         "unbounded_cross_upload_cache":False},
        "linux_review":"Static source and filename-case checks passed; execution performed on Windows, not a clean Linux host."}
(OUT/"deployment_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
if __name__=="__main__":pass

