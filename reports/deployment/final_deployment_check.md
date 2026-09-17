# Final deployment check

Status: **READY TO DEPLOY** for a manual deployment attempt. Local hardening and smoke checks passed; hosted Linux execution and remote LFS downloads have not been tested. No deployment was performed.

## Changes

- Replaced clipped sidebar metrics with stacked compact cards: Test accuracy **95.44%**, Macro F1 **95.51%**. Browser measurements confirm no clipping.
- Moved the 410 unresolved Category C/D cross-split similarity-pair detail into the collapsed **Methodology & limitations** expander. Validator benchmark limitations and experimental/research disclaimer remain accessible; the footer disclaimer remains visible.
- Bounded retained display previews and explanation overlays to 1200 pixels on their longest side. Model inputs and inference behavior are unchanged.
- Updated root requirements.txt and removed runtime.txt, which incorrectly specified Python 3.10.13.
- Models, thresholds, evaluation metrics, dataset splits and source images were not modified. No training was run.

## Runtime requirements

```text
streamlit==1.49.0
tensorflow-cpu==2.21.0
keras==3.14.1
numpy==2.3.4
Pillow==11.3.0
protobuf==6.33.5
h5py==3.14.0
```

AST inspection of streamlit_app.py and utils/ found external imports PIL, numpy, streamlit and tensorflow. Keras, protobuf and h5py are pinned runtime dependencies for TensorFlow/model loading. No notebook/training-only packages were added.

`python -m pip check`: **No broken requirements found.** Local execution used Python 3.13.2 and the listed versions, with the Windows `tensorflow==2.21.0` distribution. The Linux deployment requirement is `tensorflow-cpu==2.21.0`; a clean Linux installation was not executed. PyPI release metadata provides a CPython 3.13 Linux x86-64 wheel for that release: [TensorFlow CPU package metadata](https://pypi.org/pypi/tensorflow-cpu/2.21.0/json).

## Python target

Select **Python 3.13** manually in Community Cloud Advanced settings. There is no runtime.txt fallback or conflicting Python 3.10 declaration. Python selection follows the [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Model artifacts and Git LFS

| File | Exact bytes | Git tracked | Working tree | HEAD Git blob |
| --- | ---: | --- | --- | --- |
| model/inceptionv3_best_model.keras | 137,022,324 | Yes | Actual binary | LFS pointer |
| model/lemon_leaf_validator.keras | 4,336,307 | Yes | Actual binary | LFS pointer |

SHA-256 values:

```text
inceptionv3_best_model.keras
c9e07156c3be748d93378e687ecbd847123c30ca7ed3664b8f2303e14499e817
lemon_leaf_validator.keras
b4d21d79003758979cb82428a52930d3615bcf0af2656aeeeafa2211cb87a297
```

Existing .gitattributes contains `*.keras filter=lfs diff=lfs merge=lfs -text`. Both files have the expected LFS attributes; working files contain actual ZIP/Keras binaries and match their LFS object hashes. Git LFS 3.6.1 is installed. `git lfs fsck` returned **Git LFS fsck OK**. LFS is already configured; no history migration is needed or was run. Remote object availability and account quota were not verified. Community Cloud supports LFS repositories according to its [file organization documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Paths and Linux compatibility

Runtime source audit found zero hardcoded local Windows paths and zero Windows-only imports. Required artifact filename capitalization matches exactly. Metadata paths are repository-relative with forward slashes, resolved using pathlib from the module location. TemporaryDirectory usage is portable. No drive E: dependency exists in the default runtime configuration. Leave the optional LEMOSCAN_MODEL_METADATA override unset on Cloud.

Deploy streamlit_app.py, utils/, .streamlit/config.toml, requirements.txt, both model binaries, model/model_metadata.json, model/lemon_leaf_validator_metadata.json and training/inceptionv3_class_indices.json. Training datasets and notebooks are not required for inference. Static Linux compatibility checks passed; actual execution was on Windows.

## Loading and memory

Disease and validator models use lazy process-wide module caches guarded by locks, equivalent to shared resource caching. They are not loaded afresh on each Streamlit rerun. Landing-page rendering does not eagerly load models. Session state retains only the current upload result, preview and overlay; a new upload or Clear invalidates them. There is no unbounded cross-upload result cache. Temporary analysis files are cleaned up.

Observed local Streamlit process working set after tests: **287,940,608 bytes**; peak working set: **1,309,237,248 bytes** (approximately 1.22 GiB). This Windows measurement is not a prediction of Linux usage. Hosted memory headroom, cold model loading and concurrent sessions still require monitoring. Display thumbnails reduce retained UI memory without altering inference inputs.

## Local smoke tests

The running app at localhost:8510 was tested through a real Edge browser using actual models after changes.

| Case | Result |
| --- | --- |
| Sidebar full percentages, narrow sidebar | PASS; values uncut, scroll width equals client width |
| Valid lemon image | PASS; analysis and explanation rendered |
| Solid green | PASS; rejected before disease analysis |
| Non-lemon bean leaf | PASS; rejected |
| Non-lemon object | PASS; rejected |
| Repeated same-upload rerun / explanation toggle | PASS; prediction preserved |
| Corrupt image | PASS; controlled error, no visible traceback |
| Mobile 390px viewport | PASS; no horizontal overflow |

Controller tests with mocked inference also passed: no automatic inference, exactly one inference across cached reruns, new-upload invalidation, validator/disease failure handling, unavailable Grad-CAM handling and uncertainty display. These smoke tests do not recompute or change evaluation metrics. `git diff --check` passed.

Evidence: [deployment audit](deployment_audit.json), [browser results](../streamlit_ui/browser_smoke_results.json), [controller results](../streamlit_ui/controller_smoke_results.json). Screenshots are in reports/streamlit_ui/. This report supersedes earlier deployment-readiness notes from the UI implementation stage.

## Remaining blockers and verification limits

No known local code or dependency blocker remains. Before publishing, commit and push the selected deployment files and ensure the required LFS objects upload successfully. Remote LFS availability, clean Linux dependency installation and hosted resource headroom remain deployment-time checks. If the host cannot accommodate inference memory, deployment is blocked on hosting capacity; local smoke success does not establish that capacity.

## Exact manual deployment steps

Run from the repository root. These commands stage the deployment changes only, not unrelated datasets or browser profiles. Inspect the staged diff before committing.

```powershell
git branch --show-current
git lfs install
git lfs ls-files
git lfs fsck
git add streamlit_app.py requirements.txt .streamlit/config.toml
git add -u -- runtime.txt
git add scripts/audit_deployment.py scripts/test_streamlit_browser.py scripts/test_streamlit_controller.py reports/deployment/
git diff --cached --stat
git diff --cached
git commit -m "Harden Streamlit deployment"
git lfs push origin HEAD
git push -u origin HEAD
```

Current branch: `rebuild/robust-inceptionv3`. Origin: `https://github.com/Shamma-Samiha/LemoScan-AI.git`. If either push fails, resolve that failure before deploying. On a fresh checkout, run `git lfs pull` to materialize model binaries; do not deploy pointer text as a model. No `git lfs migrate` command is required.

1. In Streamlit Community Cloud, create an app for `Shamma-Samiha/LemoScan-AI`.
2. Select the pushed `rebuild/robust-inceptionv3` branch and entrypoint `streamlit_app.py`.
3. Open Advanced settings and select Python **3.13**. Leave local metadata-path overrides unset.
4. Deploy manually and inspect build logs for dependency installation and successful LFS retrieval.
5. Repeat valid lemon, solid green, non-lemon and same-image rerun checks on the hosted app. Confirm both models load and monitor memory, including the first explanation request.
6. Share the app after hosted checks pass. If memory is exhausted, use hosting with sufficient RAM; do not change frozen model thresholds or evaluation claims to work around deployment failures.
