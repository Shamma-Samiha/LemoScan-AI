# Streamlit UI and deployment readiness

## Verified local environment
Python 3.13.2; TensorFlow 2.21.0; Keras 3.14.1; Streamlit 1.49.0;
NumPy 2.3.4; Pillow 11.3.0; protobuf 6.33.5; h5py 3.14.0.

## Existing deployment mismatch
- requirements.txt pins tensorflow-cpu 2.20.0, NumPy 2.1.3, Pillow 10.4.0 and protobuf 5.29.5.
- runtime.txt requests Python 3.10.13.
- These are not the environment that loaded and verified the two frozen models.
- TensorFlow 2.21.0 requires protobuf >=6.31.1,<8 and Keras >=3.12.0, according to installed package metadata. Streamlit 1.49.0 permits protobuf <7. The verified protobuf 6.33.5 meets both.
- Pin Keras 3.14.1 to match model serialization. Pin h5py 3.14.0 to match the verified .keras weight loader.
- Runtime inference uses TensorFlow/Keras, NumPy and Pillow. It does not need TFDS, scikit-learn, notebooks or training data. No new inference dependency was added for this UI.
- Flask is not part of the primary Streamlit deployment requirements. A separate Flask deployment would also need Flask/Werkzeug.

## Exact recommended requirements.txt
Use requirements.recommended.txt in this folder as the reviewed replacement:
```text
streamlit==1.49.0
tensorflow-cpu==2.21.0
keras==3.14.1
numpy==2.3.4
Pillow==11.3.0
protobuf==6.33.5
h5py==3.14.0
```
The existing requirements.txt and runtime.txt have deliberately not been changed.
No packages were installed in the working environment.

For a host that consumes runtime.txt, replace its contents with:
```text
python-3.13.2
```
For Streamlit Community Cloud, select Python **3.13** in Advanced settings rather than relying on runtime.txt; the platform controls the patch version. See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

The [PyPI tensorflow-cpu 2.21.0 release metadata](https://pypi.org/pypi/tensorflow-cpu/2.21.0/json) was checked: a cp313 manylinux_2_27_x86_64 wheel exists. This is an availability check, not a clean Linux installation test.

## Required deployment artifacts
- streamlit_app.py, .streamlit/config.toml and utils/
- model/inceptionv3_best_model.keras (137,022,324 bytes)
- model/model_metadata.json
- training/inceptionv3_class_indices.json
- model/lemon_leaf_validator.keras (4,336,307 bytes)
- model/lemon_leaf_validator_metadata.json

The repository already marks *.keras for Git LFS. Ensure the host materializes both real model binaries, not LFS pointer text. The running application does not need source dataset folders, validator download archives, or training notebooks.
The larger model and TensorFlow memory footprint still require a target-host cold-start/resource check.

## Readiness
Local functionality and responsive browser checks are recorded alongside this report. Before deployment, apply the reviewed pins, choose the Python version, install into a clean target environment, confirm both models load from LFS, and repeat the short upload smoke checks there. No deployment was performed.

