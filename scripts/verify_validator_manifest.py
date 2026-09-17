"""Rebuild the seeded manifest in memory; never write or modify image files."""
from pathlib import Path
from unittest.mock import patch
import io
import pandas as pd
from PIL import Image
import prepare_leaf_validator_data as data
manifest_path=data.OUT/"validator_dataset_manifest.csv"
existing=pd.read_csv(manifest_path,keep_default_na=False)
real_exists=Path.exists
real_image_save=Image.Image.save
def exists(path):
    return False if path==manifest_path else real_exists(path)
def write_bytes(path, value):
    assert path.read_bytes()==value, f"Rebuild changed an external image: {path}"
    return len(value)
def image_save(image, path, *args, **kwargs):
    buffer=io.BytesIO()
    real_image_save(image,buffer,format="PNG")
    assert Path(path).read_bytes()==buffer.getvalue(), f"Rebuild changed a synthetic image: {path}"
with patch.object(Path,"exists",exists), patch.object(Path,"write_bytes",write_bytes), patch.object(Path,"write_text",return_value=0), patch.object(Image.Image,"save",image_save), patch.object(pd.DataFrame,"to_csv"):
    rebuilt=data.build_dataset()
pd.testing.assert_frame_equal(existing,rebuilt,check_dtype=False)
print("PASS: Fresh seed-42 reconstruction matches all 4,100 manifest rows; no files changed.")

