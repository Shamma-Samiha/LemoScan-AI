"""Shared RGB decode and resize for prediction and explanations.

Pillow decodes uploads; float32 TensorFlow bilinear resize without antialiasing
matches validation resize. JPEG decoder implementations can differ slightly.
No EXIF rotation is applied, matching the training pipeline.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from PIL import Image, UnidentifiedImageError
from utils.model_loader import get_image_size, get_preprocess_function

SUPPORTED_FORMATS = {"JPEG", "PNG"}
MAX_PIXELS = 40_000_000

class ImageValidationError(ValueError):
    """Technical image failure; does not imply a semantic non-leaf decision."""

@dataclass(frozen=True)
class PreparedImage:
    rgb: Image.Image
    tensor: object

def _decode(image):
    if image.format is not None and image.format not in SUPPORTED_FORMATS:
        raise ImageValidationError("Supported image formats are JPEG and PNG.")
    width, height = image.size
    if min(width, height) < 1 or width * height > MAX_PIXELS:
        raise ImageValidationError(f"Image dimensions must be nonempty and at most {MAX_PIXELS:,} pixels.")
    image.load()
    return image.convert("RGB").copy()

def prepare_image(source):
    """Accept Pillow image, path or binary file object; technical validation only."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            if isinstance(source, Image.Image):
                rgb = _decode(source)
            else:
                with Image.open(source) as image:
                    rgb = _decode(image)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("Image cannot be decoded safely or is corrupted.") from exc
    import tensorflow as tf
    array = tf.image.resize(np.asarray(rgb, dtype=np.float32), get_image_size(), method="bilinear", antialias=False)
    return PreparedImage(rgb=rgb, tensor=get_preprocess_function()(array)[None, ...])
