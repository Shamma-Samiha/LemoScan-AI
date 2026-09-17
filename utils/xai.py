"""Grad-CAM for an explicit previously predicted target; never selects a class."""
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from utils.model_loader import get_model
from utils.preprocessing import PreparedImage

def get_gradcam_layer():
    backbone = get_model().get_layer("inception_v3")
    # mixed10 concatenates the final convolution branches before global pooling.
    layer = backbone.get_layer("mixed10")
    if len(layer.output.shape) != 4 or backbone.output is not layer.output:
        raise ValueError("Expected mixed10 to be the backbone's final spatial feature map.")
    return backbone, layer

def generate_gradcam(prepared, target_class_index, output_path=None):
    """Explain the shared PreparedImage for the explicit prediction class.

    A differentiable forward pass is required for gradients. It never runs argmax.
    The original model's layers, activations and weights are not modified.
    """
    if not isinstance(prepared, PreparedImage):
        raise TypeError("Grad-CAM requires the PreparedImage used for prediction.")
    if isinstance(target_class_index, bool) or not isinstance(target_class_index, (int, np.integer)) or not 0 <= target_class_index < 6:
        raise ValueError("target_class_index must be an integer from 0 to 5.")
    model = get_model()
    backbone, layer = get_gradcam_layer()
    features = backbone(prepared.tensor, training=False)
    with tf.GradientTape() as tape:
        tape.watch(features)
        scores = features
        for head_layer in model.layers[model.layers.index(backbone) + 1:]:
            scores = head_layer(scores, training=False)
        target_score = scores[:, target_class_index]
    gradients = tape.gradient(target_score, features)
    if gradients is None or not bool(tf.reduce_all(tf.math.is_finite(gradients))):
        raise RuntimeError("Grad-CAM gradients are unavailable or non-finite.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(features * weights, axis=-1))[0]
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap)).numpy()
    if heatmap.ndim != 2 or min(heatmap.shape) < 1 or not np.isfinite(heatmap).all():
        raise RuntimeError("Invalid Grad-CAM heatmap.")
    result = {"status": "available", "target_class_index": int(target_class_index),
              "layer": f"{backbone.name}/{layer.name}", "heatmap": heatmap,
              "target_probability": float(scores[0, target_class_index]), "heatmap_shape": list(heatmap.shape), "all_zero": bool(not np.any(heatmap)), "output_path": None}
    if output_path is not None:
        image = prepared.rgb
        resized = np.asarray(Image.fromarray(heatmap).resize(image.size, Image.Resampling.BILINEAR))
        colors = np.stack([np.ones_like(resized), 1-resized, np.zeros_like(resized)], axis=-1) * 255
        alpha = 0.38 * resized[..., None]
        overlay = np.clip(np.asarray(image, dtype=np.float32)*(1-alpha) + colors*alpha, 0, 255).astype(np.uint8)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(path)
        result["output_path"] = str(path)
    return result
