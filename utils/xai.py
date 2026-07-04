"""Explain model predictions with Grad-CAM."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from utils.prediction import get_model


IMAGE_SIZE = (224, 224)


def _output_rank(layer):
    """Return a layer's output rank, or None when it cannot be determined."""
    try:
        return len(layer.output.shape)
    except (AttributeError, TypeError):
        return None


def _find_last_convolutional_layer(model):
    """Find the last layer that produces a convolutional feature map.

    ResNet50 is nested inside this classifier and its output is the final 7x7
    convolutional feature map. This also supports models whose convolutional
    layers are stored directly in the classifier.
    """
    for layer in reversed(model.layers):
        if _output_rank(layer) == 4:
            return layer

    raise ValueError("Could not find a convolutional feature layer for Grad-CAM.")


def _heatmap_to_rgb(heatmap):
    """Convert a 0-1 heatmap to a simple yellow-to-red RGB colour map."""
    red = np.ones_like(heatmap)
    green = 1.0 - heatmap
    blue = np.zeros_like(heatmap)
    return np.stack([red, green, blue], axis=-1) * 255.0


def _preprocess_image(image_path):
    """Resize and preprocess an image exactly as ResNet50 expects."""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)


def _resize_and_focus_heatmap(heatmap, image_size):
    """Smooth the heatmap and keep only its stronger activation regions."""
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255))

    # Bicubic resizing and a light blur remove blockiness from the 7x7 map.
    heatmap_image = heatmap_image.resize(image_size, Image.Resampling.BICUBIC)
    blur_radius = max(2.0, min(image_size) * 0.012)
    heatmap_image = heatmap_image.filter(ImageFilter.GaussianBlur(blur_radius))
    smooth_heatmap = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    # Suppress weak responses that commonly colour irrelevant background areas.
    positive_values = smooth_heatmap[smooth_heatmap > 0]
    if positive_values.size == 0:
        return np.zeros_like(smooth_heatmap)

    threshold = np.percentile(positive_values, 65)
    focused_heatmap = np.clip(
        (smooth_heatmap - threshold) / (smooth_heatmap.max() - threshold + 1e-8),
        0,
        1,
    )

    # Smoothstep keeps activation boundaries soft instead of producing hard edges.
    return focused_heatmap**2 * (3.0 - 2.0 * focused_heatmap)


def generate_gradcam(image_path, output_path):
    """Generate and save a Grad-CAM overlay for the predicted class."""
    model = get_model()
    last_conv_layer = _find_last_convolutional_layer(model)

    # Apply the exact preprocessing expected by ResNet50.
    processed_image = _preprocess_image(image_path)

    with tf.GradientTape() as tape:
        if isinstance(last_conv_layer, tf.keras.Model):
            # Keras 3 requires a nested ResNet and its classifier head to be
            # called explicitly so gradients stay connected to the feature map.
            feature_maps = last_conv_layer(processed_image, training=False)
            predictions = feature_maps
            target_was_reached = False

            for layer in model.layers:
                if target_was_reached:
                    predictions = layer(predictions, training=False)
                elif layer is last_conv_layer:
                    target_was_reached = True
        else:
            # Ordinary functional models can expose a convolutional layer.
            gradient_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output],
            )
            feature_maps, predictions = gradient_model(
                processed_image, training=False
            )

        predicted_index = tf.argmax(predictions[0])
        predicted_score = predictions[:, predicted_index]

    # Average the gradients to measure the importance of each feature channel.
    gradients = tape.gradient(predicted_score, feature_maps)
    if gradients is None:
        raise RuntimeError("Grad-CAM could not calculate gradients for this model.")

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()

    with Image.open(image_path) as image:
        original_image = image.convert("RGB")
        original_array = np.asarray(original_image, dtype=np.float32)

    focused_heatmap = _resize_and_focus_heatmap(heatmap.numpy(), original_image.size)
    heatmap_rgb = _heatmap_to_rgb(focused_heatmap)

    # Lower, activation-based opacity keeps the leaf clearly visible underneath.
    alpha = (0.38 * focused_heatmap)[..., np.newaxis]
    overlay = original_array * (1.0 - alpha) + heatmap_rgb * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)

    return str(output_path)
