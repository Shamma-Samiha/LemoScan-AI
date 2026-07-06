"""Load and share the trained Keras model across the application."""

from pathlib import Path
from threading import Lock


# Build the path from this file so it works locally and on Render.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "resnet_best_model.keras"

# These module-level values ensure the application has one shared model instance.
_model = None
_model_lock = Lock()


def get_model():
    """Load the trained model lazily, then reuse the same instance."""
    global _model

    if _model is None:
        # The lock prevents two simultaneous first requests loading two copies.
        with _model_lock:
            if _model is None:
                from tensorflow.keras.models import load_model

                if not MODEL_PATH.exists():
                    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

                # Training optimizer state is unnecessary for prediction.
                _model = load_model(MODEL_PATH, compile=False)

    return _model
