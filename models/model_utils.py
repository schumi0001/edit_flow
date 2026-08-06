import os
import joblib
from typing import Optional

DEFAULT_MODEL_PATH = os.path.join("models", "anomaly_detector.joblib")


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolve the model path from an explicit argument, environment override, or default."""
    if model_path:
        return model_path

    env_path = os.getenv("ANOMALY_MODEL_PATH")
    if env_path:
        return env_path

    return DEFAULT_MODEL_PATH


def load_model(model_path: Optional[str] = None):
    """Load a trained model from disk using the shared resolution logic."""
    resolved_path = resolve_model_path(model_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Trained model not found at {resolved_path}. Run train_model.py first."
        )
    return joblib.load(resolved_path)
