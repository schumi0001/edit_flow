import glob
import gzip
import json
import os
import pandas as pd
from typing import Optional


FEATURES_DIR = "data/lake/features"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.joblib")


def _read_jsonl_records(path: str) -> list[dict]:
    """Read a JSONL or gzipped JSONL file into a list of dictionaries."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Historical data file not found: {path}")

    opener = gzip.open if path.endswith(".gz") else open
    records = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)

    return records


def _normalize_record(record: dict) -> dict:
    """Map common Wikimedia field names to the columns expected by the feature builder."""
    normalized = dict(record)

    page_title = normalized.get("page_title") or normalized.get("title") or normalized.get("page")
    if page_title is not None:
        normalized["page_title"] = page_title

    user = normalized.get("user") or normalized.get("user_name")
    if user is not None:
        normalized["user"] = user

    bot = normalized.get("bot")
    if bot is None and "bot" not in normalized:
        normalized["bot"] = False

    minor = normalized.get("minor")
    if minor is None and "minor" not in normalized:
        normalized["minor"] = False

    old_length = normalized.get("old_length") or normalized.get("oldlen")
    new_length = normalized.get("new_length") or normalized.get("newlen")
    if old_length is not None and new_length is not None:
        normalized["byte_change"] = int(new_length) - int(old_length)
    elif "byte_change" not in normalized:
        normalized["byte_change"] = 0

    timestamp = normalized.get("timestamp") or normalized.get("ts") or normalized.get("time")
    if timestamp is None:
        normalized["timestamp"] = 0

    return normalized


def load_historical_features_from_jsonl(path: str) -> pd.DataFrame:
    """Build feature rows from a historical JSONL stream or gzipped archive of Wikimedia-style edit events."""
    records = _read_jsonl_records(path)
    if not records:
        raise ValueError(f"No records found in {path}")

    normalized_rows = [_normalize_record(record) for record in records]
    df = pd.DataFrame(normalized_rows)

    required_cols = {"page_title", "user", "bot", "minor", "byte_change", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Historical data is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["bot"] = df["bot"].fillna(False).astype(bool)
    df["minor"] = df["minor"].fillna(False).astype(bool)
    df["byte_change"] = pd.to_numeric(df["byte_change"], errors="coerce").fillna(0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    features = (
        df.groupby("page_title")
        .agg(
            edit_count=("page_title", "size"),
            unique_editors=("user", lambda s: s.nunique()),
            total_byte_changes=("byte_change", lambda s: abs(s).sum()),
            bot_ratio=("bot", lambda s: float(s.mean())),
            minor_edit_ratio=("minor", lambda s: float(s.mean())),
        )
        .reset_index()
    )

    features["relative_growth"] = features["total_byte_changes"] / features["edit_count"]
    features["human_bot_friction"] = features["minor_edit_ratio"] - features["bot_ratio"]
    features["editor_concentration"] = features["unique_editors"] / features["edit_count"]
    features = features.fillna(0.0)

    return features


def load_feature_lake_frames(features_dir: str = FEATURES_DIR) -> pd.DataFrame:
    """Load a feature lake from parquet files, if present."""
    pattern = os.path.join(features_dir, "**", "*.parquet")
    parquet_files = glob.glob(pattern, recursive=True)
    if not parquet_files:
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in parquet_files]
    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def train_from_history(jsonl_path: Optional[str] = None, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """Train a model from either historical JSONL data or the existing feature lake."""
    if jsonl_path:
        features_df = load_historical_features_from_jsonl(jsonl_path)
    else:
        features_df = load_feature_lake_frames()
        if features_df.empty:
            raise FileNotFoundError(
                f"No feature lake parquet files found in {FEATURES_DIR}."
            )

    feature_cols = [
        "edit_count",
        "unique_editors",
        "total_byte_changes",
        "bot_ratio",
        "minor_edit_ratio",
        "relative_growth",
        "human_bot_friction",
        "editor_concentration",
    ]
    X = features_df[feature_cols]

    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    import joblib

    pipeline = make_pipeline(
        RobustScaler(),
        IsolationForest(contamination=0.01, random_state=42, n_jobs=-1),
    )
    pipeline.fit(X)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Trained model saved to {model_path}")
    return features_df
