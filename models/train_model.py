import argparse
import os
import pandas as pd

try:
    from .historical_training import train_from_history
    from .model_utils import resolve_model_path
except ImportError:  # pragma: no cover - allows running the file directly
    from historical_training import train_from_history
    from model_utils import resolve_model_path

FEATURES_DIR = "data/lake/features"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.joblib")


def main():
    parser = argparse.ArgumentParser(description="Train the anomaly detection model")
    parser.add_argument(
        "--history-jsonl",
        dest="history_jsonl",
        help="Optional path to a historical Wikimedia-style JSONL file for offline training",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Where to save the trained model. Defaults to the shared model path or ANOMALY_MODEL_PATH.",
    )
    args = parser.parse_args()
    model_path = resolve_model_path(args.model_path)

    if args.history_jsonl:
        print(f"Training from historical file: {args.history_jsonl}")
        df = train_from_history(args.history_jsonl, model_path=model_path)
    else:
        print("Training from the local feature lake parquet files...")
        df = train_from_history(model_path=model_path)

    df = df.drop_duplicates(subset=["page_title"]).dropna()
    print(f"Loaded {len(df)} pages for training.")

    print("\n=== LOCAL DETECTION TEST RESULTS ===")
    print(df[["page_title", "edit_count", "total_byte_changes"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
