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
DEFAULT_HISTORY_JSONL = "data/historical/en.wikipedia.org.recentchanges.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Train the anomaly detection model")
    parser.add_argument(
        "--history-jsonl",
        dest="history_jsonl",
        default=DEFAULT_HISTORY_JSONL,
        help=(
            "Path to a historical Wikimedia-style JSONL file for offline "
            f"training (default: {DEFAULT_HISTORY_JSONL}, produced by "
            "scripts/download_wikimedia_history.py)"
        ),
    )
    parser.add_argument(
        "--feature-lake",
        action="store_true",
        help=(
            "Train from the local live-data feature lake "
            f"({FEATURES_DIR}) instead of historical JSONL. Not the "
            "default: that lake is built from whatever this pipeline "
            "instance has streamed through the live wikipedia-edits "
            "topic, which may include test/synthetic events published "
            "there during development."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Where to save the trained model. Defaults to the shared model path or ANOMALY_MODEL_PATH.",
    )
    args = parser.parse_args()
    model_path = resolve_model_path(args.model_path)

    if args.feature_lake:
        print("Training from the local feature lake parquet files...")
        df = train_from_history(model_path=model_path)
    else:
        if not os.path.exists(args.history_jsonl):
            raise FileNotFoundError(
                f"Historical data file not found: {args.history_jsonl}\n"
                "Run scripts/download_wikimedia_history.py first, or pass "
                "--history-jsonl to point at a different file."
            )
        print(f"Training from historical file: {args.history_jsonl}")
        df = train_from_history(args.history_jsonl, model_path=model_path)

    df = df.drop_duplicates(subset=["page_title"]).dropna()
    print(f"Loaded {len(df)} pages for training.")

    print("\n=== LOCAL DETECTION TEST RESULTS ===")
    print(df[["page_title", "edit_count", "total_byte_changes"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
