#!/usr/bin/env python3
"""Download a Wikimedia recent-changes archive, normalize it, and train the model.

This script is intentionally simple:
- it downloads a gzipped JSONL archive from Wikimedia's recent changes feed;
- it writes a normalized JSONL file compatible with the historical training loader;
- it then trains the anomaly model from that file.
"""

import argparse
import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.historical_training import load_historical_features_from_jsonl
from models.train_model import main as train_main


def download_archive(url: str, output_path: Path) -> Path:
    print(f"Downloading {url} -> {output_path}")
    urllib.request.urlretrieve(url, output_path)
    return output_path


def normalize_archive(input_path: Path, output_path: Path) -> Path:
    print(f"Normalizing {input_path} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(input_path, "rt", encoding="utf-8") as src:
        with open(output_path, "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(payload, dict):
                    continue

                normalized = {
                    "page_title": payload.get("title") or payload.get("page_title") or payload.get("page"),
                    "user": payload.get("user") or payload.get("user_name"),
                    "bot": bool(payload.get("bot", False)),
                    "minor": bool(payload.get("minor", False)),
                    "timestamp": payload.get("timestamp") or payload.get("ts") or payload.get("time") or 0,
                }

                old_length = payload.get("old_length") or payload.get("oldlen")
                new_length = payload.get("new_length") or payload.get("newlen")
                if old_length is not None and new_length is not None:
                    normalized["byte_change"] = int(new_length) - int(old_length)
                else:
                    normalized["byte_change"] = 0

                dst.write(json.dumps(normalized) + "\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and train from a Wikimedia recent-changes archive")
    parser.add_argument(
        "--url",
        default="https://stream.wikimedia.org/v2/stream/recentchange",
        help="Wikimedia recent changes stream URL. This script expects a gzipped JSONL archive if you use a file URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "historical"),
        help="Directory where the normalized history file will be written",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional output model path",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / "recentchanges.jsonl.gz"
    normalized_path = output_dir / "recentchanges.normalized.jsonl"

    if not archive_path.exists():
        download_archive(args.url, archive_path)

    normalize_archive(archive_path, normalized_path)
    print(f"Normalized history written to {normalized_path}")

    # Train directly from the normalized file.
    import subprocess
    command = [sys.executable, str(ROOT / "models" / "train_model.py"), "--history-jsonl", str(normalized_path)]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    subprocess.run(command, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
