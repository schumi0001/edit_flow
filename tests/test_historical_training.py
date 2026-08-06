import gzip
import os
import tempfile
import json
import unittest

from models.historical_training import load_historical_features_from_jsonl


class HistoricalTrainingTests(unittest.TestCase):
    def test_load_historical_features_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({
                        "page_title": "Alpha",
                        "user": "u1",
                        "bot": False,
                        "minor": False,
                        "byte_change": 120,
                        "timestamp": 1,
                    }) + "\n"
                )
                handle.write(
                    json.dumps({
                        "page_title": "Alpha",
                        "user": "u2",
                        "bot": True,
                        "minor": True,
                        "byte_change": -80,
                        "timestamp": 2,
                    }) + "\n"
                )
                handle.write(
                    json.dumps({
                        "page_title": "Beta",
                        "user": "u3",
                        "bot": False,
                        "minor": False,
                        "byte_change": 300,
                        "timestamp": 3,
                    }) + "\n"
                )

            df = load_historical_features_from_jsonl(path)

            self.assertEqual(len(df), 2)
            self.assertIn("page_title", df.columns)
            self.assertIn("edit_count", df.columns)
            self.assertIn("relative_growth", df.columns)
            self.assertEqual(int(df.loc[df["page_title"] == "Alpha", "edit_count"].iloc[0]), 2)
            self.assertGreater(df.loc[df["page_title"] == "Alpha", "bot_ratio"].iloc[0], 0.0)

    def test_load_historical_features_from_gzip_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.jsonl.gz")
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({
                        "title": "Gamma",
                        "user": "u4",
                        "bot": False,
                        "minor": False,
                        "oldlen": 100,
                        "newlen": 180,
                        "timestamp": 4,
                    }) + "\n"
                )
                handle.write(
                    json.dumps({
                        "title": "Gamma",
                        "user": "u5",
                        "bot": True,
                        "minor": True,
                        "oldlen": 180,
                        "newlen": 220,
                        "timestamp": 5,
                    }) + "\n"
                )

            df = load_historical_features_from_jsonl(path)

            self.assertEqual(len(df), 1)
            self.assertEqual(int(df.loc[df["page_title"] == "Gamma", "edit_count"].iloc[0]), 2)


if __name__ == "__main__":
    unittest.main()
