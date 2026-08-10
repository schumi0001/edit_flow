import gzip
import os
import tempfile
import json
import unittest

from models.historical_training import WINDOW_SECONDS, load_historical_features_from_jsonl


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

    def test_edits_in_different_windows_are_not_merged(self):
        """Two edits to the same page far apart in time should become two
        separate feature rows (one per window), not a single lifetime-total
        row -- this is what keeps training features on the same time scale
        as the live 15-minute-window inference features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({
                        "page_title": "Delta",
                        "user": "u1",
                        "bot": False,
                        "minor": False,
                        "byte_change": 100,
                        "timestamp": 0,
                    }) + "\n"
                )
                handle.write(
                    json.dumps({
                        "page_title": "Delta",
                        "user": "u2",
                        "bot": False,
                        "minor": False,
                        "byte_change": 200,
                        "timestamp": WINDOW_SECONDS * 10,
                    }) + "\n"
                )

            df = load_historical_features_from_jsonl(path)

            delta_rows = df[df["page_title"] == "Delta"]
            self.assertEqual(len(delta_rows), 2)
            self.assertTrue((delta_rows["edit_count"] == 1).all())

    def test_edits_within_same_window_are_merged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                for i in range(3):
                    handle.write(
                        json.dumps({
                            "page_title": "Epsilon",
                            "user": f"u{i}",
                            "bot": False,
                            "minor": False,
                            "byte_change": 10,
                            "timestamp": i,
                        }) + "\n"
                    )

            df = load_historical_features_from_jsonl(path)

            epsilon_rows = df[df["page_title"] == "Epsilon"]
            self.assertEqual(len(epsilon_rows), 1)
            self.assertEqual(int(epsilon_rows["edit_count"].iloc[0]), 3)


if __name__ == "__main__":
    unittest.main()
