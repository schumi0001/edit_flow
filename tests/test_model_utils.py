import os
import tempfile
import unittest

from models.model_utils import resolve_model_path, load_model


class ModelUtilsTests(unittest.TestCase):
    def test_resolve_model_path_prefers_explicit_path(self):
        self.assertEqual(resolve_model_path("/tmp/custom.joblib"), "/tmp/custom.joblib")

    def test_resolve_model_path_uses_environment_override(self):
        old = os.environ.get("ANOMALY_MODEL_PATH")
        os.environ["ANOMALY_MODEL_PATH"] = "/tmp/env.joblib"
        try:
            self.assertEqual(resolve_model_path(), "/tmp/env.joblib")
        finally:
            if old is None:
                os.environ.pop("ANOMALY_MODEL_PATH", None)
            else:
                os.environ["ANOMALY_MODEL_PATH"] = old

    def test_load_model_uses_resolved_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            import joblib
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import RobustScaler
            from sklearn.ensemble import IsolationForest

            model = make_pipeline(
                RobustScaler(),
                IsolationForest(contamination=0.01, random_state=42)
            )
            joblib.dump(model, path)
            loaded = load_model(path)
            self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
