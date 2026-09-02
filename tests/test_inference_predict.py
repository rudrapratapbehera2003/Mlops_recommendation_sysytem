import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.api.main as api_main
from src.inference.predict import load_model, predict_recommendations


class DummyPredictor:
    def predict(self, X):
        return [1.0, 2.0, 3.0]


class DummyPreprocessor:
    def transform(self, X):
        return X


def test_load_model_manifest(tmp_path):
    bundle = {
        "selected_model": "content_based_random_forest",
        "models": {
            "cbf": {
                "model_type": "content_based_random_forest",
                "model": {"value": 42},
            },
            "cf": {"model_type": "collaborative_filtering", "model": {"value": 84}},
        },
    }
    model_path = tmp_path / "dummy_model.joblib"
    joblib.dump(bundle, model_path)

    loaded = load_model(str(model_path))
    assert loaded["selected_model"] == "content_based_random_forest"


def test_api_module_imports():
    assert hasattr(api_main, "app")
    assert api_main.app.title == "Course Recommendation API"


def test_predict_recommendations_returns_payload(tmp_path):
    bundle = {
        "selected_model": "content_based_random_forest",
        "models": {
            "cbf": {
                "model_type": "content_based_random_forest",
                "model": DummyPredictor(),
                "preprocessor": DummyPreprocessor(),
                "feature_cols": ["course_id"],
            },
            "cf": {
                "model_type": "collaborative_filtering",
                "model": {"pred_matrix": None},
            },
        },
    }
    model_path = tmp_path / "dummy_model.joblib"
    joblib.dump(bundle, model_path)

    result = predict_recommendations(
        user_id=10,
        top_n=3,
        model_path=str(model_path),
        model_type="content_based_random_forest",
        candidate_features=[{"course_id": 1}, {"course_id": 2}, {"course_id": 3}],
    )

    assert result["user_id"] == 10
    assert result["top_n"] == 3
    assert result["model_type"] == "content_based_random_forest"
    assert isinstance(result["recommendations"], list)
