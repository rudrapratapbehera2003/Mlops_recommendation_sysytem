from types import SimpleNamespace

from src.models.train_model import ModelTraining


class DummyConfig:
    def __init__(self):
        self.values = {
            "model.save_path": "models/recommender_model.joblib",
            "model.cf_model.index": "user_id",
            "model.cf_model.columns": "course_id",
            "model.cf_model.values": "rating",
            "model.cf_model.n_components": 100,
            "model.cf_model.random_state": 42,
            "model.random_regressor_model.n_estimators": 150,
            "model.random_regressor_model.max_depth": 15,
            "model.random_regressor_model.random_state": 42,
            "model.random_regressor_model.n_jobs": 1,
            "mlflow.tracking_uri": "http://localhost:5000",
            "mlflow.artifact_uri": "./mlartifacts",
            "mlflow.experiment_name": "course_recommendation.dev",
            "mlflow.model_registry_name": "course_recommendation",
            "mlflow.model_stage": "Staging",
        }

    def get(self, key, default=None):
        return self.values.get(key, default)


def test_register_model_version_promotes_to_staging(monkeypatch):
    trainer = ModelTraining(DummyConfig())
    captured = {}

    class FakeClient:
        def get_registered_model(self, name):
            return None

        def create_registered_model(self, name):
            captured["registered_model"] = name
            return {"name": name}

        def create_model_version(self, name, source, run_id=None):
            captured["version"] = {"name": name, "source": source, "run_id": run_id}
            return SimpleNamespace(name=name, version="1")

        def transition_model_version_stage(self, name, version, stage):
            captured["stage"] = {"name": name, "version": version, "stage": stage}
            return SimpleNamespace()

    monkeypatch.setattr("src.models.train_model.MlflowClient", lambda: FakeClient())
    monkeypatch.setattr("src.models.train_model.mlflow.set_tracking_uri", lambda uri: None)
    monkeypatch.setattr("src.models.train_model.mlflow.set_experiment", lambda name: None)
    monkeypatch.setattr("src.models.train_model.mlflow.sklearn.log_model", lambda model, artifact_path: None)

    version = trainer._register_model_version(
        model_name="course_recommendation",
        model_uri="runs:/abc123/content_based_random_forest",
        run_id="abc123",
        stage="Staging",
    )

    assert version["stage"] == "Staging"
    assert captured["registered_model"] == "course_recommendation"
    assert captured["stage"]["name"] == "course_recommendation"
