import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)

from src.config.loader import ConfigLoader
from src.data.data_transformation import RecommenderDataTransformation
from src.logger.logger import get_logger

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.save_model_path = self.config.get("model.save_path")
        # CF-Model Parameters
        self.cf_index = self.config.get("model.cf_model.index", "user_id")
        self.cf_column = self.config.get("model.cf_model.columns", "course_id")
        self.cf_values = self.config.get("model.cf_model.values", "rating")
        self.cf_n_components = self.config.get("model.cf_model.n_components", 100)
        self.cf_random_state = self.config.get("model.cf_model.random_state", 42)
        self.cf_max_rows = self.config.get("model.cf_model.max_rows", 10000)
        # CBF - Random Forest Model Parameters
        self.cbf_n_estimators = self.config.get(
            "model.random_regressor_model.n_estimators", 150
        )
        self.cbf_max_depth = self.config.get(
            "model.random_regressor_model.max_depth", 15
        )
        self.cbf_random_state = self.config.get(
            "model.random_regressor_model.random_state", 42
        )
        self.cbf_n_jobs = self.config.get("model.random_regressor_model.n_jobs", 1)
        # Mlflow tracking uri
        self.mlflow_tracking_uri = self.config.get(
            "mlflow.tracking_uri",
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        )
        self.mlflow_artifact_uri = self.config.get(
            "mlflow.artifact_uri", os.getenv("MLFLOW_ARTIFACT_URI", "./mlartifacts")
        )
        self.mlflow_experiment = self.config.get(
            "mlflow.experiment_name",
            os.getenv("MLFLOW_EXPERIMENT_NAME", "course_recommendation.dev"),
        )

    def implementing_cf_model(self, train_df: pd.DataFrame):
        logger.info("Starting the implementation of colaborative filtering model...")

        # Sample if dataset is too large to avoid OOM
        original_len = len(train_df)
        if original_len > self.cf_max_rows:
            logger.info(
                f"Downsampling dataset from {original_len} to {self.cf_max_rows} rows for CF model"
            )
            train_df = train_df.sample(
                n=self.cf_max_rows, random_state=self.cf_random_state
            )

        logger.info(f"Creating pivot table with {len(train_df)} rows...")
        pivot = train_df.pivot_table(
            index=self.cf_index, columns=self.cf_column, values=self.cf_values
        )
        logger.info(f"Pivot table shape: {pivot.shape}")
        if pivot.empty:
            logger.error("Collaborative filtering matrix is empty.")
            raise ValueError("Collaborative filer matrix is empty.")

        logger.info("Computing user means...")
        user_means = pivot.mean(axis=1)
        logger.info("Centering pivot matrix...")
        pivot_centered = pivot.sub(user_means, axis=0).fillna(0)

        n_components = min(
            self.cf_n_components,
            pivot_centered.shape[0] - 1,
            pivot_centered.shape[1] - 1,
        )
        logger.info(f"Using n_components={n_components} for SVD")

        if n_components < 1:
            logger.error("Not enought uses or courses for TruncatedSVD.")
            raise ValueError("Not enought uses or courses for TruncatedSVD.")

        logger.info(f"Fitting TruncatedSVD with {n_components} components...")
        svd = TruncatedSVD(
            n_components=n_components, random_state=self.cf_random_state, n_iter=100
        )
        logger.info("Starting SVD fit_transform...")
        latent = svd.fit_transform(pivot_centered)
        logger.info(f"SVD complete. Latent shape: {latent.shape}")

        logger.info("Reconstructing matrix...")
        reconstructed = np.dot(latent, svd.components_)

        logger.info("Creating prediction matrix...")
        pred_matrix = pd.DataFrame(
            reconstructed, index=pivot.index, columns=pivot.columns
        ).add(user_means, axis=0)

        logger.info("Successfully trained colaborative filtering model.....")

        return {"svd": svd, "pred_matrix": pred_matrix, "user_means": user_means}

    def evaluate_cf_model(self, cf_model, test_df: pd.DataFrame):
        # Filtering test set for users and courses present in the training matrix
        logger.info("Evaluating  CF Model.")

        pred_matrix = cf_model["pred_matrix"]

        mask = test_df[self.cf_index].isin(pred_matrix.index) & test_df[
            self.cf_column
        ].isin(pred_matrix.columns)
        eval_df = test_df[mask].copy()

        if eval_df.empty:
            logger.error("No common users/courses found between train and test data.")
            return 0.0, 0.0

        user_indices = pred_matrix.index.get_indexer(eval_df[self.cf_index])
        course_indices = pred_matrix.columns.get_indexer(eval_df[self.cf_column])
        eval_df["pred"] = pred_matrix.to_numpy()[user_indices, course_indices]

        mse = mean_squared_error(eval_df["rating"], eval_df["pred"])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(eval_df["rating"], eval_df["pred"])
        logger.info(f"CF Model Metrices values --> RMSE:{rmse}, MAE:{mae}")

        return float(rmse), float(mae)

    def implementing_randomforest_cbf(self, X_train, y_train):
        logger.info(
            "Starting the implementation of contentet based filtering model using RandomForestRegressor model."
        )

        cbf_model = RandomForestRegressor(
            n_estimators=self.cbf_n_estimators,
            max_depth=self.cbf_max_depth,
            random_state=self.cbf_random_state,
            n_jobs=self.cbf_n_jobs,
        )

        cbf_model.fit(X_train, y_train)

        logger.info("Successfully trained the CBF Model using RandomForestRegressor.")

        return cbf_model

    def evaluate_cbf_model(self, cbf_model, X_test, y_test):
        logger.info("Evaluating CBF Model")
        cbf_preds = cbf_model.predict(X_test)

        cbf_rmse = root_mean_squared_error(y_test, cbf_preds)
        cbf_mae = mean_absolute_error(y_test, cbf_preds)

        logger.info(
            f"CBF RandomForestRegressor Model Metrices --> RMSE:{cbf_rmse}, MAE:{cbf_mae}"
        )

        return float(cbf_rmse), float(cbf_mae)

    def _save_final_model(self, model_bundle):
        model_path = self.save_model_path

        if not os.path.splitext(model_path)[1]:
            model_path = os.path.join(model_path, "recommender_model.joblib")

        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(model_bundle, model_path)

        return model_path

    def _register_model_version(
        self, model_name: str, model_uri: str, run_id: str, stage: str = "Staging"
    ):
        client = MlflowClient()

        try:
            registered_model = client.get_registered_model(model_name)
        except mlflow.exceptions.RestException:
            registered_model = None

        if registered_model is None:
            client.create_registered_model(model_name)

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage,
        )

        return {
            "name": model_name,
            "version": model_version.version,
            "stage": stage,
            "run_id": run_id,
            "source": model_uri,
        }

    def compairing_to_find_best_model(self):
        logger.info("Starting model selection and training process...")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.mlflow_experiment)
        if experiment and experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
        mlflow.set_experiment(self.mlflow_experiment)

        logger.info("Loading and transforming data...")
        transformation = RecommenderDataTransformation(self.config)
        (
            X_train,
            y_train,
            X_test,
            y_test,
            train_df,
            test_df,
            preprocessor,
            feature_cols,
            num_cols,
            cate_cols,
        ) = transformation.run_transformation_pipeline()
        logger.info(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")

        with mlflow.start_run(run_name="model_compairison") as run:
            mlflow.log_params(
                {
                    "cf_n_components": self.cf_n_components,
                    "cf_random_state": self.cf_random_state,
                    "cbf_n_estimators": self.cbf_n_estimators,
                    "cbf_max_depth": self.cbf_max_depth,
                    "cbf_random_state": self.cbf_random_state,
                    "cf_max_rows": self.cf_max_rows,
                }
            )

            logger.info("Training CF model...")
            cf_model = self.implementing_cf_model(train_df)
            logger.info("Evaluating CF model...")
            cf_rmse, cf_mae = self.evaluate_cf_model(cf_model, test_df)

            mlflow.log_metrics({"cf_rmse": cf_rmse, "cf_mae": cf_mae})

            logger.info("Logging CF model to MLflow...")
            mlflow.sklearn.log_model(cf_model["svd"], "collaborative_filtering_svd")

            logger.info("Training CBF model...")
            cbf_model = self.implementing_randomforest_cbf(X_train, y_train)
            logger.info("Evaluating CBF model...")
            cbf_rmse, cbf_mae = self.evaluate_cbf_model(cbf_model, X_test, y_test)

            mlflow.log_metrics({"cbf_rmse": cbf_rmse, "cbf_mae": cbf_mae})
            logger.info("Logging CBF model to MLflow...")
            mlflow.sklearn.log_model(cbf_model, "content_based_random_forest")

            results = {
                "cf": {"rmse": cf_rmse, "mae": cf_mae},
                "cbf": {"rmse": cbf_rmse, "mae": cbf_mae},
            }

            best_model_name = min(
                results, key=lambda name: (results[name]["rmse"], results[name]["mae"])
            )
            logger.info("Best model selected: %s", best_model_name)

            # Retraining the selected model using the complete dataset
            logger.info("Combining train and test sets for final model...")
            full_interaction = pd.concat([train_df, test_df], ignore_index=True)

            logger.info("Training CF model on full dataset (sampled)...")
            cf_full_model = self.implementing_cf_model(full_interaction)
            cf_bundle = {
                "model_type": "collaborative_filtering",
                "model": cf_full_model,
                "pred_matrix": cf_full_model["pred_matrix"],
                "user_means": cf_full_model["user_means"],
            }

            logger.info("Training CBF model on full dataset...")
            X_full = (
                sparse.vstack([X_train, X_test])
                if sparse.issparse(X_train) or sparse.issparse(X_test)
                else np.vstack([X_train, X_test])
            )
            y_full = np.concatenate([y_train, y_test])
            cbf_full_model = self.implementing_randomforest_cbf(X_full, y_full)
            cbf_bundle = {
                "model_type": "content_based_random_forest",
                "model": cbf_full_model,
                "preprocessor": preprocessor,
                "feature_cols": feature_cols,
                "num_cols": num_cols,
                "cate_cols": cate_cols,
                "target_col": self.config.get("data.target_col", "rating"),
            }

            logger.info("Creating final model bundle...")
            final_bundle = {
                "selected_model": best_model_name,
                "models": {
                    "cf": cf_bundle,
                    "cbf": cbf_bundle,
                },
                "metadata": {
                    "project_name": self.config.get(
                        "project.name", "mlops_recommendation_system"
                    ),
                    "environment": self.config.get("project.environment", "dev"),
                    "mlflow_experiment": self.mlflow_experiment,
                    "mlflow_run_id": run.info.run_id,
                },
            }

            logger.info("Saving final model...")
            model_path = self._save_final_model(final_bundle)

            mlflow.log_param("selected_model", best_model_name)
            mlflow.log_param("final_model_path", model_path)
            mlflow.log_artifact(model_path, artifact_path="final_model")

            logger.info("Registering model in MLflow...")
            registry_name = self.config.get(
                "mlflow.model_registry_name", "course_recommendation"
            )
            target_stage = self.config.get("mlflow.model_stage", "Staging")
            model_uri = f"runs:/{run.info.run_id}/{best_model_name}"
            registered_version = self._register_model_version(
                model_name=registry_name,
                model_uri=model_uri,
                run_id=run.info.run_id,
                stage=target_stage,
            )

            logger.info("Final model saved to: %s", model_path)
            logger.info("Registered model version: %s", registered_version)

            return {
                "selected_model": best_model_name,
                "metrics": results,
                "model_path": model_path,
                "mlflow_run_id": run.info.run_id,
                "registered_model": registered_version,
            }


if __name__ == "__main__":
    config = ConfigLoader()
    trainer = ModelTraining(config)
    result = trainer.compairing_to_find_best_model()
    print(result)
