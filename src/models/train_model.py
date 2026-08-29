import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import numpy as np
import mlflow
import joblib
from scipy import sparse
from mlflow.tracking import MlflowClient

from src.logging.logger import get_logger
from src.config.loader import ConfigLoader
from src.data.data_transformation import RecommenderDataTransformation
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from src.data.data_transformation import RecommenderDataTransformation
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.save_model_path = self.config.get("model.save_path")
        #CF-Model Parameters
        self.cf_index = self.config.get("model.cf_model.index","user_id")
        self.cf_column = self.config.get("model.cf_model.columns","course_id")
        self.cf_values = self.config.get("model.cf_model.values","rating")
        self.cf_n_components = self.config.get("model.cf_model.n_components", 100)
        self.cf_random_state = self.config.get("model.cf_model.random_state", 42)
        #CBF - Random Forest Model Parameters
        self.cbf_n_estimators = self.config.get("model.random_regressor_model.n_estimators", 150)
        self.cbf_max_depth = self.config.get("model.random_regressor_model.max_depth", 15)
        self.cbf_random_state = self.config.get("model.random_regressor_model.random_state", 42)
        self.cbf_n_jobs = self.config.get("model.random_regressor_model.n_jobs", 1)
        # Mlflow tracking uri
        self.mlflow_tracking_uri = self.config.get("mlflow.tracking_uri", None)
        self.mlflow_experiment = self.config.get("mlflow.experiment_name","Course_Recommendation_System")
        
        
    def implementing_cf_model(self, train_df:pd.DataFrame):
        logger.info("Starting the implementation of colaborative filtering model...")
        
        pivot = train_df.pivot_table(
            index=self.cf_index,
            columns=self.cf_column,
            values=self.cf_values
        )
        if pivot.empty:
            logger.error("Collaborative filtering matrix is empty.")
            raise ValueError("Collaborative filer matrix is empty.")
        
        user_means = pivot.mean(axis=1)
        pivot_centered = pivot.sub(user_means,axis=0).fillna(0)
        
        n_components = min(
            self.cf_n_components,
            pivot_centered.shape[0] - 1,
            pivot_centered.shape[1] -1 
        )
        
        if n_components < 1:
            logger.error("Not enought uses or courses for TruncatedSVD.")
            raise ValueError("Not enought uses or courses for TruncatedSVD.")
        
        
        svd  = TruncatedSVD(n_components=n_components, random_state=self.cf_random_state)
        
        latent = svd.fit_transform(pivot_centered)
        
        reconstructed = np.dot(latent, svd.components_)
        
        pred_matrix = pd.DataFrame(
            reconstructed,
            index=pivot.index,
            columns=pivot.columns
        ).add(user_means, axis=0)
        
        logger.info("Successfully trained colaborative filtering model.....")
        
        return {
            "svd": svd,
            "pred_matrix": pred_matrix,
            "user_means": user_means
        }
    
    def evaluate_cf_model(self, cf_model, test_df:pd.DataFrame):
        # Filtering test set for users and courses present in the training matrix
        logger.info("Evaluating  CF Model.")
    
        pred_matrix = cf_model["pred_matrix"]
         
        mask = test_df[self.cf_index].isin(pred_matrix.index) & \
            test_df[self.cf_column].isin(pred_matrix.columns)
        eval_df = test_df[mask].copy()
        
        if eval_df.empty:
            logger.error("No common users/courses found between train and test data.")
        
        user_indices = pred_matrix.index.get_indexer(eval_df[self.cf_index])
        course_indices = pred_matrix.columns.get_indexer(eval_df[self.cf_column])
        eval_df["pred"] = pred_matrix.to_numpy()[user_indices, course_indices]
       
        mse = mean_squared_error(eval_df["rating"], eval_df["pred"])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(eval_df["rating"], eval_df["pred"])
        logger.info(f"CF Model Metrices values --> RMSE:{rmse}, MAE:{mae}")
        
        return float(rmse), float(mae)
    
    def implementing_randomforest_cbf(self, X_train,y_train):
        logger.info("Starting the implementation of contentet based filtering model using RandomForestRegressor model.")
        
        cbf_model = RandomForestRegressor(
            n_estimators= self.cbf_n_estimators,
            max_depth=self.cbf_max_depth,
            random_state=self.cbf_random_state,
            n_jobs=self.cbf_n_jobs
        )
        
        cbf_model.fit(X_train,y_train)
        
        logger.info("Successfully trained the CBF Model using RandomForestRegressor.")
        
        return cbf_model
    
    def evaluate_cbf_model(self, cbf_model, X_test, y_test):
        logger.info("Evaluating CBF Model")
        cbf_preds = cbf_model.predict(X_test)
        
        cbf_rmse = root_mean_squared_error(y_test, cbf_preds)
        cbf_mae = mean_absolute_error(y_test, cbf_preds)
        
        logger.info(f"CBF RandomForestRegressor Model Metrices --> RMSE:{cbf_rmse}, MAE:{cbf_mae}")
        
        return float(cbf_rmse), float(cbf_mae)
    
    def _save_final_model(self, model_bundle):
        model_path = self.save_model_path
        
        if not os.path.splitext(model_path)[1]:
            model_path = os.path.join(
                model_path,
                "recommender_model.joblib"
            )
        
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(model_bundle, model_path)
        
        return model_path
    
    def compairing_to_find_best_model(self):
        logger.info("Starting model selection and training process...")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.mlflow_experiment)
        if experiment and experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
        mlflow.set_experiment(self.mlflow_experiment)
        transformation = RecommenderDataTransformation(self.config)
        (X_train, y_train, X_test, y_test, train_df, test_df ) = transformation.run_transformation_pipeline()
        
        with mlflow.start_run(run_name="model_compairison") as run:
            mlflow.log_params({
                "cf_n_components": self.cf_n_components,
                "cf_random_state": self.cf_random_state,
                "cbf_n_estimators": self.cbf_n_estimators,
                "cbf_max_depth": self.cbf_max_depth,
                "cbf_random_state": self.cbf_random_state
            })
            
            cf_model = self.implementing_cf_model(train_df)
            cf_rmse, cf_mae = self.evaluate_cf_model(cf_model, test_df)
            
            mlflow.log_metrics({
                "cf_rmse": cf_rmse,
                "cf_mae": cf_mae
            })
            
            mlflow.sklearn.log_model(
                cf_model['svd'],
                "collaborative_filtering_svd"
            )
            
            cbf_model = self.implementing_randomforest_cbf(X_train,y_train)
            cbf_rmse, cbf_mae = self.evaluate_cbf_model(cbf_model, X_test, y_test)
            
            mlflow.log_metrics({
                "cbf_rmse": cbf_rmse,
                "cbf_mae": cbf_mae
            })
            mlflow.sklearn.log_model(
                cbf_model,
                "content_based_random_forest"
            )
            
            results ={
                "cf": {"rmse": cf_rmse, "mae": cf_mae},
                "cbf": {"rmse": cbf_rmse, "mae": cbf_mae}
            }
            
            best_model_name = min(
                results,
                key=lambda name: (
                    results[name]["rmse"],
                    results[name]["mae"]
                )
            )
            logger.info("Best model selected: %s", best_model_name)
            
            # Retraining the selected model using the complete dataset
            full_interaction = pd.concat(
                [train_df,test_df],
                ignore_index=True
            )
            
            if best_model_name == "cf":
                final_model = self.implementing_cf_model(full_interaction)
                model_bundle = {
                    "model_type": "collaborative_filtering",
                    "model": final_model
                }
            else:
                X_full = (
                    sparse.vstack([X_train, X_test])
                    if sparse.issparse(X_train) or sparse.issparse(X_test)
                    else np.vstack([X_train, X_test])
                )
                y_full = np.concatenate([y_train, y_test])
                final_model = self.implementing_randomforest_cbf(X_full,y_full)
                model_bundle = {
                    "model_type": "content_based_random_forest",
                    "model" : final_model
                }
            
            model_path = self._save_final_model(model_bundle)
            
            mlflow.log_param("selected_model", best_model_name)
            mlflow.log_param("final_model_path", model_path)
            mlflow.log_artifact(model_path, artifact_path="final_model")
            
            logger.info("Final model saved to: %s", model_path)
            
            return {
                "selected_model" : best_model_name,
                "metrics": results,
                "model_path": model_path,
                "mlflow_run_id": run.info.run_id
            }

if __name__ == "__main__":
    config = ConfigLoader()
    trainer = ModelTraining(config)
    result  = trainer.compairing_to_find_best_model()
    print(result)