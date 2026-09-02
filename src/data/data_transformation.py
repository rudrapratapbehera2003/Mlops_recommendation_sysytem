import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.loader import ConfigLoader
from src.logger.logger import get_logger

logger = get_logger(__name__)


class RecommenderDataTransformation:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.target_col = self.config.get("data.target_col", "rating")
        self.feature_path = self.config.get(
            "data.engineered_data_path", "data/processed/engineered_recommend_data.csv"
        )

    def load_enginnered_data(self) -> pd.DataFrame:
        if not os.path.exists(self.feature_path):
            logger.error(f"Enginnered feature file is missing at: {self.feature_path}")
            raise FileNotFoundError(
                f"Enginnered file is missing at: {self.feature_path}"
            )

        df = pd.read_csv(self.feature_path)
        logger.info(f"Enginnered file is successfully loaded.....")
        return df

    def recommender_dataset_preprocessing(self, df: pd.DataFrame):
        logger.info("Splitting dataset to apply the transformation techniques.......")

        train_df, test_df = train_test_split(df, random_state=42, shuffle=True)

        exclude_cols = [
            self.target_col,
            "user_id",
            "course_id",
            "course_name",
            "instructor",
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        logger.info(
            f"IMPORTANT: Removed user_id from features for preference-based model"
        )
        logger.info(f"Using course feature columns (preferences): {feature_cols}")

        num_cols = [
            x
            for x in df[feature_cols]
            .select_dtypes(include=["int64", "float64"])
            .columns.tolist()
            if x != self.target_col
        ]
        cate_cols = (
            df[feature_cols]
            .select_dtypes(include=["object", "category"])
            .columns.tolist()
        )

        logger.info(f"Numerical features (preferences): {num_cols}")
        logger.info(f"Categorical features (preferences): {cate_cols}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical_col", StandardScaler(), num_cols),
                ("cate", OneHotEncoder(handle_unknown="ignore"), cate_cols),
            ]
        )

        logger.info(f"Fitting preprocessor on training data...")
        X_train = preprocessor.fit_transform(train_df[feature_cols])
        X_test = preprocessor.transform(test_df[feature_cols])

        y_train = train_df[self.target_col].values
        y_test = test_df[self.target_col].values

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")

        logger.info(
            "Successfully splitting the dataset for further model training......."
        )
        logger.info("Model will work for ANY user - not tied to specific user_ids")

        return (
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
        )

    def run_transformation_pipeline(self):
        logger.info("Starting recommender data transformation pipeline.....")
        logger.info(
            "IMPORTANT: This pipeline trains preference-based model (no user_id)"
        )

        df_enginnered = self.load_enginnered_data()
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
        ) = self.recommender_dataset_preprocessing(df=df_enginnered)

        logger.info(
            "Data transformation pipeline for recommendation system is completed."
        )
        logger.info(
            "Model features are preference-based: difficulty, price, feedback, enrollment, etc."
        )
        logger.info(
            "Ready for model building with ANY user, not tied to specific user_ids......"
        )

        return (
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
        )


if __name__ == "__main__":
    config_loader = ConfigLoader()
    transformation = RecommenderDataTransformation(config=config_loader)
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
