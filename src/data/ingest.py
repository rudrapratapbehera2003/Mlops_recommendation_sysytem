import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pandas as pd

from src.config.loader import ConfigLoader
from src.data.data_contract import DataContract
from src.logger.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.contract = DataContract()
        self.raw_path = self.config.get("data.raw_data_path")
        self.processed_path = self.config.get("data.processed_data_path")
        self.minimum_allowed_nulls = self.config.get("data.minimum_allowed_null_values")
        self.filling_with_mean = (
            self.config.get("handle_missing_values.columns_for_mean") or []
        )
        self.filling_with_median = (
            self.config.get("handle_missing_values.columns_for_median") or []
        )

    def load_raw_data(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            logger.error(f"Raw data path is missing at: {self.raw_path}")
            raise FileNotFoundError(f"Raw data file not found at: {self.raw_path}")

        logger.info(f"Successfully located raw data file at: {self.raw_path}")
        return pd.read_excel(self.raw_path, sheet_name="Sheet1")

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        numerical_variables = [
            col
            for col in df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            if col not in ["user_id", "course_id"]
        ]
        categorical_variables = df.select_dtypes(include=["object"]).columns.tolist()

        for col in numerical_variables:
            if df[col].isnull().sum() > 0:
                if col in self.filling_with_mean:
                    df[col] = df[col].fillna(df[col].mean())
                elif col in self.filling_with_median:
                    df[col] = df[col].fillna(df[col].median())

        for col in categorical_variables:
            if df[col].isnull().sum() > 0:
                modes = df[col].mode()
                if not modes.empty:
                    df[col] = df[col].fillna(modes[0])

        return df

    def save_processed_df(self, df: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        logger.info(f"Baseline data successfully saved to: {self.processed_path}")

    def run_ingestion_pipeline(self) -> pd.DataFrame:
        logger.info("Starting data ingestion workflow...")
        raw_df = self.load_raw_data()
        processed_df = self.fill_missing_values(raw_df)
        self.save_processed_df(processed_df)
        logger.info("Data ingestion workflow completed successfully.")
        return processed_df


if __name__ == "__main__":
    config_loader = ConfigLoader()
    ingestion = DataIngestion(config=config_loader)
    ingestion.run_ingestion_pipeline()
