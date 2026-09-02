import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import pandas as pd

from src.config.loader import ConfigLoader
from src.data.data_contract import DataContract
from src.logger.logger import get_logger

logger = get_logger(__name__)


class DataValidator:

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.contract = DataContract()
        self.processed_path = self.config.get("data.processed_data_path")
        self.report_dir = "data/validation_reports"

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.processed_path):
            logger.error(f"Processed data file is missing at: {self.processed_path}")
            raise FileNotFoundError(
                f"Processed file not found at: {self.processed_path}"
            )
        logger.info(f"Successfully loaded dataset from: {self.processed_path}")
        return pd.read_csv(self.processed_path)

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        logger.info("Starting comprehensive data validation checks...")
        validation_passed = True
        errors = []

        for col in self.contract.NUMERICAL_COLUMNS:
            if col not in df.columns:
                errors.append(
                    f"Missing Column: Numerical column '{col}' is missing from DataFrame."
                )
                validation_passed = False
                continue

            if not np.issubdtype(df[col].dtype, np.number):
                errors.append(
                    f"Type Mismatch: Numerical column '{col}' expected numeric, got '{df[col].dtype}'."
                )
                validation_passed = False

            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(
                    f"Constraint violation: Numerical column '{col}' contains {null_count} missing values."
                )
                validation_passed = False

        for col in self.contract.CATEGORICAL_COLUMNS:
            if col not in df.columns:
                errors.append(
                    f"Missing Column: Categorical column '{col}' is missing from DataFrame."
                )
                validation_passed = False
                continue
            if not (
                pd.api.types.is_string_dtype(df[col])
                or pd.api.types.is_object_dtype(df[col])
            ):
                errors.append(
                    f"Type Mismatch: Categorical column '{col}' expected string/object, got '{df[col].dtype}'."
                )
                validation_passed = False

            null_values = df[col].isnull().sum()
            if null_values > 0:
                errors.append(
                    f"Constraint violation: Categorical column '{col}' contains {null_values} missing values."
                )
                validation_passed = False

        if validation_passed:
            logger.info("All validation constraints passed successfully.")
            self._save_report("SUCCESS: All constraints passed.")
        else:
            logger.error("Data contract breach detected! Listing violations:")
            for err in errors:
                logger.error(f" - {err}")
            self._save_report("\n".join(errors))
            raise ValueError(
                "Pipeline stopped: Incoming data failed validation thresholds."
            )

        return validation_passed

    def run(self, df=None) -> bool:
        data = self.load_data() if df is None else df
        return self.validate_dataset(data)

    def _save_report(self, message: str) -> None:
        os.makedirs(self.report_dir, exist_ok=True)
        report_path = os.path.join(self.report_dir, "latest_validation_report.txt")
        with open(report_path, "w") as f:
            f.write(message)
        logger.info(f"Validation report saved to: {report_path}")


if __name__ == "__main__":
    config_loader = ConfigLoader()
    validator = DataValidator(config=config_loader)
    processed_df = validator.load_data()
    validator.validate_dataset(processed_df)
