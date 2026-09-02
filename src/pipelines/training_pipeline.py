import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Optional

from src.config.loader import ConfigLoader
from src.data.ingest import DataIngestion
from src.data.validate import DataValidator
from src.features.build_features import FeatureEnginner
from src.logger.logger import get_logger
from src.models.train_model import ModelTraining

logger = get_logger(__name__)


class TrainingPipeline:
    def __init__(self, config: Optional[ConfigLoader] = None):
        self.config = config or ConfigLoader()
        self.ingestor = DataIngestion(self.config)
        self.validator = DataValidator(self.config)
        self.feature_enginner = FeatureEnginner(self.config)
        self.trainer = ModelTraining(self.config)

    def run(self):
        logger.info("Starting end-to-end training pipeline.")

        cleaned_df = self.ingestor.run_ingestion_pipeline()
        self.validator.run(cleaned_df)
        engiineered_df = self.feature_enginner.run_feature_engineering()
        result = self.trainer.compairing_to_find_best_model()

        logger.info("Training pipeline completed successfully.")

        return {
            "cleaned_data_path": self.config.get("data.processed_data_path"),
            "engineered_data_path": self.config.get("data.engineered_data_path"),
            "model_name": result["selected_model"],
            "model_path": result["model_path"],
        }


if __name__ == "__main__":
    config = ConfigLoader()
    training_pipeline = TrainingPipeline(config)
    training_pipeline.run()
