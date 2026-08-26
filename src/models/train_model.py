import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import numpy as np
import mlflow

from src.logging.logger import get_logger
from src.config.loader import ConfigLoader
from src.data.data_transformation import RecommenderDataTransformation
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor


logger = get_logger(__name__)

class ModelTrainig:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.save_model_path = self.config.get("model.save_path")
        

    def evaluate_cf_model(self):
        return 
        
        
