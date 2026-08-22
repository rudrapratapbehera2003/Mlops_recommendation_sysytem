import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import pandas as pd
from src.config.loader import ConfigLoader
from src.logging.logger import get_logger
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder


logger = get_logger(__name__)


class FeatureEnginner:
    def __init__(self, config:ConfigLoader):
        self.config = config
        self.processed_path = self.config.get("data.processed_data_path")
        self.feature_output_path = self.config.get(
            "data.engineered_data_path", "data/processed/engineered_features.csv"
        )
        
    def cliping_outliers(self, df:pd.DataFrame) -> pd.DataFrame:
        
        return 

    
    def encoding_categorical_features(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting categorical feature encoding....")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cate', OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                make_column_selector(dtype_include=['object','category']))
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )
        encoded_array = preprocessor.fit_transform(df)
        new_columns = preprocessor.get_feature_names_out()
        encoded_df = pd.DataFrame(encoded_array, columns=new_columns, index=df.index)
        
        logger.info(f"Encoding complete. Shape changed from {df.shape} to {encoded_df.shape}...... ")
        return encoded_df
        
        