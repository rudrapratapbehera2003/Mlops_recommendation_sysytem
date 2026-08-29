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
            "data.engineered_data_path", "data/processed/engineered_recommend_data.csv"
        )
        
    def load_cleaned_datset(self) -> pd.DataFrame:
        if not os.path.exists(self.processed_path):
            logger.error(f"Cleanes baseline data is missing at: {self.processed_path}")
            raise FileNotFoundError(f"File is not found at: {self.processed_path}")
        
        logger.info(f"Successfully located raw data file at: {self.processed_path}")
        df = pd.read_csv(self.processed_path)
        
        return df
        
        
    def cliping_outliers(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting the clipping of outliers in the numerical variables...... ")
        numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        outlier_summary = {}
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5*IQR
            upper_limit  = Q3 + 1.5*IQR
            outliers = (df[col] > upper_limit) | (df[col] < lower_limit)
            outlier_summary[col] = outliers.sum().item()
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        logger.info(f"Clipping of the outliers in numerical column is completed successfully with summary: {outlier_summary}")
        
        logger.info("Further checking whether the outliers have been clipped or not.....")
        outlier_summary_checking = {}
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5*IQR
            upper_limit = Q3 + 1.5*IQR
            outliers_ = (df[col] > upper_limit) | (df[col] < lower_limit)
            outlier_summary_checking[col] = outliers_.sum().item()
        logger.info(f"Outlier summary after clipping:{outlier_summary_checking}")
            
        return df
    
    def run_feature_engineering(self):
        logger.info("Starting feature enginnering workflow....")
        raw_df = self.load_cleaned_datset()
        clean_df = self.cliping_outliers(raw_df)
        os.makedirs(os.path.dirname(self.feature_output_path), exist_ok=True)
        clean_df.to_csv(self.feature_output_path,index=False)
        
        logger.info(f"Feature enginnering successfull. Matrix saved to: {self.feature_output_path}")
        return clean_df

    
    # def encoding_categorical_features(self, df:pd.DataFrame) -> pd.DataFrame:
    #     logger.info("Starting categorical feature encoding....")
        
    #     preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('cate', OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    #             make_column_selector(dtype_include=['object','category']))
    #         ],
    #         remainder="passthrough",
    #         verbose_feature_names_out=False
    #     )
    #     encoded_array = preprocessor.fit_transform(df)
    #     new_columns = preprocessor.get_feature_names_out()
    #     encoded_df = pd.DataFrame(encoded_array, columns=new_columns, index=df.index)
        
    #     logger.info(f"Encoding complete. Shape changed from {df.shape} to {encoded_df.shape}...... ")
    #     return encoded_df
        
if __name__=="__main__":
    config_loader = ConfigLoader()
    enginner = FeatureEnginner(config=config_loader)
    df = enginner.run_feature_engineering()
        