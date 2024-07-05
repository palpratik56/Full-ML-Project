import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_trans_config = DataTransformationConfig()

    def get_data_transformer_object(self, num_cols, cat_cols):
        # This function is responsible for data transformation
        try:
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())]
            )

            cat_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,num_cols),
                ("cat_pipelines",cat_pipeline,cat_cols) ])

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")

            logging.info("Obtaining preprocessing object...")

            num_cols = ["math_score","writing_score"]
            cat_cols = train_df.select_dtypes(include=object).columns
            
            preprocessing_obj=self.get_data_transformer_object(num_cols=num_cols, cat_cols=cat_cols)

            target_column_name="average"
            drop_columns = ["reading_score", "total_score","average"]

            input_feature_train_df=train_df.drop(columns=drop_columns)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data...")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_trans_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return ( train_arr, test_arr, self.data_trans_config.preprocessor_obj_file_path,  )
        except Exception as e:
            raise CustomException(e, sys)
