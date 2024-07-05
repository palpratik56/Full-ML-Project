import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join('artifacts',"train.csv")
    test_path: str = os.path.join('artifacts',"test.csv")
    raw_path: str = os.path.join('artifacts',"data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingest(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(os.path.join('notebooks',"New_data.csv"))
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_path,index=False,header=True)

            logging.info("Data Ingestion completed")
            return(
                self.ingestion_config.train_path,self.ingestion_config.test_path)
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingest()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
