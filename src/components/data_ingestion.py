import pandas as pd 
import numpy as np
from src.logger.logger import logging
from src.exception.exception import CustomeException

import os
import sys
from sklearn.model_selection import train_test_split # type: ignore
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            data=pd.read_csv("https://raw.githubusercontent.com/keetrap/test/main/train.csv")
            logging.info("Readed data from the source")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Raw Dataset Saved in Artifact Folder")
            
            logging.info("Performing Train Test Split")
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("Train Test split Completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("Data Ingestion Completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("There is an error in Data Ingestion")
            raise CustomeException(e,sys)



