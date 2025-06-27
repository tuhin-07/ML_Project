from src.exception import CustomExecption
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os ,sys
import pandas as pd

from src.components.data_transformer import DataTransformer
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts",'train.csv')
    test_data_path:str =os.path.join("artifacts",'test.csv')
    raw_data_path:str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the Ingestion Process")

        try:
            df = pd.read_csv("C:/Users/Tuhin/OneDrive/Desktop/ML-Project/notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)


            logging.info("Train test split initiated")
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            CustomExecption(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()

    data_tranformer = DataTransformer()
    train_data,test_data,_ = data_tranformer.initiate_data_transformation(train_path,test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_data,test_data))



