import os,sys
import numpy as np 
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomExecption
from src.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_path:str = os.path.join("artifacts","preprocessor.pkl")

class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def data_transformer_object(self):
        numerical_col = ['writing_score','reading_score']
        categorical_col = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

        try:
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('one hotEncoding',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info("Numerical and Categorical Pipeline Completed")

            preprocessor = ColumnTransformer(
                [
                ('cat_pipeline',cat_pipeline,categorical_col),
                ('num_pipeline',num_pipeline,numerical_col)
                ]
            )

            return preprocessor
        

        except Exception as e:
            raise CustomExecption(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.data_transformer_object()

            target_col = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df = test_df[target_col]


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(file_path = self.data_transformer_config.preprocessor_obj_path,
                        obj = preprocessing_obj)

            
            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_path
            )
        
        except Exception as e:
            raise CustomExecption(e,sys)
