from dataclasses import dataclass
import os,sys

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor

from src.exception import CustomExecption
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path:str = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_data,test_data):
        X_train,y_train,X_test,y_test = (train_data[:,:-1],train_data[:,-1],test_data[:,:-1],test_data[:,-1])

        models ={
            'Linear_regression':LinearRegression(),
            'XGBRegressor':XGBRegressor(),
            'DecisionTreeRegressor':DecisionTreeRegressor(),
            'CatBoostRegressor':CatBoostRegressor(),
            'KNeighborsRegressor':KNeighborsRegressor(),
            'AdaBoostRegressor':AdaBoostRegressor(),
            'RandomForestRegressor':RandomForestRegressor(),
            'GradientBoostingRegressor':GradientBoostingRegressor(),

        }

        model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
