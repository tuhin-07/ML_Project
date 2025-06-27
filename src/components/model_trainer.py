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
            'Linear Regression':LinearRegression(),
            'XGB Regressor':XGBRegressor(),
            'Decision Tree Regressor':DecisionTreeRegressor(),
            'CatBoost Regressor':CatBoostRegressor(verbose=False),
            # 'KNeighbors Regressor':KNeighborsRegressor(),
            'AdaBoost Regressor':AdaBoostRegressor(),
            'Random Forest Regressor':RandomForestRegressor(),
            'Gradient Boosting Regressor':GradientBoostingRegressor(),

        }

        params={
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGB Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

        model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)

        best_model_score = max(list(model_report.values()))

        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        best_model = models[best_model_name]

        if best_model_score < 0.6:
            raise CustomExecption("No best model found")
        
        logging.info(f"Best model found and it is {best_model_name} and its score is {best_model_score}")

        save_object(
            file_path=self.model_trainer_config.model_path,
            obj=best_model
        )
        predicted = best_model.predict(X_test)

        r2_Score = r2_score(y_test,predicted)

        return r2_Score


