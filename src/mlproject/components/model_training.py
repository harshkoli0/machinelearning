import os 
import sys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from src.mlproject.utils import save_object,evaluate_models 
 
 
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(n_estimators=20),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False)
            }

            params = {
                "Random Forest": {
                    "n_estimators": [8, 16, 32],
                    "max_depth": [3, 5],
                    "max_features": ['sqrt', 'log2']
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [50, 100],
                    "subsample": [0.7, 0.9]
                },
                "XGBoost": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5]
                },
                "CatBoost Regressor": {
                    "iterations": [100, 200],
                    "depth": [3, 5],
                    "learning_rate": [0.1]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1, 0.5]
                }
            }

            model_report, best_model = evaluate_models(x_train, y_train, x_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)
    
