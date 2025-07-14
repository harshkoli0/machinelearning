import os
import sys
import pandas as pd
import pymysql
from dotenv import load_dotenv
from pathlib import Path
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
'''
# ✅ Auto-detect .env in the current working directory
load_dotenv()

# ✅ Read environment variables
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

# ✅ Debug print
print("=== .env LOAD TEST (Method 3) ===")
print(f"Host     : {host}")
print(f"User     : {user}")
print(f"Password : {password}")
print(f"DB       : {db}")
print("=================================")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db
        )
        logging.info("✅ Connection established successfully.")
        
        df = pd.read_sql_query('SELECT * FROM studentsss', mydb)
        logging.info("✅ SQL query executed successfully.")
        print(df.head())
        return df

    except Exception as ex:
        logging.error("❌ Failed to read SQL data.")
        raise CustomException(ex, sys)
        '''
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb")as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
  
def evaluate_models(x_train, y_train, x_test, y_test, models:dict, param:dict):
    try:
        report = {}
        best_model_overall = None
        best_score = float('-inf')

        for model_name, model in models.items():
            if model_name not in param or not param[model_name]:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = r2_score(y_test, y_pred)
                report[model_name] = score
                if score > best_score:
                    best_score = score
                    best_model_overall = model
                continue

            print(f"🔍 Tuning {model_name}")
            gs = GridSearchCV(estimator=model, param_grid=param[model_name], cv=3, n_jobs=-1, verbose=1)
            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_
            y_pred = best_model.predict(x_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score
            if score > best_score:
                best_score = score
                best_model_overall = best_model

        return report, best_model_overall


    except Exception as e:
        raise CustomException(e, sys)

