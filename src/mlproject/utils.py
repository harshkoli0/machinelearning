import os
import sys
import pandas as pd
import pymysql
from dotenv import load_dotenv
from pathlib import Path
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

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

