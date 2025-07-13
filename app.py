from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation



if __name__== "__main__":
    logging.info("the exceution has started")
    
    
    try:
      
      data_ingestion=DataIngestion()
      train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
      #Data_Transformation_Config=DataTransformationConfig()
      Data_Transformation=DataTransformation()
      Data_Transformation.initiate_data_transformation(train_data_path,test_data_path)
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)