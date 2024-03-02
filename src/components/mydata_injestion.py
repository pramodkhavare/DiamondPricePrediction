import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
# from src.utils import evaluate_model
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.mydata_transformation import DataTransformation
from src.components.mymodel_trainer import ModelTraining


@dataclass
class DataInjectionConfig():
    train_data_path=os.path.join('myartifacts','train.csv')
    test_data_path=os.path.join('myartifacts','test.csv')
    raw_data_path=os.path.join('myartifacts','raw.csv')
    


class DataInjection():
    def __init__(self) -> None:
        self.data_injectioin_config=DataInjectionConfig()
        
    def initiate_data_injection(self):
        logging.info('Data injection process has been started')
        
        try:
            logging.info('Data reading')
            data=pd.read_csv(os.path.join('notebook','data','gemestone.csv'))
            
            os.makedirs(os.path.dirname(self.data_injectioin_config.raw_data_path),exist_ok=True)
            # logging.info(f'We are able to create new directory with name {dir_name}')
            
            train_data,test_data=train_test_split(data,test_size=0.3,random_state=121)
            logging.info('Data separated into train and test file')
            
            train_data.to_csv(self.data_injectioin_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_injectioin_config.test_data_path,index=False,header=True)
            data.to_csv(self.data_injectioin_config.raw_data_path)
            logging.info('Data saved into train and test data file')
            
            return(
                self.data_injectioin_config.train_data_path,
                self.data_injectioin_config.test_data_path
            )
            
                        
            
            
            
        
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        
        
# if __name__=='__main__':
#     object=DataInjection()
#     train_data_path,test_data_path=object.initiate_data_injection()  
#     data_transformer=DataTransformation()
#     train_data_arr,test_data_arr,_=data_transformer.initiate_data_transformation(train_data_path,test_data_path)
#     model=ModelTraining()
#     model.initiate_model_training(train_data=train_data_arr,test_data=test_data_arr)
