import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from src.components.mydata_transformation import DataTransformation
from src.utils import evaluate_model




@dataclass
class ModelTrainerConfig:
    trained_data_config=os.path.join('myartifacts','model.pkl')
    

class ModelTraining:
    def __init__(self) -> None:
        self.model_training_config=ModelTrainerConfig()
        
        
    def initiate_model_training(self,train_data,test_data):
       try:
           logging.info('Model Training Started')
           X_train,X_test,y_train,y_test=(
               train_data[:,:-1],
               test_data[:,:-1],
               train_data[:,-1],
               test_data[:,-1]
           )
           logging.info('Data has been separated')
           logging.info(X_train)
           logging.info(y_train)
           
           models={
               'linear_regression':LinearRegression(),
               'lasso':Lasso(),
               'ridge':Ridge(),
               'elasticnet':ElasticNet()
           }
           
           
           logging.info('Evaluating Model')
           model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
           print("*"*35)
           logging.info(f'Model Report:{model_report}')
           
           best_model_score=max(sorted(list(model_report.values())))
           
           best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
           best_model=models[best_model_name]

        
           print(f'Best model found is {best_model} with accuracy {best_model_score}')
           print('\n====================================================================================\n')
           logging.info(f'Best Model Found , Model Name : {best_model} , R2 Score : {best_model_score}')
           
           save_object(
               file_path=self.model_training_config.trained_data_config,
               obj=best_model
           )
           
        
       
       
       except Exception as e:
           logging.info('Unable to train model')
           raise CustomException(e,sys)
           