
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.mydata_injestion import DataInjection
from src.components.mydata_transformation import DataTransformation
from src.components.mymodel_trainer import ModelTraining


if __name__=='__main__':
    object=DataInjection()
    train_data_path,test_data_path=object.initiate_data_injection()  
    data_transformer=DataTransformation()
    train_data_arr,test_data_arr,_=data_transformer.initiate_data_transformation(train_data_path,test_data_path)
    model=ModelTraining()
    model.initiate_model_training(train_data=train_data_arr,test_data=test_data_arr)
    