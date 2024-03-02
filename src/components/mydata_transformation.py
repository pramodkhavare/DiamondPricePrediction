import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
# from src.utils import 

@dataclass
class DataTransFormationConfig:
    preprocessor_object_file_path=os.path.join('myartifacts','preprocessor.pkl')
    
class DataTransformation():
    def __init__(self) -> None:
        self.data_trtansformation_config=DataTransFormationConfig()
        
        
        
        
        
    def get_data_transformation_object(self):
        try:
            logging.info('Pipeline creation has been started')
            
            #Deciding ranking in catagorical data
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            
            numrical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )
            catagorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scalar',StandardScaler())
                ]
            )
            
            preprocessor=ColumnTransformer([
                ('num_pipeline',numrical_pipeline,numerical_cols),
                ('cat_pipeline',catagorical_pipeline,categorical_cols)
            ])
            
            logging.info('Pipeline created successfully')
            
            return preprocessor
        
        
        except Exception as e:
            logging.info("Unable to create Pipeline")
            raise CustomException(e,sys)
        
        
        
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info('Data Transformation started')
            
            logging.info('Data reading from test data file and train data file')
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            
            logging.info('Data reading completed')
            logging.info(f'Train data record \n{train_data.head()}')
            logging.info(f'Test data record \n{test_data.head()}')
            
            logging.info('Access Data Transformer Object')
            preprocessor_obj=self.get_data_transformation_object()
            
            target_data_column='price' 
            drop_column=['id','price']
            
            input_feature_train_data=train_data.drop(columns=drop_column,axis=1)
            input_feature_test_data=test_data.drop(columns=drop_column,axis=1)
            
            target_feature_train_data=train_data[target_data_column]
            target_feature_test_data=test_data[target_data_column]
            
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_data)
            
            logging.info('Processor applied on train and test data')
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_data)]
            
            save_object(
                file_path=self.data_trtansformation_config.preprocessor_object_file_path,
                obj=preprocessor_obj
            )
            logging.info('Preprocessor saved in .pkl file sussessfully')
            
            logging.info('Data Transformation finished sussefully')
            return(
                train_arr,
                test_arr,
                self.data_trtansformation_config.preprocessor_object_file_path,
            )
            
            
        
        except Exception as e:
            logging.info('Error In Data Transformation')
            raise CustomException(e,sys)