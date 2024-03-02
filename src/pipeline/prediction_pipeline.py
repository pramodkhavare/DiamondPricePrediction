import sys
import os
from src.utils import load_object
from src.utils import save_object
import pandas as pd
from src.logger import logging
from src.exception import CustomException

class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def pipeline_config(self,feature):
        
        try:
            logging.info('Reading preprocessor and model pkl file')
            prerocessor_path=os.path.join('myartifacts','preprocessor.pkl')
            mode_path=os.path.join('myartifacts','model.pkl')
            
            
            preprocessor=load_object(prerocessor_path)
            model=load_object(mode_path)
            logging.info('.pkl file converted')
            
            scaled_data=preprocessor.transform(feature)
            
            y_prediction=model.predict(scaled_data)
            
            return y_prediction
        
    
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData():
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str )->None:
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity 
    
    
    def get_data_as_dataframe(self):
        try:
            data_dictionary={
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity] 
                
            }
            
            df=pd.DataFrame(data_dictionary)
            logging.info("Dataframe created")
            return df
        
        
        
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)