import sys
import pandas as pd
import os
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:            
            model_path = os.path.join(os.getcwd(), 'artifacts', 'model.pkl')
            preprocessor_path = os.path.join(os.getcwd(), 'artifacts', 'preprocessor.pkl')
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_transformed = preprocessor.transform(features)
            pred = model.predict(data_transformed)
            return pred
        
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 location: str,
                 sqft: float,
                 bath: float, 
                 bhk: float):
        
        self.location = location
        self.sqft = sqft
        self.bath = bath
        self.bhk = bhk

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'location': [self.location],
                'sqft': [self.sqft],
                'bath': [self.bath],
                'bhk': [self.bhk]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered for prediction')
            return df
        
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)