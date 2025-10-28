import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
import json
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from src.exceptions import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    locations_file_path = os.path.join('artifacts','locations.json')

class DataTransformation:
    def __init__(self): 
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        EXACTLY matches your Jupyter notebook preprocessing
        '''
        try:
            # Use EXACT same column names as your Jupyter notebook
            categorical_cols = ['location']  
            numerical_cols = ['sqft', 'bath', 'bhk']  # Match your notebook columns
            
            logging.info(f'Categorical Columns: {categorical_cols}')
            logging.info(f'Numerical Columns: {numerical_cols}')

            # EXACT same OneHotEncoder parameters as your notebook
            ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
            
            # EXACT same preprocessor as your notebook
            preprocessor = make_column_transformer(
                (ohe, categorical_cols),      # OneHot for location
                remainder='passthrough'       # Passthrough numerical (NO SCALING)
            )

            logging.info('Preprocessor object created (matching notebook exactly)')
            return preprocessor
        
        except Exception as e:
            logging.info('Exception occurred in Data Transformation Phase')
            raise CustomException(e, sys)

    def save_locations_to_json(self, train_df, test_df):
        """
        Save all unique locations to JSON file for frontend autocomplete
        """
        try:
            # Combine locations from both train and test data
            train_locations = train_df['location'].unique().tolist()
            test_locations = test_df['location'].unique().tolist()
            
            # Get all unique locations and sort them
            all_locations = sorted(list(set(train_locations + test_locations)))
            
            # Create locations data
            locations_data = {
                'locations': all_locations,
                'total_count': len(all_locations),
                'train_count': len(train_locations),
                'test_count': len(test_locations),
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
            # Save to JSON file
            os.makedirs(os.path.dirname(self.data_transformation_config.locations_file_path), exist_ok=True)
            with open(self.data_transformation_config.locations_file_path, 'w') as f:
                json.dump(locations_data, f, indent=2)
            
            logging.info(f'✅ Locations JSON file saved: {self.data_transformation_config.locations_file_path}')
            logging.info(f'📊 Total unique locations: {len(all_locations)}')
            logging.info(f'📍 Sample locations: {all_locations[:5]}')
            
            return all_locations
            
        except Exception as e:
            logging.error(f'❌ Failed to save locations JSON: {e}')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        Main method to transform data - matches notebook exactly
        '''
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            # CRITICAL: Check if your data matches Jupyter notebook structure
            logging.info(f'Train columns: {train_df.columns.tolist()}')
            logging.info(f'Test columns: {test_df.columns.tolist()}')

            # ✅ NEW: Save locations to JSON file for frontend
            logging.info("Generating locations JSON file for frontend...")
            self.save_locations_to_json(train_df, test_df)

            preprocessing_obj = self.get_data_transformation_object()

            # Use EXACT same column names as your Jupyter notebook
            target_column_name = 'price'
            feature_columns = ['location', 'sqft', 'bath', 'bhk']  # Must match notebook
            
            # Debug info
            logging.info("=== DEBUG INFO ===")
            logging.info(f"Train data types:\n{train_df[feature_columns].dtypes}")
            logging.info(f"Test data types:\n{test_df[feature_columns].dtypes}")
            logging.info(f"Train 'location' unique values sample: {train_df['location'].unique()[:10]}")
            logging.info(f"Train numerical stats:\n{train_df[['sqft', 'bath', 'bhk']].describe()}")            
            
            # Verify columns exist
            missing_cols_train = [col for col in feature_columns + [target_column_name] if col not in train_df.columns]
            missing_cols_test = [col for col in feature_columns + [target_column_name] if col not in test_df.columns]
            
            if missing_cols_train or missing_cols_test:
                raise CustomException(f"Missing columns: Train{missing_cols_train}, Test{missing_cols_test}", sys)

            # Prepare features and target
            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object (matching notebook)...")
            
            # Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Transformed shapes - Train: {input_feature_train_arr.shape}, Test: {input_feature_test_arr.shape}")

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info('Preprocessor pickle file saved')

            # Prepare arrays for model training
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info('Exception occurred in initiate_data_transformation function')
            raise CustomException(e, sys)