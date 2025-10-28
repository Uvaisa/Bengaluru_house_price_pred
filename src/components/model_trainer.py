import sys
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        '''
        Train model with EXACT same approach as Jupyter notebook
        '''
        try:
            logging.info("Starting model training (matching notebook approach)")
            
            # Split train and test data
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logging.info(f"Training on data - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            # Use LinearRegression (same as your notebook)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Single split score (same as your notebook)
            single_split_score = model.score(X_test, y_test)
            
            # Cross-validation (same as your notebook)
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            X_combined = np.vstack([X_train, X_test])
            y_combined = np.hstack([y_train, y_test])
            cv_scores = cross_val_score(model, X_combined, y_combined, cv=cv)
            
            logging.info(f"Single split R²: {single_split_score:.6f}")
            logging.info(f"CV Scores: {cv_scores}")
            logging.info(f"CV Mean: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
            
            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")
            
            return single_split_score, cv_scores.mean()
            
        except Exception as e:
            logging.info('Exception occurred in model training')
            raise CustomException(e, sys)