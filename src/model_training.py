import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import yaml
import joblib
import os

class ModelTrainer:
    def __init__(self, config_path='src/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {
            'linear_regression': LinearRegression(**self.config['models']['linear_regression']),
            'xgboost': XGBRegressor(**self.config['models']['xgboost']),
            'lightgbm': LGBMRegressor(**self.config['models']['lightgbm']),
            'catboost': CatBoostRegressor(**self.config['models']['catboost'])
        }
        
        self.results = {}
    
    def train_model(self, model_name, X_train, y_train):
        """Train a specific model."""
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance using multiple metrics."""
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def cross_validate(self, model, X, y):
        """Perform cross-validation on the model."""
        cv_scores = cross_val_score(
            model, X, y,
            cv=self.config['training']['cv_folds'],
            scoring=self.config['training']['scoring'],
            n_jobs=self.config['training']['n_jobs']
        )
        return -cv_scores.mean()  # Convert to positive MSE
    
    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models."""
        for model_name in self.models.keys():
            # Train model
            model = self.train_model(model_name, X_train, y_train)
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, X_test, y_test)
            
            # Cross-validation
            cv_score = self.cross_validate(model, X_train, y_train)
            
            # Store results
            self.results[model_name] = {
                'test_metrics': test_metrics,
                'cv_score': cv_score,
                'model': model
            }
            
            print(f"\nResults for {model_name}:")
            print(f"Test RMSE: {test_metrics['RMSE']:.2f}")
            print(f"Test MAE: {test_metrics['MAE']:.2f}")
            print(f"Test R2: {test_metrics['R2']:.2f}")
            print(f"CV MSE: {cv_score:.2f}")
    
    def save_models(self, output_dir='models'):
        """Save trained models to disk."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for model_name, result in self.results.items():
            model_path = os.path.join(output_dir, f'{model_name}.joblib')
            joblib.dump(result['model'], model_path)
            print(f"Saved {model_name} to {model_path}")
    
    def get_best_model(self):
        """Return the best performing model based on RMSE."""
        best_model_name = min(
            self.results.keys(),
            key=lambda x: self.results[x]['test_metrics']['RMSE']
        )
        return best_model_name, self.results[best_model_name]['model']

if __name__ == "__main__":
    from data_preparation import DataPreprocessor
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    # Train and evaluate models
    trainer = ModelTrainer()
    trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    # Save models
    trainer.save_models()
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model()
    print(f"\nBest performing model: {best_model_name}") 