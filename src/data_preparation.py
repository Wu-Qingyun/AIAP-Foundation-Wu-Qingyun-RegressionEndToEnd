import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import os

class DataPreprocessor:
    def __init__(self, config_path='src/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the dataset from the specified path."""
        return pd.read_csv(self.config['data']['train_path'])
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and outliers."""
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Convert remaining_lease to numeric
        df['remaining_lease'] = pd.to_numeric(df['remaining_lease'].str.extract(r'(\d+)')[0])
        
        # Remove outliers using IQR method for numerical columns
        numerical_cols = ['floor_area_sqm', 'resale_price', 'remaining_lease']
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        
        return df
    
    def feature_engineering(self, df):
        """Create new features and transform existing ones."""
        # Convert month to datetime
        df['month'] = pd.to_datetime(df['month'])
        df['sale_year'] = df['month'].dt.year
        df['sale_month'] = df['month'].dt.month
        
        # Calculate price per square meter
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
        
        # Extract block number from block
        df['block_number'] = pd.to_numeric(df['block'].str.extract(r'(\d+)')[0], errors='coerce')
        
        # Drop original columns that are no longer needed
        df = df.drop(['month', 'block', 'town_id', 'flatm_id'], axis=1)
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables using Label Encoding."""
        categorical_cols = ['flat_type', 'storey_range', 'town_name', 'flatm_name', 'street_name']
        
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        return df
    
    def scale_numerical_features(self, df):
        """Scale numerical features using StandardScaler."""
        numerical_cols = ['floor_area_sqm', 'remaining_lease', 'sale_year', 'sale_month', 
                         'price_per_sqm', 'block_number']
        
        # Ensure all columns are numeric
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def prepare_data(self):
        """Main method to prepare the data for training."""
        # Load data
        df = self.load_data()
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale numerical features
        df = self.scale_numerical_features(df)
        
        # Split features and target
        X = df.drop('resale_price', axis=1)
        y = df['resale_price']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    print("Data preparation completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}") 