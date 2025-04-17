import pytest
import pandas as pd
import numpy as np
from src.data_preparation import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'id': range(1, 6),
        'month': ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05'],
        'flat_type': ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '3 ROOM'],
        'block': ['123', '456', '789', '101', '202'],
        'street_name': ['Main St', 'Park Ave', 'Lake Rd', 'Hill St', 'River Rd'],
        'storey_range': ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15'],
        'floor_area_sqm': [45.0, 67.0, 90.0, 110.0, 68.0],
        'lease_commence_date': [1985, 1990, 1995, 2000, 2005],
        'remaining_lease': ['68 years', '63 years', '58 years', '53 years', '48 years'],
        'resale_price': [250000.0, 350000.0, 450000.0, 550000.0, 360000.0],
        'town_id': [1, 2, 3, 4, 5],
        'flatm_id': [1, 2, 3, 4, 5],
        'town_name': ['Town1', 'Town2', 'Town3', None, 'Town5'],
        'flatm_name': ['Flat1', 'Flat2', None, 'Flat4', 'Flat5']
    })

@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance."""
    return DataPreprocessor()

def test_load_data(preprocessor):
    """Test data loading functionality."""
    try:
        df = preprocessor.load_data()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    except Exception as e:
        pytest.fail(f"Data loading failed: {str(e)}")

def test_clean_data(preprocessor, sample_data):
    """Test data cleaning functionality."""
    # Test missing value handling
    cleaned_df = preprocessor.clean_data(sample_data)
    assert cleaned_df['town_name'].isnull().sum() == 0
    assert cleaned_df['flatm_name'].isnull().sum() == 0
    
    # Test data type conversion
    assert pd.api.types.is_numeric_dtype(cleaned_df['remaining_lease'])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['month'])
    assert pd.api.types.is_numeric_dtype(cleaned_df['block_number'])

def test_feature_engineering(preprocessor, sample_data):
    """Test feature engineering functionality."""
    # Clean data first
    cleaned_df = preprocessor.clean_data(sample_data)
    
    # Apply feature engineering
    engineered_df = preprocessor.feature_engineering(cleaned_df)
    
    # Check derived features
    assert 'price_per_sqm' in engineered_df.columns
    assert 'block_number' in engineered_df.columns
    
    # Verify calculations
    assert all(engineered_df['price_per_sqm'] == 
              engineered_df['resale_price'] / engineered_df['floor_area_sqm'])

def test_encode_categorical_features(preprocessor, sample_data):
    """Test categorical feature encoding."""
    # Clean and engineer data first
    cleaned_df = preprocessor.clean_data(sample_data)
    engineered_df = preprocessor.feature_engineering(cleaned_df)
    
    # Encode categorical features
    encoded_df = preprocessor.encode_categorical_features(engineered_df)
    
    # Check if categorical columns are encoded
    categorical_cols = ['flat_type', 'storey_range']
    for col in categorical_cols:
        assert pd.api.types.is_numeric_dtype(encoded_df[col])

def test_scale_numerical_features(preprocessor, sample_data):
    """Test numerical feature scaling."""
    # Prepare data
    cleaned_df = preprocessor.clean_data(sample_data)
    engineered_df = preprocessor.feature_engineering(cleaned_df)
    encoded_df = preprocessor.encode_categorical_features(engineered_df)
    
    # Scale numerical features
    scaled_df = preprocessor.scale_numerical_features(encoded_df)
    
    # Check if numerical columns are scaled
    numerical_cols = ['floor_area_sqm', 'resale_price', 'remaining_lease']
    for col in numerical_cols:
        assert abs(scaled_df[col].mean()) < 1e-10  # Mean should be close to 0
        assert abs(scaled_df[col].std() - 1.0) < 1e-10  # Std should be close to 1

def test_prepare_data(preprocessor):
    """Test the complete data preparation pipeline."""
    try:
        X_train, X_test, y_train, y_test = preprocessor.prepare_data()
        
        # Check if data is split correctly
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Check if shapes are consistent
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # Check if test size is approximately 20%
        assert abs(len(X_test) / (len(X_train) + len(X_test)) - 0.2) < 0.01
    except Exception as e:
        pytest.fail(f"Data preparation pipeline failed: {str(e)}")

def test_empty_dataframe_handling(preprocessor):
    """Test handling of empty DataFrames."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        preprocessor.clean_data(empty_df)

def test_invalid_data_types(preprocessor, sample_data):
    """Test handling of invalid data types."""
    # Modify data to include invalid values
    sample_data.loc[0, 'remaining_lease'] = 'invalid'
    sample_data.loc[1, 'block'] = 'invalid'
    
    # Test if the preprocessor handles invalid values gracefully
    cleaned_df = preprocessor.clean_data(sample_data)
    assert pd.api.types.is_numeric_dtype(cleaned_df['remaining_lease'])
    assert pd.api.types.is_numeric_dtype(cleaned_df['block_number']) 