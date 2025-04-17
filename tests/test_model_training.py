import pytest
import pandas as pd
import numpy as np
from src.model_training import ModelTrainer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'floor_area_sqm': np.random.uniform(30, 150, n_samples),
        'remaining_lease': np.random.uniform(40, 99, n_samples),
        'price_per_sqm': np.random.uniform(3000, 8000, n_samples),
        'flat_type': np.random.randint(0, 5, n_samples),
        'storey_range': np.random.randint(0, 5, n_samples)
    })
    
    y = 5000 * X['floor_area_sqm'] + 1000 * X['remaining_lease'] + np.random.normal(0, 10000, n_samples)
    
    return X, y

@pytest.fixture
def trainer():
    """Create a ModelTrainer instance."""
    return ModelTrainer()

def test_train_model(trainer, sample_data):
    """Test model training functionality."""
    X, y = sample_data
    
    # Test training a single model
    model = LinearRegression()
    trained_model = trainer.train_model(model, X, y)
    
    assert hasattr(trained_model, 'predict')
    assert hasattr(trained_model, 'coef_')

def test_evaluate_model(trainer, sample_data):
    """Test model evaluation functionality."""
    X, y = sample_data
    
    # Train and evaluate a model
    model = LinearRegression()
    trained_model = trainer.train_model(model, X, y)
    metrics = trainer.evaluate_model(trained_model, X, y)
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

def test_cross_validate(trainer, sample_data):
    """Test cross-validation functionality."""
    X, y = sample_data
    
    # Test cross-validation
    model = LinearRegression()
    cv_scores = trainer.cross_validate(model, X, y)
    
    assert 'rmse_mean' in cv_scores
    assert 'rmse_std' in cv_scores
    assert isinstance(cv_scores['rmse_mean'], float)
    assert isinstance(cv_scores['rmse_std'], float)

def test_train_and_evaluate_all(trainer, sample_data):
    """Test training and evaluating all models."""
    X, y = sample_data
    
    # Test training and evaluating all models
    results = trainer.train_and_evaluate_all(X, y)
    
    assert isinstance(results, dict)
    assert len(results) > 0
    for model_name, metrics in results.items():
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics

def test_save_models(trainer, sample_data, tmp_path):
    """Test model saving functionality."""
    X, y = sample_data
    
    # Train a model
    model = LinearRegression()
    trained_model = trainer.train_model(model, X, y)
    
    # Save the model
    save_path = tmp_path / "test_model.pkl"
    trainer.save_models({'test_model': trained_model}, str(save_path))
    
    assert save_path.exists()

def test_get_best_model(trainer, sample_data):
    """Test getting the best performing model."""
    X, y = sample_data
    
    # Train multiple models
    models = {
        'linear': LinearRegression(),
        'rf': RandomForestRegressor(n_estimators=10),
        'xgb': xgb.XGBRegressor(n_estimators=10),
        'lgb': lgb.LGBMRegressor(n_estimators=10),
        'cb': cb.CatBoostRegressor(iterations=10, verbose=False)
    }
    
    results = {}
    for name, model in models.items():
        trained_model = trainer.train_model(model, X, y)
        metrics = trainer.evaluate_model(trained_model, X, y)
        results[name] = metrics
    
    best_model_name = trainer.get_best_model(results)
    assert best_model_name in models.keys()

def test_model_hyperparameters(trainer, sample_data):
    """Test model hyperparameter handling."""
    X, y = sample_data
    
    # Test with different hyperparameters
    model = RandomForestRegressor(n_estimators=10, max_depth=5)
    trained_model = trainer.train_model(model, X, y)
    
    assert hasattr(trained_model, 'predict')
    assert trained_model.n_estimators == 10
    assert trained_model.max_depth == 5

def test_data_validation(trainer):
    """Test data validation in model training."""
    # Test with empty data
    empty_X = pd.DataFrame()
    empty_y = pd.Series()
    
    with pytest.raises(ValueError):
        trainer.train_model(LinearRegression(), empty_X, empty_y)

def test_model_persistence(trainer, sample_data, tmp_path):
    """Test model persistence functionality."""
    X, y = sample_data
    
    # Train and save multiple models
    models = {
        'linear': LinearRegression(),
        'rf': RandomForestRegressor(n_estimators=10)
    }
    
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = trainer.train_model(model, X, y)
    
    # Save models
    save_path = tmp_path / "models"
    trainer.save_models(trained_models, str(save_path))
    
    # Verify saved files
    assert (save_path / "linear.pkl").exists()
    assert (save_path / "rf.pkl").exists() 