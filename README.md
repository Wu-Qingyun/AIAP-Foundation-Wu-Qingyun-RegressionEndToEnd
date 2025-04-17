# HDB Flat Resale Price Prediction

This project aims to predict the resale prices of HDB flats in Singapore using machine learning models. The project includes comprehensive data analysis, feature engineering, and model evaluation.

## Project Structure
```
root /
    |- eda.ipynb          # Exploratory data analysis notebook
    |- README.md          # Project documentation
    |- requirements.txt   # Project dependencies
    |- data /
           |- data.csv    # Dataset
    |- src /
           |- data_preparation.py  # Data preprocessing module
           |- model_training.py    # Model training module
           |- config.yaml          # Configuration file
    |- main.py            # Main execution script
    |- tests/             # Unit tests
           |- test_data_preparation.py
           |- test_model_training.py
    |- models/            # Saved model files
    |- logs/              # Log files
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hdb-price-prediction.git
cd hdb-price-prediction
```

2. Create a virtual environment (recommended):
```bash
conda create -n aiap python=3.9
conda activate aiap
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Run the main script:
```bash
python main.py
```

## Project Components

### EDA (eda.ipynb)
- Comprehensive exploratory data analysis
- Data visualization and statistical analysis
- Feature relationship analysis
- Key findings:
  - Dataset contains 84,465 records after removing duplicates
  - Missing values in `town_name` (0.88%) and `flatm_name` (0.58%)
  - Strong correlation between `floor_area_sqm` and `resale_price`
  - Significant price variation by `flat_type` and `storey_range`
  - Derived features: price per square meter, flat age

### Data Preparation (src/data_preparation.py)
- Data cleaning and preprocessing
  - Handles missing values using mode for categorical and median for numerical
  - Removes outliers using IQR method
  - Converts data types (remaining_lease, block_number, month)
- Feature engineering
  - Price per square meter calculation
  - Flat age computation
  - Block number extraction
- Data transformation
  - Label encoding for categorical variables
  - Standard scaling for numerical features

### Model Training (src/model_training.py)
- Implementation of multiple regression models
  - Linear Regression (baseline)
  - XGBoost
  - LightGBM
  - CatBoost
- Model evaluation and comparison
  - Cross-validation with 5 folds
  - Multiple evaluation metrics
- Hyperparameter tuning
  - Grid search for optimal parameters
  - Parallel processing for efficiency

### Configuration (src/config.yaml)
- Model parameters
  - Learning rates
  - Tree depths
  - Number of estimators
- File paths
  - Data locations
  - Model save paths
- Training configurations
  - Cross-validation settings
  - Evaluation metrics
  - Parallel processing options

## Models Evaluated

1. Linear Regression (Baseline)
   - Simple and interpretable
   - Serves as performance benchmark

2. XGBoost
   - Handles non-linear relationships
   - Robust to outliers
   - Parameters:
     - n_estimators: 1000
     - learning_rate: 0.01
     - max_depth: 7

3. LightGBM
   - Fast training speed
   - Memory efficient
   - Parameters:
     - n_estimators: 1000
     - learning_rate: 0.01
     - max_depth: 7

4. CatBoost
   - Handles categorical features well
   - Robust to overfitting
   - Parameters:
     - iterations: 1000
     - learning_rate: 0.01
     - depth: 7

## Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

## Results
[To be updated after model training]

## Testing
The project includes unit tests for data preparation and model training modules. Run tests using:
```bash
pytest tests/
```

## Logging
Logs are stored in the `logs/` directory with timestamps. They include:
- Data preprocessing steps
- Model training progress
- Evaluation metrics
- Error messages

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Version Control
- Git is used for version control
- Main branch: production-ready code
- Development branch: ongoing development
- Feature branches: new features and fixes

## License
[Your chosen license] 