import os
from src.data_preparation import DataPreprocessor
from src.model_training import ModelTrainer

def main():
    print("Starting HDB Flat Resale Price Prediction Pipeline...")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Data preparation
    print("\n1. Data Preparation")
    print("------------------")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Model training and evaluation
    print("\n2. Model Training and Evaluation")
    print("-------------------------------")
    trainer = ModelTrainer()
    trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    # Save models
    print("\n3. Saving Models")
    print("---------------")
    trainer.save_models()
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model()
    print(f"\nBest performing model: {best_model_name}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 