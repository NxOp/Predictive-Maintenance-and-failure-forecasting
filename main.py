"""
Main script for aircraft predictive maintenance.

This script orchestrates the entire workflow from data loading to model evaluation.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

# Import project modules
from src.data_processing import process_data_pipeline
from src.feature_engineering import engineer_features_pipeline
from src.model import LSTMModel, CNNLSTMModel, XGBoostModel, RandomForestModel
from src.evaluation import evaluate_model_on_test_data, compare_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aircraft_maintenance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data/raw', 'data/processed', 'models', 'results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def run_pipeline(data_filepath: str, sequence_length: int = 30, models_to_train: List[str] = None):
    """
    Run the complete predictive maintenance pipeline
    
    Args:
        data_filepath: Path to the raw data file
        sequence_length: Length of sequences for time series models
        models_to_train: List of models to train (if None, train all)
    
    Returns:
        Dictionary of results
    """
    # DEV NOTE: This pipeline took forever to get right. The sequence handling
    # was particularly tricky. If anyone's reading this code, I'm sorry for the
    # occasional frustrated comments. Three weeks of debugging will do that to you.
    
    start_time = datetime.now()
    logger.info(f"Starting predictive maintenance pipeline at {start_time}")
    
    # Step 1: Process data
    logger.info("Step 1: Processing data...")
    processed_data = process_data_pipeline(data_filepath, sequence_length)
    
    # Step 2: Engineer features (for non-sequence models)
    logger.info("Step 2: Engineering features...")
    # We'll use the processed data directly for sequence models
    # For non-sequence models, we'll flatten the data and engineer features
    
    # Get dimensions
    n_samples_train = processed_data['X_train'].shape[0]
    n_samples_val = processed_data['X_val'].shape[0]
    n_samples_test = processed_data['X_test'].shape[0]
    n_features = processed_data['X_train'].shape[2]
    
    logger.info(f"Data dimensions: {n_samples_train} train, {n_samples_val} validation, {n_samples_test} test samples with {n_features} features")
    
    # Step 3: Train models
    logger.info("Step 3: Training models...")
    
    # Define available models
    available_models = {
        'lstm': LSTMModel(sequence_length, n_features),
        'cnn_lstm': CNNLSTMModel(sequence_length, n_features),
        'xgboost': XGBoostModel(),
        'random_forest': RandomForestModel()
    }
    
    # If no models specified, train all
    if models_to_train is None:
        models_to_train = list(available_models.keys())
    
    # Train and evaluate each model
    model_results = {}
    
    for model_name in models_to_train:
        if model_name not in available_models:
            logger.warning(f"Unknown model: {model_name}")
            continue
        
        logger.info(f"Training {model_name} model...")
        model = available_models[model_name]
        
        # Train model
        if model_name in ['lstm', 'cnn_lstm']:
            # For sequence models
            model.train(
                processed_data['X_train'], 
                processed_data['y_train'],
                processed_data['X_val'],
                processed_data['y_val'],
                epochs=100,
                batch_size=32,
                patience=10,
                model_path=f"models/{model_name}_model.h5"
            )
        else:
            # For non-sequence models, flatten the input
            X_train_flat = processed_data['X_train'].reshape(n_samples_train, -1)
            X_val_flat = processed_data['X_val'].reshape(n_samples_val, -1)
            
            model.train(
                X_train_flat,
                processed_data['y_train'],
                X_val_flat,
                processed_data['y_val']
            )
            
            # Save model
            model.save(f"models/{model_name}_model.pkl")
        
        # Step 4: Evaluate model
        logger.info(f"Evaluating {model_name} model...")
        
        # For evaluation, we need to create dummy engine IDs and cycles for plotting
        # In a real scenario, these would come from the test data
        dummy_engine_ids = np.repeat(range(1, 11), n_samples_test // 10)
        dummy_cycles = np.tile(range(1, n_samples_test // 10 + 1), 10)
        
        # Evaluate model
        results = evaluate_model_on_test_data(
            model,
            processed_data['X_test'],
            processed_data['y_test'],
            engine_ids=dummy_engine_ids,
            cycles=dummy_cycles,
            output_dir="results"
        )
        
        # Store results
        model_results[model_name] = results
    
    # Step 5: Compare models
    logger.info("Step 5: Comparing models...")
    best_model = compare_models(model_results, output_path="results/model_comparison.png")
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pipeline completed at {end_time} (Duration: {duration})")
    logger.info(f"Best model: {best_model}")
    
    return {
        'processed_data': processed_data,
        'model_results': model_results,
        'best_model': best_model
    }

def main():
    """Main function"""
    # Create directories
    create_directories()
    
    # Check if data exists
    data_filepath = "data/raw/train_FD001.txt"
    
    if not os.path.exists(data_filepath):
        logger.error(f"Data file not found: {data_filepath}")
        logger.info("Please download the NASA Turbofan Engine Degradation Simulation Dataset from Kaggle:")
        logger.info("https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        return
    
    # Run pipeline
    results = run_pipeline(
        data_filepath=data_filepath,
        sequence_length=30,
        models_to_train=['lstm', 'cnn_lstm', 'xgboost', 'random_forest']
    )
    
    logger.info("Analysis complete. Results are available in the 'results' directory.")

if __name__ == "__main__":
    main()