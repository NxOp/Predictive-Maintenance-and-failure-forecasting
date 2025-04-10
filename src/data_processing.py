"""
Data processing module for aircraft predictive maintenance.

This module contains functions for loading, cleaning, and preprocessing the NASA Turbofan
Engine Degradation Simulation Dataset for predictive maintenance tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Union
import logging
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Column names for the NASA dataset
# The dataset doesn't have headers, so we define them here
NASA_COLUMNS = [
    'engine_id', 'cycle', 
    'setting1', 'setting2', 'setting3', 
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
]

# Sensors that are known to be useful based on literature
USEFUL_SENSORS = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21'
]

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the NASA Turbofan Engine dataset
    
    Args:
        filepath: Path to the dataset file
        
    Returns:
        DataFrame with the loaded data
    """
    logger.info(f"Loading data from {filepath}")
    
    try:
        # Load data with the predefined column names
        data = pd.read_csv(filepath, sep=' ', header=None, names=NASA_COLUMNS)
        
        # Remove columns with NaN values (the dataset has trailing spaces)
        data = data.dropna(axis=1)
        
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers
    
    Args:
        data: Raw data DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    
    # Make a copy to avoid modifying the original
    cleaned_data = data.copy()
    
    # Check for missing values
    missing_values = cleaned_data.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Found {missing_values.sum()} missing values")
        # Fill missing values with median for each column
        for col in cleaned_data.columns:
            if missing_values[col] > 0:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
    
    # Check for and handle outliers using IQR method
    for col in cleaned_data.columns:
        if col not in ['engine_id', 'cycle']:  # Skip non-numeric columns
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
            
            if outliers > 0:
                logger.info(f"Found {outliers} outliers in column {col}")
                
                # Replace outliers with bounds
                cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
    
    logger.info("Data cleaning completed")
    return cleaned_data

def calculate_rul(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each engine
    
    Args:
        data: Cleaned data DataFrame
        
    Returns:
        DataFrame with RUL column added
    """
    logger.info("Calculating RUL...")
    
    # Make a copy to avoid modifying the original
    rul_data = data.copy()
    
    # Group by engine_id and find the maximum cycle for each engine
    max_cycles = rul_data.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Merge with the original data
    rul_data = rul_data.merge(max_cycles, on='engine_id', how='left')
    
    # Calculate RUL as the difference between max cycle and current cycle
    rul_data['RUL'] = rul_data['max_cycle'] - rul_data['cycle']
    
    # Drop the max_cycle column as it's no longer needed
    rul_data = rul_data.drop('max_cycle', axis=1)
    
    logger.info("RUL calculation completed")
    return rul_data

def normalize_data(data: pd.DataFrame, scaler_type: str = 'minmax') -> Tuple[pd.DataFrame, Any]:
    """
    Normalize the sensor data
    
    Args:
        data: DataFrame with RUL
        scaler_type: Type of scaler to use ('minmax' or 'standard')
        
    Returns:
        Tuple of normalized DataFrame and the fitted scaler
    """
    logger.info(f"Normalizing data using {scaler_type} scaling...")
    
    # Make a copy to avoid modifying the original
    normalized_data = data.copy()
    
    # Select columns to normalize (all except engine_id, cycle, and RUL)
    cols_to_normalize = [col for col in normalized_data.columns 
                         if col not in ['engine_id', 'cycle', 'RUL']]
    
    # Initialize the appropriate scaler
    if scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit and transform the selected columns
    normalized_data[cols_to_normalize] = scaler.fit_transform(normalized_data[cols_to_normalize])
    
    logger.info("Data normalization completed")
    return normalized_data, scaler

def create_sequences(data: pd.DataFrame, sequence_length: int, 
                    step: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for time series modeling
    
    Args:
        data: Normalized DataFrame with RUL
        sequence_length: Length of each sequence
        step: Step size between sequences
        
    Returns:
        Tuple of (X_sequences, y_targets, engine_ids)
    """
    logger.info(f"Creating sequences with length {sequence_length} and step {step}...")
    
    X_sequences = []
    y_targets = []
    engine_ids = []
    
    # Group data by engine_id
    grouped_data = data.groupby('engine_id')
    
    # For each engine, create sequences
    for engine_id, group in tqdm(grouped_data, desc="Creating sequences"):
        # Sort by cycle
        group = group.sort_values('cycle')
        
        # Extract features and target
        features = group.drop(['engine_id', 'RUL'], axis=1).values
        targets = group['RUL'].values
        
        # Create sequences
        for i in range(0, len(group) - sequence_length + 1, step):
            X_sequences.append(features[i:i+sequence_length])
            # Use the RUL at the end of the sequence as the target
            y_targets.append(targets[i+sequence_length-1])
            engine_ids.append(engine_id)
    
    # Convert to numpy arrays
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)
    engine_ids = np.array(engine_ids)
    
    logger.info(f"Created {len(X_sequences)} sequences")
    return X_sequences, y_targets, engine_ids

def split_data(X: np.ndarray, y: np.ndarray, engine_ids: np.ndarray = None, 
              test_size: float = 0.2, val_size: float = 0.2, 
              random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Split data into training, validation, and test sets
    
    Args:
        X: Feature sequences
        y: Target values
        engine_ids: Engine IDs for each sequence
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed
        
    Returns:
        Dictionary with train, validation, and test sets
    """
    logger.info("Splitting data into train, validation, and test sets...")
    
    # First split: training + validation vs test
    if engine_ids is not None:
        # Split by engine_id to avoid data leakage
        unique_engines = np.unique(engine_ids)
        n_engines = len(unique_engines)
        
        # Shuffle engines
        np.random.seed(random_state)
        np.random.shuffle(unique_engines)
        
        # Split engines into train+val and test
        n_test = int(n_engines * test_size)
        test_engines = unique_engines[:n_test]
        train_val_engines = unique_engines[n_test:]
        
        # Create masks
        test_mask = np.isin(engine_ids, test_engines)
        train_val_mask = ~test_mask
        
        # Split data
        X_train_val, X_test = X[train_val_mask], X[test_mask]
        y_train_val, y_test = y[train_val_mask], y[test_mask]
        
        if engine_ids is not None:
            engine_ids_train_val = engine_ids[train_val_mask]
        
        # Second split: training vs validation
        if engine_ids is not None:
            # Split by engine_id
            unique_train_val_engines = np.unique(engine_ids_train_val)
            n_train_val_engines = len(unique_train_val_engines)
            
            # Calculate number of validation engines
            n_val = int(n_train_val_engines * (val_size / (1 - test_size)))
            
            # Split engines
            val_engines = unique_train_val_engines[:n_val]
            train_engines = unique_train_val_engines[n_val:]
            
            # Create masks
            val_mask = np.isin(engine_ids_train_val, val_engines)
            train_mask = ~val_mask
            
            # Split data
            X_train, X_val = X_train_val[train_mask], X_train_val[val_mask]
            y_train, y_val = y_train_val[train_mask], y_train_val[val_mask]
            
            if engine_ids is not None:
                engine_ids_train = engine_ids_train_val[train_mask]
                engine_ids_val = engine_ids_train_val[val_mask]
        else:
            # Regular split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=random_state
            )
    else:
        # Regular split without considering engine_ids
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=random_state
        )
    
    logger.info(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
    
    # Return as dictionary
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Add engine_ids if provided
    if engine_ids is not None:
        result['engine_ids_train'] = engine_ids_train
        result['engine_ids_val'] = engine_ids_val
        result['engine_ids_test'] = engine_ids[test_mask]
    
    return result

def process_data_pipeline(data_filepath: str, sequence_length: int = 30, 
                         scaler_type: str = 'minmax', test_size: float = 0.2, 
                         val_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Complete data processing pipeline
    
    Args:
        data_filepath: Path to the raw data file
        sequence_length: Length of sequences for time series models
        scaler_type: Type of scaler to use
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed
        
    Returns:
        Dictionary with processed data
    """
    logger.info("Starting data processing pipeline...")
    
    # Step 1: Load data
    data = load_data(data_filepath)
    
    # Step 2: Clean data
    cleaned_data = clean_data(data)
    
    # Step 3: Calculate RUL
    rul_data = calculate_rul(cleaned_data)
    
    # Step 4: Normalize data
    normalized_data, scaler = normalize_data(rul_data, scaler_type)
    
    # Step 5: Create sequences
    X_sequences, y_targets, engine_ids = create_sequences(normalized_data, sequence_length)
    
    # Step 6: Split data
    split_data_dict = split_data(
        X_sequences, y_targets, engine_ids, 
        test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # Add additional information to the result
    result = split_data_dict.copy()
    result['scaler'] = scaler
    result['features'] = [col for col in normalized_data.columns 
                         if col not in ['engine_id', 'cycle', 'RUL']]
    result['sequence_length'] = sequence_length
    
    logger.info("Data processing pipeline completed")
    return result

def visualize_data(data: pd.DataFrame, output_dir: str = None):
    """
    Visualize the dataset
    
    Args:
        data: DataFrame to visualize
        output_dir: Directory to save plots
    """
    logger.info("Visualizing data...")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Distribution of RUL
    plt.figure(figsize=(10, 6))
    sns.histplot(data['RUL'], kde=True)
    plt.title('Distribution of Remaining Useful Life (RUL)')
    plt.xlabel('RUL')
    plt.ylabel('Frequency')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'rul_distribution.png'))
        plt.close()
    else:
        plt.show()
    
    # 2. RUL vs Cycle for a sample of engines
    plt.figure(figsize=(12, 8))
    sample_engines = np.random.choice(data['engine_id'].unique(), min(5, len(data['engine_id'].unique())), replace=False)
    
    for engine_id in sample_engines:
        engine_data = data[data['engine_id'] == engine_id]
        plt.plot(engine_data['cycle'], engine_data['RUL'], marker='o', linestyle='-', label=f'Engine {engine_id}')
    
    plt.title('RUL vs Cycle for Sample Engines')
    plt.xlabel('Cycle')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'rul_vs_cycle.png'))
        plt.close()
    else:
        plt.show()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(14, 10))
    correlation = data.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()
    else:
        plt.show()
    
    # 4. Sensor readings over time for a sample engine
    sample_engine = sample_engines[0]
    engine_data = data[data['engine_id'] == sample_engine].sort_values('cycle')
    
    # Select a subset of sensors
    sensors_to_plot = USEFUL_SENSORS[:6]  # Plot first 6 useful sensors
    
    plt.figure(figsize=(14, 10))
    for sensor in sensors_to_plot:
        plt.plot(engine_data['cycle'], engine_data[sensor], marker='.', linestyle='-', label=sensor)
    
    plt.title(f'Sensor Readings Over Time for Engine {sample_engine}')
    plt.xlabel('Cycle')
    plt.ylabel('Normalized Sensor Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'sensor_readings.png'))
        plt.close()
    else:
        plt.show()
    
    logger.info("Data visualization completed")

if __name__ == "__main__":
    # Example usage
    data_filepath = "../data/raw/train_FD001.txt"
    
    if os.path.exists(data_filepath):
        # Load and process data
        data = load_data(data_filepath)
        cleaned_data = clean_data(data)
        rul_data = calculate_rul(cleaned_data)
        
        # Visualize data
        visualize_data(rul_data, output_dir="../results/data_visualization")
        
        # Process data pipeline
        processed_data = process_data_pipeline(data_filepath)
        
        print(f"Processed data shapes:")
        print(f"X_train: {processed_data['X_train'].shape}")
        print(f"y_train: {processed_data['y_train'].shape}")
        print(f"X_val: {processed_data['X_val'].shape}")
        print(f"y_val: {processed_data['y_val'].shape}")
        print(f"X_test: {processed_data['X_test'].shape}")
        print(f"y_test: {processed_data['y_test'].shape}")
    else:
        print(f"Data file not found: {data_filepath}")
        print("Please download the NASA Turbofan Engine Degradation Simulation Dataset.")