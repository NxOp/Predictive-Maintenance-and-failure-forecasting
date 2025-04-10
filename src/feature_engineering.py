"""
Feature engineering module for aircraft predictive maintenance.

This module handles feature creation, selection, and transformation to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from cycle information.
    
    Args:
        df: DataFrame with cycle information
        
    Returns:
        DataFrame with additional time-based features
    """
    logger.info("Creating time-based features...")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # DEV NOTE: After reviewing literature, these cycle-based features seem to be
    # most predictive. Still not convinced about the exponential decay, but the
    # reviewers insisted on it. Will revisit if results are poor.
    
    # For each engine, create features based on cycle
    engine_groups = df_new.groupby('engine_id')
    
    # Create list to hold transformed dataframes
    transformed_dfs = []
    
    for engine_id, group in engine_groups:
        group = group.sort_values('cycle')
        
        # Calculate percentage of max cycle
        max_cycle = group['cycle'].max()
        group['cycle_pct'] = group['cycle'] / max_cycle
        
        # Create exponential decay feature (higher value as engine approaches failure)
        group['exp_decay'] = np.exp(group['cycle'] / max_cycle * 5) - 1
        
        # Rolling statistics for sensor readings
        for col in group.columns:
            if col.startswith('sensor'):
                # 5-cycle rolling mean
                group[f'{col}_roll_mean_5'] = group[col].rolling(window=5, min_periods=1).mean()
                
                # 5-cycle rolling standard deviation
                group[f'{col}_roll_std_5'] = group[col].rolling(window=5, min_periods=1).std().fillna(0)
                
                # Rate of change (first derivative)
                group[f'{col}_rate'] = group[col].diff().fillna(0)
                
                # Acceleration (second derivative)
                group[f'{col}_accel'] = group[f'{col}_rate'].diff().fillna(0)
        
        transformed_dfs.append(group)
    
    # Combine all transformed dataframes
    df_new = pd.concat(transformed_dfs)
    
    logger.info(f"Created time-based features. New shape: {df_new.shape}")
    return df_new

def create_sensor_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between sensors.
    
    Args:
        df: DataFrame with sensor readings
        
    Returns:
        DataFrame with additional interaction features
    """
    logger.info("Creating sensor interaction features...")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # DEV NOTE: These interactions were suggested in the literature, but honestly
    # I'm not convinced they add much value. The ratios make physical sense for
    # some sensor pairs, but the products? Keeping them for now to be thorough.
    
    # Get sensor columns
    sensor_cols = [col for col in df_new.columns if col.startswith('sensor')]
    
    # Create important sensor ratios (based on domain knowledge)
    # These pairs are known to have physical relationships in aircraft engines
    important_pairs = [
        ('sensor2', 'sensor4'),  # Temperature ratio
        ('sensor7', 'sensor12'),  # Pressure ratio
        ('sensor9', 'sensor14'),  # Fan speed to core speed
        ('sensor3', 'sensor15')   # Fuel flow to exhaust temperature
    ]
    
    for col1, col2 in important_pairs:
        if col1 in df_new.columns and col2 in df_new.columns:
            # Create ratio feature (handling division by zero)
            denominator = df_new[col2].copy()
            # Replace zeros with a small value to avoid division by zero
            denominator = denominator.replace(0, 1e-10)
            df_new[f'{col1}_to_{col2}_ratio'] = df_new[col1] / denominator
            
            # Create product feature
            df_new[f'{col1}_by_{col2}_product'] = df_new[col1] * df_new[col2]
    
    logger.info(f"Created sensor interaction features. New shape: {df_new.shape}")
    return df_new

def select_features(df: pd.DataFrame, target_col: str, k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the most important features using statistical tests.
    
    Args:
        df: DataFrame with features
        target_col: Name of the target column
        k: Number of features to select
        
    Returns:
        Tuple of (DataFrame with selected features, list of selected feature names)
    """
    logger.info(f"Selecting top {k} features...")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # DEV NOTE: Tried mutual information, but f_regression seems to work better
    # for this continuous target. Spent way too much time on this decision...
    
    # Get feature columns (exclude non-numeric and target)
    feature_cols = [col for col in df_new.select_dtypes(include=[np.number]).columns 
                   if col != target_col and col != 'engine_id']
    
    # Create feature matrix and target vector
    X = df_new[feature_cols]
    y = df_new[target_col]
    
    # Select k best features
    selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
    selector.fit(X, y)
    
    # Get selected feature indices and names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]
    
    logger.info(f"Selected features: {selected_features}")
    
    # Return DataFrame with only selected features and the target
    return df_new[selected_features + [target_col]], selected_features

def apply_pca(df: pd.DataFrame, target_col: str, n_components: int = 10) -> pd.DataFrame:
    """
    Apply Principal Component Analysis for dimensionality reduction.
    
    Args:
        df: DataFrame with features
        target_col: Name of the target column
        n_components: Number of PCA components to keep
        
    Returns:
        DataFrame with PCA components
    """
    logger.info(f"Applying PCA with {n_components} components...")
    
    # DEV NOTE: I'm still not convinced PCA is the right approach here since
    # interpretability is important for maintenance decisions. But the computational
    # benefits are significant, so including it as an option.
    
    # Get feature columns (exclude non-numeric and target)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col != target_col and col != 'engine_id']
    
    # Create feature matrix
    X = df[feature_cols]
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, len(feature_cols)))
    pca_result = pca.fit_transform(X)
    
    # Create DataFrame with PCA components
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
    )
    
    # Add engine_id and target back if they exist in original df
    if 'engine_id' in df.columns:
        pca_df['engine_id'] = df['engine_id'].values
    
    pca_df[target_col] = df[target_col].values
    
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return pca_df

def detect_and_handle_multicollinearity(df: pd.DataFrame, target_col: str, threshold: float = 0.85) -> pd.DataFrame:
    """
    Detect and handle multicollinearity among features.
    
    Args:
        df: DataFrame with features
        target_col: Name of the target column
        threshold: Correlation threshold above which to consider features as collinear
        
    Returns:
        DataFrame with reduced multicollinearity
    """
    logger.info(f"Detecting multicollinearity with threshold {threshold}...")
    
    # DEV NOTE: This is frustrating. Some of these sensors clearly measure related
    # phenomena, but removing them feels like throwing away information. Going with
    # a conservative threshold for now, but might need to revisit.
    
    # Get feature columns (exclude non-numeric and target)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col != target_col and col != 'engine_id']
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    
    # Create a mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    logger.info(f"Dropping {len(to_drop)} collinear features: {to_drop}")
    
    # Drop highly correlated features
    df_reduced = df.drop(columns=to_drop)
    
    return df_reduced

def plot_feature_importance(df: pd.DataFrame, target_col: str, output_path: str = None):
    """
    Plot feature correlations with the target variable.
    
    Args:
        df: DataFrame with features
        target_col: Name of the target column
        output_path: Path to save the plot (if None, plot is displayed)
    """
    logger.info("Plotting feature importance...")
    
    # Get feature columns (exclude non-numeric and target)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col != target_col and col != 'engine_id']
    
    # Calculate correlations with target
    correlations = []
    for col in feature_cols:
        corr = stats.pearsonr(df[col], df[target_col])[0]
        correlations.append((col, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 15 for readability
    top_features = correlations[:15]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh([x[0] for x in top_features], [x[1] for x in top_features])
    
    # Add correlation values to the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                 ha='left', va='center')
    
    plt.xlabel('Absolute Correlation with RUL')
    plt.title('Top 15 Features by Correlation with Remaining Useful Life')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()

def engineer_features_pipeline(df: pd.DataFrame, target_col: str = 'RUL') -> Dict:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: DataFrame with raw features
        target_col: Name of the target column
        
    Returns:
        Dictionary containing processed data and selected features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Create time-based features
    df_time = create_time_features(df)
    
    # Create sensor interaction features
    df_interactions = create_sensor_interaction_features(df_time)
    
    # Handle multicollinearity
    df_reduced = detect_and_handle_multicollinearity(df_interactions, target_col)
    
    # Select best features
    df_selected, selected_features = select_features(df_reduced, target_col)
    
    # Apply PCA (optional, commented out by default)
    # df_pca = apply_pca(df_reduced, target_col)
    
    logger.info("Feature engineering pipeline completed")
    
    # Return processed data and metadata
    return {
        'data': df_selected,
        'selected_features': selected_features,
        'original_shape': df.shape,
        'final_shape': df_selected.shape
    }

if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    n_samples = 1000
    n_engines = 10
    
    # Create dummy data
    dummy_data = {
        'engine_id': np.repeat(range(1, n_engines+1), n_samples // n_engines),
        'cycle': np.tile(range(1, n_samples // n_engines + 1), n_engines),
    }
    
    # Add sensor readings
    for i in range(1, 22):
        dummy_data[f'sensor{i}'] = np.random.normal(100, 20, n_samples) + \
                                  dummy_data['cycle'] * np.random.uniform(0.01, 0.1, 1)
    
    # Create DataFrame
    df = pd.DataFrame(dummy_data)
    
    # Calculate RUL (max_cycle - cycle)
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop('max_cycle', axis=1)
    
    # Run feature engineering pipeline
    result = engineer_features_pipeline(df)
    
    logger.info(f"Original shape: {result['original_shape']}")
    logger.info(f"Final shape: {result['final_shape']}")
    logger.info(f"Selected features: {result['selected_features']}")
    
    # Plot feature importance
    plot_feature_importance(result['data'], 'RUL')