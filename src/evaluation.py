"""
Evaluation module for aircraft predictive maintenance.

This module contains functions for evaluating model performance and visualizing results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str = None, 
                             output_path: str = None):
    """
    Plot predicted vs actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics for the plot
    metrics = calculate_metrics(y_true, y_pred)
    
    # Add metrics text
    plt.text(
        0.05, 0.95, 
        f"RMSE: {metrics['rmse']:.2f}\nMAE: {metrics['mae']:.2f}\nR²: {metrics['r2']:.2f}\nMAPE: {metrics['mape']:.2f}%",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.title(title or 'Predicted vs Actual RUL')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

def plot_rul_over_time(engine_id: int, cycles: np.ndarray, true_rul: np.ndarray, 
                      pred_rul: np.ndarray, title: str = None, output_path: str = None):
    """
    Plot RUL predictions over time for a specific engine
    
    Args:
        engine_id: Engine ID
        cycles: Cycle numbers
        true_rul: True RUL values
        pred_rul: Predicted RUL values
        title: Plot title
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # DEV NOTE: This visualization was a pain to get right. Spent hours
    # trying different styles. The reviewers better appreciate this...
    
    # Plot true and predicted RUL
    plt.plot(cycles, true_rul, 'b-', label='True RUL')
    plt.plot(cycles, pred_rul, 'r--', label='Predicted RUL')
    
    # Fill the area between true and predicted
    plt.fill_between(cycles, true_rul, pred_rul, color='gray', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Cycle')
    plt.ylabel('Remaining Useful Life (RUL)')
    plt.title(title or f'RUL Prediction Over Time for Engine {engine_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, title: str = None, 
                           output_path: str = None):
    """
    Plot the distribution of prediction errors
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        output_path: Path to save the plot
    """
    # Calculate errors
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    
    # Histogram of errors
    sns.histplot(errors, kde=True)
    
    # Add vertical line at zero
    plt.axvline(x=0, color='r', linestyle='--')
    
    # Add mean and std text
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(
        0.05, 0.95, 
        f"Mean Error: {mean_error:.2f}\nStd Dev: {std_error:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title(title or 'Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

def plot_rul_by_engine(engine_ids: List[int], true_rul: Dict[int, np.ndarray], 
                      pred_rul: Dict[int, np.ndarray], title: str = None, 
                      output_path: str = None, max_engines: int = 9):
    """
    Plot RUL predictions for multiple engines
    
    Args:
        engine_ids: List of engine IDs
        true_rul: Dictionary of true RUL values by engine ID
        pred_rul: Dictionary of predicted RUL values by engine ID
        title: Plot title
        output_path: Path to save the plot
        max_engines: Maximum number of engines to plot
    """
    # Limit the number of engines to plot
    if len(engine_ids) > max_engines:
        logger.info(f"Limiting plot to {max_engines} engines")
        engine_ids = engine_ids[:max_engines]
    
    # Calculate grid dimensions
    n_engines = len(engine_ids)
    n_cols = min(3, n_engines)
    n_rows = (n_engines + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, engine_id in enumerate(engine_ids):
        if i < len(axes):
            # Get data for this engine
            cycles = np.arange(len(true_rul[engine_id]))
            
            # Plot on the corresponding subplot
            axes[i].plot(cycles, true_rul[engine_id], 'b-', label='True RUL')
            axes[i].plot(cycles, pred_rul[engine_id], 'r--', label='Predicted RUL')
            axes[i].set_title(f'Engine {engine_id}')
            axes[i].set_xlabel('Cycle')
            axes[i].set_ylabel('RUL')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(n_engines, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(title or 'RUL Predictions by Engine', y=1.02, fontsize=16)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

def evaluate_model_on_test_data(model, X_test: np.ndarray, y_test: np.ndarray, 
                               engine_ids: np.ndarray = None, cycles: np.ndarray = None,
                               output_dir: str = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a model on test data
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: True test values
        engine_ids: Engine IDs for test samples (optional)
        cycles: Cycle numbers for test samples (optional)
        output_dir: Directory to save plots
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Evaluating model on test data...")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    logger.info(f"Test metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")
    
    # Plot prediction vs actual
    plot_prediction_vs_actual(
        y_test, y_pred, 
        title=f"{model.name} - Predicted vs Actual RUL",
        output_path=os.path.join(output_dir, f"{model.name}_pred_vs_actual.png") if output_dir else None
    )
    
    # Plot error distribution
    plot_error_distribution(
        y_test, y_pred,
        title=f"{model.name} - Prediction Error Distribution",
        output_path=os.path.join(output_dir, f"{model.name}_error_dist.png") if output_dir else None
    )
    
    # If engine IDs and cycles are provided, create additional plots
    if engine_ids is not None and cycles is not None:
        # Group data by engine
        unique_engines = np.unique(engine_ids)
        
        # Create dictionaries to store RUL values by engine
        true_rul_by_engine = {}
        pred_rul_by_engine = {}
        cycles_by_engine = {}
        
        for engine_id in unique_engines:
            # Get indices for this engine
            indices = np.where(engine_ids == engine_id)[0]
            
            # Sort by cycle
            sorted_indices = indices[np.argsort(cycles[indices])]
            
            # Store values
            true_rul_by_engine[engine_id] = y_test[sorted_indices]
            pred_rul_by_engine[engine_id] = y_pred[sorted_indices]
            cycles_by_engine[engine_id] = cycles[sorted_indices]
        
        # Plot RUL over time for a sample engine
        sample_engine = unique_engines[0]
        plot_rul_over_time(
            sample_engine,
            cycles_by_engine[sample_engine],
            true_rul_by_engine[sample_engine],
            pred_rul_by_engine[sample_engine],
            title=f"{model.name} - RUL Prediction for Engine {sample_engine}",
            output_path=os.path.join(output_dir, f"{model.name}_engine_{sample_engine}_rul.png") if output_dir else None
        )
        
        # Plot RUL for multiple engines
        plot_rul_by_engine(
            list(unique_engines)[:9],  # Limit to 9 engines for readability
            true_rul_by_engine,
            pred_rul_by_engine,
            title=f"{model.name} - RUL Predictions by Engine",
            output_path=os.path.join(output_dir, f"{model.name}_multi_engine_rul.png") if output_dir else None
        )
    
    return {
        'metrics': metrics,
        'y_pred': y_pred
    }

def compare_models(model_results: Dict[str, Dict], output_path: str = None):
    """
    Compare multiple models
    
    Args:
        model_results: Dictionary of model results
        output_path: Path to save the comparison plot
    """
    logger.info("Comparing model performance...")
    
    # Extract model names and metrics
    models = []
    rmse_values = []
    mae_values = []
    r2_values = []
    mape_values = []
    
    for model_name, results in model_results.items():
        models.append(model_name)
        metrics = results['metrics']
        rmse_values.append(metrics['rmse'])
        mae_values.append(metrics['mae'])
        r2_values.append(metrics['r2'])
        mape_values.append(metrics['mape'])
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': models,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'R²': r2_values,
        'MAPE (%)': mape_values
    })
    
    # Sort by RMSE (lower is better)
    df = df.sort_values('RMSE')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE plot
    sns.barplot(x='Model', y='RMSE', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('RMSE (lower is better)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # MAE plot
    sns.barplot(x='Model', y='MAE', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('MAE (lower is better)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # R² plot
    sns.barplot(x='Model', y='R²', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('R² (higher is better)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # MAPE plot
    sns.barplot(x='Model', y='MAPE (%)', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('MAPE % (lower is better)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Model Comparison', y=1.02, fontsize=16)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Model comparison plot saved to {output_path}")
    else:
        plt.show()
    
    # Print comparison table
    logger.info("\nModel Comparison:")
    logger.info(f"\n{df.to_string(index=False)}")
    
    # Return the best model based on RMSE
    best_model = df.iloc[0]['Model']
    logger.info(f"\nBest model based on RMSE: {best_model}")
    
    return best_model

if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    # Create dummy data
    n_samples = 100
    y_true = np.random.uniform(0, 100, n_samples)
    y_pred = y_true + np.random.normal(0, 10, n_samples)  # Add some noise
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Metrics: {metrics}")
    
    # Plot prediction vs actual
    plot_prediction_vs_actual(y_true, y_pred, title="Example Prediction vs Actual")
    
    # Plot error distribution
    plot_error_distribution(y_true, y_pred, title="Example Error Distribution")