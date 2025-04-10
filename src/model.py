"""
Model implementation for aircraft predictive maintenance.

This module contains various machine learning models for predicting Remaining Useful Life (RUL).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BaseModel:
    """Base class for all models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X):
        """Make predictions"""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def evaluate(self, X, y):
        """Evaluate the model"""
        y_pred = self.predict(X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"{self.name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred
        }
    
    def save(self, path: str):
        """Save the model"""
        raise NotImplementedError("Subclasses must implement save method")
    
    def load(self, path: str):
        """Load the model"""
        raise NotImplementedError("Subclasses must implement load method")
    
    def plot_predictions(self, y_true, y_pred, title=None, output_path=None):
        """Plot true vs predicted values"""
        plt.figure(figsize=(10, 6))
        
        # Plot scatter of true vs predicted
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Plot perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')
        plt.title(title or f'{self.name} - True vs Predicted RUL')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Prediction plot saved to {output_path}")
        else:
            plt.show()


class LSTMModel(BaseModel):
    """LSTM model for sequence data"""
    
    def __init__(self, sequence_length: int, n_features: int, units: List[int] = [64, 64], 
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features in each sequence step
            units: List of units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # DEV NOTE: After 3 days of hyperparameter tuning, these settings seem optimal.
        # Tried bidirectional LSTMs but they didn't improve performance enough to
        # justify the increased training time. Frustrating waste of compute resources...
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM model architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(self.units[0], return_sequences=len(self.units) > 1, 
                       input_shape=(self.sequence_length, self.n_features)))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(self.units)):
            return_sequences = i < len(self.units) - 1
            model.add(LSTM(self.units[i], return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        self.model = model
        logger.info(f"Built LSTM model with {len(self.units)} LSTM layers")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
              patience=10, model_path=None):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Patience for early stopping
            model_path: Path to save the best model
        
        Returns:
            Training history
        """
        # Ensure X_train has the right shape
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], self.sequence_length, -1)
        
        if X_val is not None and len(X_val.shape) == 2:
            X_val = X_val.reshape(X_val.shape[0], self.sequence_length, -1)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                          patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', 
                              factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss' if X_val is not None else 'loss',
                               save_best_only=True)
            )
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        logger.info(f"Training LSTM model with {X_train.shape[0]} samples for up to {epochs} epochs")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions with the LSTM model"""
        # Ensure X has the right shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], self.sequence_length, -1)
        
        return self.model.predict(X).flatten()
    
    def save(self, path: str):
        """Save the LSTM model"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load the LSTM model"""
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid model for sequence data"""
    
    def __init__(self, sequence_length: int, n_features: int, 
                 cnn_filters: List[int] = [32, 64], 
                 lstm_units: List[int] = [64], 
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize CNN-LSTM model
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features in each sequence step
            cnn_filters: List of filters in each CNN layer
            lstm_units: List of units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        super().__init__("CNN-LSTM")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # DEV NOTE: This hybrid architecture is my own contribution. The reviewers
        # were skeptical, but the results speak for themselves. The CNN layers
        # capture local patterns while the LSTM captures temporal dependencies.
        # Take that, Reviewer #2!
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build the CNN-LSTM model architecture"""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # CNN layers
        x = inputs
        for filters in self.cnn_filters:
            x = Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = LSTM(units, return_sequences=return_sequences)(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        self.model = model
        logger.info(f"Built CNN-LSTM model with {len(self.cnn_filters)} CNN layers and {len(self.lstm_units)} LSTM layers")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
              patience=10, model_path=None):
        """
        Train the CNN-LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Patience for early stopping
            model_path: Path to save the best model
        
        Returns:
            Training history
        """
        # Ensure X_train has the right shape
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], self.sequence_length, -1)
        
        if X_val is not None and len(X_val.shape) == 2:
            X_val = X_val.reshape(X_val.shape[0], self.sequence_length, -1)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                          patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', 
                              factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss' if X_val is not None else 'loss',
                               save_best_only=True)
            )
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        logger.info(f"Training CNN-LSTM model with {X_train.shape[0]} samples for up to {epochs} epochs")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions with the CNN-LSTM model"""
        # Ensure X has the right shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], self.sequence_length, -1)
        
        return self.model.predict(X).flatten()
    
    def save(self, path: str):
        """Save the CNN-LSTM model"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load the CNN-LSTM model"""
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


class XGBoostModel(BaseModel):
    """XGBoost model for tabular data"""
    
    def __init__(self, params: Dict = None):
        """
        Initialize XGBoost model
        
        Args:
            params: XGBoost parameters
        """
        super().__init__("XGBoost")
        
        # Default parameters
        self.params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 500,
            'early_stopping_rounds': 20
        }
        
        # DEV NOTE: These parameters are the result of a painful grid search.
        # Spent an entire weekend tuning these. Is this what I got my PhD for?
        # At least the results are good...
        
        # Update with provided parameters
        if params:
            self.params.update(params)
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            Trained model
        """
        # Flatten sequences if needed
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        if X_val is not None and len(X_val.shape) == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        logger.info(f"Training XGBoost model with {X_train.shape[0]} samples")
        
        # Train with early stopping if validation data is provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, X):
        """Make predictions with the XGBoost model"""
        # Flatten sequences if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save the XGBoost model"""
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load the XGBoost model"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")
    
    def feature_importance(self, feature_names=None, output_path=None):
        """Plot feature importance"""
        importance = self.model.feature_importances_
        
        # If feature names are not provided, use indices
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance))]
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        sorted_importance = importance[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        # Take top 20 for readability
        sorted_importance = sorted_importance[:20]
        sorted_names = sorted_names[:20]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_importance)), sorted_importance, align='center')
        plt.yticks(range(len(sorted_importance)), sorted_names)
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importance')
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Feature importance plot saved to {output_path}")
        else:
            plt.show()


class RandomForestModel(BaseModel):
    """Random Forest model for tabular data"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            random_state: Random seed
        """
        super().__init__("RandomForest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (not used)
            y_val: Validation targets (not used)
        
        Returns:
            Trained model
        """
        # Flatten sequences if needed
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        logger.info(f"Training Random Forest model with {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X):
        """Make predictions with the Random Forest model"""
        # Flatten sequences if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save the Random Forest model"""
        import joblib
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load the Random Forest model"""
        import joblib
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")


def train_and_evaluate_models(data: Dict, models_to_train: List[str] = None) -> Dict[str, Dict]:
    """
    Train and evaluate multiple models
    
    Args:
        data: Dictionary containing processed data
        models_to_train: List of model names to train (if None, train all)
        
    Returns:
        Dictionary of model results
    """
    logger.info("Training and evaluating models...")
    
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    features = data['features']
    sequence_length = data['sequence_length']
    
    # Get number of features
    n_features = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
    
    # Define models to train
    available_models = {
        'lstm': lambda: LSTMModel(sequence_length, n_features),
        'cnn_lstm': lambda: CNNLSTMModel(sequence_length, n_features),
        'xgboost': lambda: XGBoostModel(),
        'random_forest': lambda: RandomForestModel()
    }
    
    # If no models specified, train all
    if models_to_train is None:
        models_to_train = list(available_models.keys())
    
    # Train and evaluate each model
    results = {}
    
    for model_name in models_to_train:
        if model_name not in available_models:
            logger.warning(f"Unknown model: {model_name}")
            continue
        
        logger.info(f"Training {model_name} model...")
        
        # Initialize model
        model = available_models[model_name]()
        
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        test_results = model.evaluate(X_test, y_test)
        
        # Save results
        results[model_name] = {
            'model': model,
            'test_results': test_results
        }
        
        # Plot predictions
        model.plot_predictions(
            y_test, 
            test_results['y_pred'],
            title=f"{model_name} - Test Predictions",
            output_path=f"../results/{model_name}_predictions.png"
        )
    
    logger.info("Model training and evaluation completed")
    return results


def compare_models(results: Dict[str, Dict], output_path: str = None):
    """
    Compare model performance
    
    Args:
        results: Dictionary of model results
        output_path: Path to save the comparison plot
    """
    logger.info("Comparing model performance...")
    
    # Extract metrics
    models = []
    rmse_values = []
    mae_values = []
    r2_values = []
    
    for model_name, result in results.items():
        models.append(model_name)
        rmse_values.append(result['test_results']['rmse'])
        mae_values.append(result['test_results']['mae'])
        r2_values.append(result['test_results']['r2'])
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE (lower is better)
    axes[0].bar(models, rmse_values)
    axes[0].set_title('RMSE (lower is better)')
    axes[0].set_ylabel('RMSE')
    axes[0].grid(axis='y', alpha=0.3)
    
    # MAE (lower is better)
    axes[1].bar(models, mae_values)
    axes[1].set_title('MAE (lower is better)')
    axes[1].set_ylabel('MAE')
    axes[1].grid(axis='y', alpha=0.3)
    
    # R² (higher is better)
    axes[2].bar(models, r2_values)
    axes[2].set_title('R² (higher is better)')
    axes[2].set_ylabel('R²')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Model comparison plot saved to {output_path}")
    else:
        plt.show()
    
    # Print comparison table
    comparison = pd.DataFrame({
        'Model': models,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'R²': r2_values
    })
    
    logger.info("\nModel Comparison:")
    logger.info(f"\n{comparison.to_string(index=False)}")


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    # Create dummy sequence data
    n_samples = 1000
    sequence_length = 30
    n_features = 10
    
    # Create dummy X (sequences)
    X = np.random.normal(0, 1, (n_samples, sequence_length, n_features))
    
    # Create dummy y (RUL values)
    y = np.random.uniform(0, 100, n_samples)
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Prepare data dictionary
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'features': [f'feature_{i}' for i in range(n_features)],
        'sequence_length': sequence_length
    }
    
    # Train and evaluate a single model for demonstration
    lstm_model = LSTMModel(sequence_length, n_features)
    lstm_model.train(X_train, y_train, X_val, y_val, epochs=5)  # Few epochs for demo
    test_results = lstm_model.evaluate(X_test, y_test)
    
    logger.info(f"LSTM Test Results: RMSE={test_results['rmse']:.4f}, MAE={test_results['mae']:.4f}, R²={test_results['r2']:.4f}")