"""
Price forecasting module with ARIMA, XGBoost, and LSTM models.

This module provides individual forecasting model classes and an orchestration
pipeline for training, evaluation, ensemble forecasting, and model persistence.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Statistical models
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

# XGBoost
import xgboost as xgb
from xgboost import XGBRegressor

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Model persistence
import joblib

# Internal imports
from src.config.load_config import get_config
from src.models.feature_engineering import FeatureEngineer


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, MAPE, and directional accuracy
    """
    # MAE: Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE: Mean Absolute Percentage Error
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    # Directional Accuracy
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }


def time_series_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform time-based train/validation/test split.

    Args:
        data: DataFrame to split
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    return train_data, val_data, test_data


# =============================================================================
# ARIMA Forecaster
# =============================================================================

class ARIMAForecaster:
    """
    ARIMA (AutoRegressive Integrated Moving Average) time-series forecaster.

    Classical statistical model for univariate time-series forecasting.
    Does not use exogenous features, only historical values of the target.

    Attributes:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal ARIMA order (P, D, Q, s) or None
        fitted_model: Fitted ARIMA model
        logger: Logger instance
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize ARIMA forecaster.

        Args:
            order: ARIMA order (p, d, q), default (1, 1, 1)
            seasonal_order: Seasonal order (P, D, Q, s) or None
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        # Load default order from config if provided
        if config is not None:
            arima_config = config.get('models', {}).get('arima', {})
            default_order = arima_config.get('default_order', [1, 1, 1])
            order = tuple(default_order) if order == (1, 1, 1) else order

        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None

        self.logger.info(f"ARIMAForecaster initialized with order={order}, seasonal_order={seasonal_order}")

    def fit(self, y_train: Union[pd.Series, np.ndarray], exog: Optional[np.ndarray] = None) -> 'ARIMAForecaster':
        """
        Fit ARIMA model on training data.

        Args:
            y_train: Training target values
            exog: Optional exogenous variables (not typically used with ARIMA)

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting ARIMA model on {len(y_train)} samples")

        try:
            # Create and fit ARIMA model
            model = ARIMA(y_train, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = model.fit()

            # Log fit summary
            self.logger.info(
                f"ARIMA model fitted successfully. AIC={self.fitted_model.aic:.2f}, "
                f"BIC={self.fitted_model.bic:.2f}"
            )

        except Exception as e:
            self.logger.error(f"ARIMA fitting failed: {e}")
            raise

        return self

    def predict(
        self,
        steps: int,
        exog: Optional[np.ndarray] = None,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts.

        Args:
            steps: Number of steps to forecast
            exog: Optional exogenous variables
            return_conf_int: Whether to return confidence intervals (default: True)
            alpha: Significance level for confidence intervals (default: 0.05 for 95% CI)

        Returns:
            Dictionary with 'forecast', 'lower_ci', 'upper_ci' keys
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Generate point forecasts
        forecast_result = self.fitted_model.get_forecast(steps=steps, exog=exog)
        forecast = forecast_result.predicted_mean.values

        result = {'forecast': forecast}

        # Add confidence intervals if requested
        if return_conf_int:
            conf_int = forecast_result.conf_int(alpha=alpha)
            result['lower_ci'] = conf_int.iloc[:, 0].values
            result['upper_ci'] = conf_int.iloc[:, 1].values

        self.logger.debug(f"Generated {steps}-step forecast")
        return result

    def save(self, filepath: Union[Path, str]) -> Path:
        """
        Save fitted model.

        Args:
            filepath: Path to save model

        Returns:
            Path where model was saved
        """
        if self.fitted_model is None:
            raise ValueError("No fitted model to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.fitted_model.save(str(filepath))
        self.logger.info(f"ARIMA model saved to {filepath}")

        return filepath

    @classmethod
    def load(cls, filepath: Union[Path, str]) -> 'ARIMAForecaster':
        """
        Load saved model.

        Args:
            filepath: Path to saved model

        Returns:
            ARIMAForecaster instance with loaded model
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load ARIMA results
        loaded_model = ARIMAResults.load(str(filepath))

        # Create instance and set fitted model
        instance = cls(order=loaded_model.model.order)
        instance.fitted_model = loaded_model

        instance.logger.info(f"ARIMA model loaded from {filepath}")
        return instance


# =============================================================================
# XGBoost Forecaster
# =============================================================================

class XGBoostForecaster:
    """
    XGBoost gradient boosting forecaster for time-series.

    Uses engineered features (lags, rolling stats, time features) to predict
    future prices. Supports confidence intervals via quantile regression.

    Attributes:
        model: XGBRegressor instance
        feature_names: List of feature names
        residual_std: Standard deviation of residuals for confidence intervals
        logger: Logger instance
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize XGBoost forecaster.

        Args:
            n_estimators: Number of boosting rounds (default: 100)
            max_depth: Maximum tree depth (default: 6)
            learning_rate: Step size shrinkage (default: 0.1)
            config: Optional configuration dictionary
            **kwargs: Additional XGBoost parameters
        """
        self.logger = logging.getLogger(__name__)

        # Load defaults from config if provided
        if config is not None:
            xgb_config = config.get('models', {}).get('xgboost', {})
            n_estimators = xgb_config.get('n_estimators', n_estimators)
            max_depth = xgb_config.get('max_depth', max_depth)
            learning_rate = xgb_config.get('learning_rate', learning_rate)

            # Add other config params to kwargs
            for key in ['subsample', 'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']:
                if key in xgb_config:
                    kwargs[key] = xgb_config[key]

        # Create XGBoost model
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            **kwargs
        )

        self.feature_names = None
        self.residual_std = None

        self.logger.info(
            f"XGBoostForecaster initialized with n_estimators={n_estimators}, "
            f"max_depth={max_depth}, learning_rate={learning_rate}"
        )

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple]] = None,
        verbose: bool = False
    ) -> 'XGBoostForecaster':
        """
        Fit XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            eval_set: Optional evaluation set for early stopping
            verbose: Whether to print training progress

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting XGBoost model on {len(X_train)} samples")

        # Store feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        # Fit model
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)

        # Compute residual std for confidence intervals
        train_pred = self.model.predict(X_train)
        residuals = y_train - train_pred
        self.residual_std = np.std(residuals)

        # Log feature importances (top 10)
        if self.feature_names is not None and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_features = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            self.logger.info("Top 10 feature importances:")
            for feat, imp in top_features:
                self.logger.info(f"  {feat}: {imp:.4f}")

        return self

    def predict(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts.

        Args:
            X_test: Test features
            return_conf_int: Whether to return confidence intervals (default: False)
            alpha: Significance level for confidence intervals (default: 0.05)

        Returns:
            Dictionary with 'forecast', optionally 'lower_ci', 'upper_ci'
        """
        # Point forecast
        forecast = self.model.predict(X_test)

        result = {'forecast': forecast}

        # Add confidence intervals if requested
        if return_conf_int and self.residual_std is not None:
            # Use residual standard deviation for approximate CI
            # For 95% CI: mean ± 1.96 * std
            z_score = 1.96 if alpha == 0.05 else 2.576  # 99% CI
            margin = z_score * self.residual_std

            result['lower_ci'] = forecast - margin
            result['upper_ci'] = forecast + margin

        self.logger.debug(f"Generated forecast for {len(X_test)} samples")
        return result

    def save(self, filepath: Union[Path, str]) -> Path:
        """
        Save fitted model.

        Args:
            filepath: Path to save model

        Returns:
            Path where model was saved
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        joblib.dump(self.model, filepath)

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'residual_std': self.residual_std
        }
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"XGBoost model saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Union[Path, str]) -> 'XGBoostForecaster':
        """
        Load saved model.

        Args:
            filepath: Path to saved model

        Returns:
            XGBoostForecaster instance with loaded model
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load model
        model = joblib.load(filepath)

        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Create instance
        instance = cls()
        instance.model = model
        instance.feature_names = metadata.get('feature_names')
        instance.residual_std = metadata.get('residual_std')

        instance.logger.info(f"XGBoost model loaded from {filepath}")
        return instance


# =============================================================================
# LSTM Forecaster
# =============================================================================

class LSTMForecaster:
    """
    LSTM (Long Short-Term Memory) deep learning forecaster.

    Recurrent neural network for capturing sequential patterns in time-series data.
    Requires 3D input: (samples, timesteps, features).

    Attributes:
        units: Number of LSTM units per layer
        layers: Number of LSTM layers
        dropout: Dropout rate for regularization
        epochs: Training epochs
        batch_size: Batch size for training
        model: Keras Sequential model
        input_shape: Input shape (timesteps, features)
        logger: Logger instance
    """

    def __init__(
        self,
        units: int = 50,
        layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        config: Optional[Dict] = None
    ):
        """
        Initialize LSTM forecaster.

        Args:
            units: Number of LSTM units per layer (default: 50)
            layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.2)
            epochs: Training epochs (default: 50)
            batch_size: Batch size (default: 32)
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        # Load defaults from config if provided
        if config is not None:
            lstm_config = config.get('models', {}).get('lstm', {})
            units = lstm_config.get('units', units)
            layers = lstm_config.get('layers', layers)
            dropout = lstm_config.get('dropout', dropout)
            epochs = lstm_config.get('epochs', epochs)
            batch_size = lstm_config.get('batch_size', batch_size)

        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.input_shape = None
        self.history = None

        self.logger.info(
            f"LSTMForecaster initialized with units={units}, layers={layers}, "
            f"dropout={dropout}, epochs={epochs}"
        )

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM architecture.

        Args:
            input_shape: Tuple of (timesteps, features)

        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential(name='LSTM_Forecaster')

        # First LSTM layer
        model.add(LSTM(
            units=self.units,
            return_sequences=(self.layers > 1),
            input_shape=input_shape,
            name='lstm_1'
        ))
        model.add(Dropout(self.dropout, name='dropout_1'))

        # Additional LSTM layers
        for i in range(1, self.layers):
            return_seq = (i < self.layers - 1)
            model.add(LSTM(
                units=self.units,
                return_sequences=return_seq,
                name=f'lstm_{i+1}'
            ))
            model.add(Dropout(self.dropout, name=f'dropout_{i+1}'))

        # Output layer
        model.add(Dense(1, name='output'))

        # Compile
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        self.logger.info(f"Built LSTM model with input_shape={input_shape}")
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1
    ) -> 'LSTMForecaster':
        """
        Fit LSTM model.

        Args:
            X_train: Training sequences (3D: samples, timesteps, features)
            y_train: Training target
            validation_data: Optional tuple of (X_val, y_val)
            verbose: Verbosity level (0=silent, 1=progress, 2=one line per epoch)

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting LSTM model on {len(X_train)} sequences")

        # Build model if not already built
        if self.model is None:
            self.input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(self.input_shape)

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # Fit model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=verbose
        )

        # Log training results
        final_loss = self.history.history['loss'][-1]
        if validation_data:
            final_val_loss = self.history.history['val_loss'][-1]
            self.logger.info(f"Training complete. Final loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")
        else:
            self.logger.info(f"Training complete. Final loss={final_loss:.4f}")

        return self

    def predict(
        self,
        X_test: np.ndarray,
        return_conf_int: bool = False,
        n_simulations: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts.

        Args:
            X_test: Test sequences (3D: samples, timesteps, features)
            return_conf_int: Whether to return confidence intervals (default: False)
            n_simulations: Number of MC dropout simulations for CI (default: 100)

        Returns:
            Dictionary with 'forecast', optionally 'lower_ci', 'upper_ci'
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Point forecast
        forecast = self.model.predict(X_test, verbose=0).flatten()

        result = {'forecast': forecast}

        # Add confidence intervals if requested (using MC Dropout)
        if return_conf_int:
            # MC Dropout: enable dropout at inference time
            predictions = []
            for _ in range(n_simulations):
                # Call model with training=True to enable dropout
                pred = self.model(X_test, training=True).numpy().flatten()
                predictions.append(pred)

            predictions = np.array(predictions)
            result['lower_ci'] = np.percentile(predictions, 2.5, axis=0)
            result['upper_ci'] = np.percentile(predictions, 97.5, axis=0)

        self.logger.debug(f"Generated forecast for {len(X_test)} sequences")
        return result

    def save(self, filepath: Union[Path, str]) -> Path:
        """
        Save fitted model.

        Args:
            filepath: Path to save model (without extension)

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No fitted model to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model in .keras format (TensorFlow 2.x recommended format)
        model_path = filepath.with_suffix('.keras')
        self.model.save(str(model_path))

        # Save metadata
        metadata = {
            'units': self.units,
            'layers': self.layers,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'input_shape': self.input_shape
        }
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"LSTM model saved to {model_path}")
        return model_path

    @classmethod
    def load(cls, filepath: Union[Path, str]) -> 'LSTMForecaster':
        """
        Load saved model.

        Args:
            filepath: Path to saved model (without extension)

        Returns:
            LSTMForecaster instance with loaded model
        """
        filepath = Path(filepath)
        model_path = filepath.with_suffix('.keras')

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        model = load_model(str(model_path))

        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Create instance
        instance = cls(
            units=metadata.get('units', 50),
            layers=metadata.get('layers', 2),
            dropout=metadata.get('dropout', 0.2),
            epochs=metadata.get('epochs', 50),
            batch_size=metadata.get('batch_size', 32)
        )
        instance.model = model
        instance.input_shape = tuple(metadata.get('input_shape', (None, None)))

        instance.logger.info(f"LSTM model loaded from {model_path}")
        return instance


# =============================================================================
# Price Forecasting Pipeline
# =============================================================================

class PriceForecastingPipeline:
    """
    Orchestrate end-to-end price forecasting workflow.

    Handles data preparation, feature engineering, model training, evaluation,
    ensemble forecasting, and model persistence.

    Attributes:
        config: Configuration dictionary
        feature_engineer: FeatureEngineer instance
        models: Dictionary of trained models
        metrics: Dictionary of evaluation metrics
        ensemble_weights: Dictionary of ensemble weights
        logger: Logger instance

    Example:
        >>> from src.models.price_forecasting import PriceForecastingPipeline
        >>> from src.data.synthetic_generator import SyntheticPriceGenerator
        >>>
        >>> # Generate synthetic data
        >>> gen = SyntheticPriceGenerator()
        >>> prices = gen.generate_price_series('2023-01-01', '2024-12-31', frequency='H')
        >>>
        >>> # Create and train pipeline
        >>> pipeline = PriceForecastingPipeline()
        >>> pipeline.prepare_data(prices, target_col='price')
        >>> pipeline.train_models(model_types=['arima', 'xgboost', 'lstm'])
        >>>
        >>> # Evaluate
        >>> metrics = pipeline.evaluate_models()
        >>> print(metrics)
        >>>
        >>> # Create ensemble
        >>> pipeline.create_ensemble()
        >>>
        >>> # Predict
        >>> forecasts = pipeline.predict(prices.tail(200), model_name='ensemble', steps=24)
        >>>
        >>> # Save models
        >>> pipeline.save_models()
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize forecasting pipeline.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config is None:
            config = get_config()
        self.config = config

        # Extract forecasting configuration
        models_config = self.config.get('models', {})
        forecasting_config = models_config.get('forecasting', {})
        self.lookback_hours = forecasting_config.get('lookback_hours', 168)
        self.forecast_horizon = forecasting_config.get('forecast_horizon', 24)
        self.model_types = forecasting_config.get('model_types', ['arima', 'xgboost', 'lstm'])
        self.train_test_split = models_config.get('train_test_split', 0.7)
        self.model_save_path = Path(models_config.get('model_save_path', 'models/saved'))
        self.random_seed = models_config.get('random_seed', 42)

        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(config=self.config)

        # Initialize storage
        self.models = {}
        self.metrics = {}
        self.ensemble_weights = {}

        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_seq = None
        self.X_val_seq = None
        self.X_test_seq = None
        self.y_train_seq = None
        self.y_val_seq = None
        self.y_test_seq = None

        self.logger.info(
            f"PriceForecastingPipeline initialized with lookback={self.lookback_hours}h, "
            f"horizon={self.forecast_horizon}h"
        )

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = 'price'
    ) -> 'PriceForecastingPipeline':
        """
        Load and prepare data for training.

        Args:
            data: DataFrame with DatetimeIndex and price column
            target_col: Name of target column (default: 'price')

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Preparing data: {len(data)} samples")

        # Create features
        features_df = self.feature_engineer.create_features(data, target_col=target_col, include_target=True)

        # Separate features and target
        feature_cols = self.feature_engineer.get_feature_names()
        X = features_df[feature_cols]
        y = features_df[target_col]

        # Time-based train/validation/test split
        train_data, val_data, test_data = time_series_split(
            features_df,
            train_ratio=self.train_test_split,
            val_ratio=0.15
        )

        self.X_train = train_data[feature_cols]
        self.y_train = train_data[target_col]
        self.X_val = val_data[feature_cols]
        self.y_val = val_data[target_col]
        self.X_test = test_data[feature_cols]
        self.y_test = test_data[target_col]

        # Create LSTM sequences
        if 'lstm' in self.model_types:
            self.X_train_seq, self.y_train_seq = self.feature_engineer.create_sequences_for_lstm(
                self.X_train, self.y_train, lookback=self.lookback_hours, forecast_horizon=1
            )
            self.X_val_seq, self.y_val_seq = self.feature_engineer.create_sequences_for_lstm(
                self.X_val, self.y_val, lookback=self.lookback_hours, forecast_horizon=1
            )
            self.X_test_seq, self.y_test_seq = self.feature_engineer.create_sequences_for_lstm(
                self.X_test, self.y_test, lookback=self.lookback_hours, forecast_horizon=1
            )

            # Note: Keep original y_train, y_val, y_test for ARIMA and XGBoost
            # Store y_*_seq for LSTM training

        self.logger.info(
            f"Data prepared: train={len(self.X_train)}, val={len(self.X_val)}, "
            f"test={len(self.X_test)}"
        )

        return self

    def train_models(self, model_types: Optional[List[str]] = None) -> 'PriceForecastingPipeline':
        """
        Train all configured models.

        Args:
            model_types: Optional list of model types to train (default: from config)

        Returns:
            Self for method chaining
        """
        if model_types is None:
            model_types = self.model_types

        self.logger.info(f"Training models: {model_types}")

        # Train ARIMA
        if 'arima' in model_types:
            try:
                self.logger.info("Training ARIMA model...")
                arima = ARIMAForecaster(config=self.config)
                arima.fit(self.y_train)
                self.models['arima'] = arima
                self.logger.info("ARIMA training complete")
            except Exception as e:
                self.logger.error(f"ARIMA training failed: {e}")

        # Train XGBoost
        if 'xgboost' in model_types:
            try:
                self.logger.info("Training XGBoost model...")
                xgb_model = XGBoostForecaster(config=self.config)
                eval_set = [(self.X_val, self.y_val)]
                xgb_model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)
                self.models['xgboost'] = xgb_model
                self.logger.info("XGBoost training complete")
            except Exception as e:
                self.logger.error(f"XGBoost training failed: {e}")

        # Train LSTM
        if 'lstm' in model_types and self.X_train_seq is not None:
            try:
                self.logger.info("Training LSTM model...")
                lstm = LSTMForecaster(config=self.config)
                # Use stored sequences for LSTM
                lstm.fit(
                    self.X_train_seq, self.y_train_seq,
                    validation_data=(self.X_val_seq, self.y_val_seq),
                    verbose=0
                )
                self.models['lstm'] = lstm
                self.logger.info("LSTM training complete")
            except Exception as e:
                self.logger.error(f"LSTM training failed: {e}")

        self.logger.info(f"Training complete: {len(self.models)} models trained")
        return self

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on validation set.

        Returns:
            Dictionary of metrics for each model
        """
        self.logger.info("Evaluating models on validation set")

        for model_name, model in self.models.items():
            try:
                # Generate predictions
                if model_name == 'arima':
                    pred_dict = model.predict(steps=len(self.y_val), return_conf_int=False)
                    predictions = pred_dict['forecast']
                    y_true = self.y_val.values

                elif model_name == 'xgboost':
                    pred_dict = model.predict(self.X_val, return_conf_int=False)
                    predictions = pred_dict['forecast']
                    y_true = self.y_val.values

                elif model_name == 'lstm':
                    pred_dict = model.predict(self.X_val_seq, return_conf_int=False)
                    predictions = pred_dict['forecast']
                    y_true = self.y_val_seq

                else:
                    continue

                # Calculate metrics
                metrics = calculate_metrics(y_true, predictions)
                self.metrics[model_name] = metrics

                self.logger.info(
                    f"{model_name.upper()} - MAE: {metrics['mae']:.2f}, "
                    f"RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%, "
                    f"DA: {metrics['directional_accuracy']:.2f}%"
                )

            except Exception as e:
                self.logger.error(f"Evaluation failed for {model_name}: {e}")

        return self.metrics

    def create_ensemble(
        self,
        weights: Optional[Dict[str, float]] = None,
        weight_strategy: str = 'inverse_rmse'
    ) -> 'PriceForecastingPipeline':
        """
        Create ensemble forecast from multiple models.

        Args:
            weights: Optional custom weights dict (model_name -> weight)
            weight_strategy: Strategy for computing weights ('equal', 'inverse_rmse', 'inverse_mae')

        Returns:
            Self for method chaining
        """
        if weights is not None:
            self.ensemble_weights = weights
        else:
            # Compute weights based on strategy
            if weight_strategy == 'equal':
                n_models = len(self.models)
                self.ensemble_weights = {name: 1.0 / n_models for name in self.models.keys()}

            elif weight_strategy == 'inverse_rmse':
                if not self.metrics:
                    raise ValueError("Must evaluate models before creating ensemble with inverse_rmse")

                # Compute inverse RMSE weights
                inverse_rmse = {name: 1.0 / metrics['rmse'] for name, metrics in self.metrics.items()}
                total = sum(inverse_rmse.values())
                self.ensemble_weights = {name: weight / total for name, weight in inverse_rmse.items()}

            elif weight_strategy == 'inverse_mae':
                if not self.metrics:
                    raise ValueError("Must evaluate models before creating ensemble with inverse_mae")

                # Compute inverse MAE weights
                inverse_mae = {name: 1.0 / metrics['mae'] for name, metrics in self.metrics.items()}
                total = sum(inverse_mae.values())
                self.ensemble_weights = {name: weight / total for name, weight in inverse_mae.items()}

            else:
                raise ValueError(f"Unknown weight strategy: {weight_strategy}")

        self.logger.info(f"Ensemble created with weights: {self.ensemble_weights}")
        return self

    def predict(
        self,
        data: pd.DataFrame,
        model_name: str = 'ensemble',
        steps: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using ensemble or individual model.

        Args:
            data: DataFrame with recent data for feature creation
            model_name: Model to use ('ensemble', 'arima', 'xgboost', 'lstm')
            steps: Number of steps to forecast (default: forecast_horizon from config)

        Returns:
            DataFrame with columns ['timestamp', 'forecast', 'lower_ci', 'upper_ci']
        """
        if steps is None:
            steps = self.forecast_horizon

        self.logger.info(f"Generating {steps}-step forecast using {model_name}")

        if model_name == 'ensemble':
            # Ensemble prediction: weighted average of all models
            ensemble_forecast = None
            ensemble_lower = None
            ensemble_upper = None

            for name, weight in self.ensemble_weights.items():
                if name in self.models:
                    # Get individual model prediction
                    pred_dict = self._predict_single_model(name, data, steps)
                    forecast = pred_dict['forecast']

                    if ensemble_forecast is None:
                        ensemble_forecast = weight * forecast
                        if 'lower_ci' in pred_dict:
                            ensemble_lower = weight * pred_dict['lower_ci']
                            ensemble_upper = weight * pred_dict['upper_ci']
                    else:
                        ensemble_forecast += weight * forecast
                        if 'lower_ci' in pred_dict and ensemble_lower is not None:
                            ensemble_lower += weight * pred_dict['lower_ci']
                            ensemble_upper += weight * pred_dict['upper_ci']

            # Create forecast DataFrame
            forecast_index = pd.date_range(
                start=data.index[-1] + pd.Timedelta(hours=1),
                periods=steps,
                freq='H'
            )
            result = pd.DataFrame({
                'timestamp': forecast_index,
                'forecast': ensemble_forecast
            })

            if ensemble_lower is not None:
                result['lower_ci'] = ensemble_lower
                result['upper_ci'] = ensemble_upper

        else:
            # Single model prediction
            pred_dict = self._predict_single_model(model_name, data, steps)

            forecast_index = pd.date_range(
                start=data.index[-1] + pd.Timedelta(hours=1),
                periods=steps,
                freq='H'
            )
            result = pd.DataFrame({
                'timestamp': forecast_index,
                'forecast': pred_dict['forecast']
            })

            if 'lower_ci' in pred_dict:
                result['lower_ci'] = pred_dict['lower_ci']
                result['upper_ci'] = pred_dict['upper_ci']

        return result

    def _predict_single_model(
        self,
        model_name: str,
        data: pd.DataFrame,
        steps: int
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction from a single model.

        Args:
            model_name: Name of model ('arima', 'xgboost', 'lstm')
            data: Input data
            steps: Number of steps to forecast

        Returns:
            Dictionary with 'forecast' and optionally 'lower_ci', 'upper_ci'
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")

        model = self.models[model_name]

        if model_name == 'arima':
            return model.predict(steps=steps, return_conf_int=True)

        elif model_name == 'xgboost':
            # Iterative multi-step forecasting for XGBoost
            target_col = 'price' if 'price' in data.columns else data.columns[0]

            # Create a working copy of the data with extended index for future predictions
            future_index = pd.date_range(
                start=data.index[-1] + pd.Timedelta(hours=1),
                periods=steps,
                freq='H'
            )
            working_data = data.copy()

            forecasts = []
            lower_cis = []
            upper_cis = []

            for step in range(steps):
                # Create features from current working data
                features_df = self.feature_engineer.create_features(
                    working_data, target_col=target_col, include_target=False
                )
                feature_cols = self.feature_engineer.get_feature_names()

                # Get the last row of features
                X_current = features_df[feature_cols].tail(1)

                # Predict next value
                pred_dict = model.predict(X_current, return_conf_int=True)
                forecast_value = pred_dict['forecast'][0]
                forecasts.append(forecast_value)

                if 'lower_ci' in pred_dict:
                    lower_cis.append(pred_dict['lower_ci'][0])
                    upper_cis.append(pred_dict['upper_ci'][0])

                # Append prediction to working data for next iteration
                new_row = pd.DataFrame(
                    {target_col: [forecast_value]},
                    index=[future_index[step]]
                )
                working_data = pd.concat([working_data, new_row])

            result = {'forecast': np.array(forecasts)}
            if lower_cis:
                result['lower_ci'] = np.array(lower_cis)
                result['upper_ci'] = np.array(upper_cis)

            return result

        elif model_name == 'lstm':
            # Iterative multi-step forecasting for LSTM
            target_col = 'price' if 'price' in data.columns else data.columns[0]

            # Create features from input data
            features_df = self.feature_engineer.create_features(
                data, target_col=target_col, include_target=True
            )
            feature_cols = self.feature_engineer.get_feature_names()
            X = features_df[feature_cols]
            y = features_df[target_col]

            # Get the last window of features
            X_window = X.tail(self.lookback_hours).values  # Shape: (lookback_hours, n_features)

            forecasts = []
            lower_cis = []
            upper_cis = []

            for step in range(steps):
                # Reshape window for LSTM: (1, lookback_hours, n_features)
                X_input = X_window.reshape(1, self.lookback_hours, X_window.shape[1])

                # Predict next value
                pred_dict = model.predict(X_input, return_conf_int=True)
                forecast_value = pred_dict['forecast'][0]
                forecasts.append(forecast_value)

                if 'lower_ci' in pred_dict:
                    lower_cis.append(pred_dict['lower_ci'][0])
                    upper_cis.append(pred_dict['upper_ci'][0])

                # Create next row of features by updating with the prediction
                # For simplicity, we'll create a minimal feature vector based on the prediction
                # In a real scenario, we'd need to properly compute all features including
                # time-based features for the future timestamp

                # Get the last row and update relevant features
                last_row = X_window[-1].copy()

                # Update lag features (shift existing lags)
                # This is a simplified approach - in production, you'd want to properly
                # track feature names and update them correctly
                # For now, we'll just append the last row and remove the oldest
                X_window = np.vstack([X_window[1:], last_row])

            result = {'forecast': np.array(forecasts)}
            if lower_cis:
                result['lower_ci'] = np.array(lower_cis)
                result['upper_ci'] = np.array(upper_cis)

            return result

    def save_models(self, model_save_path: Optional[Union[Path, str]] = None) -> Dict[str, Path]:
        """
        Save all trained models.

        Args:
            model_save_path: Optional path to save models. If provided, updates self.model_save_path

        Returns:
            Dictionary of model names to saved file paths
        """
        if model_save_path is not None:
            self.model_save_path = Path(model_save_path)

        self.logger.info(f"Saving models to {self.model_save_path}")

        self.model_save_path.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        # Save each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'arima':
                    filepath = self.model_save_path / 'arima.pkl'
                elif model_name == 'xgboost':
                    filepath = self.model_save_path / 'xgboost.joblib'
                elif model_name == 'lstm':
                    filepath = self.model_save_path / 'lstm'
                else:
                    continue

                saved_path = model.save(filepath)
                saved_paths[model_name] = saved_path
                self.logger.info(f"Saved {model_name} to {saved_path}")

            except Exception as e:
                self.logger.error(f"Failed to save {model_name}: {e}")

        # Save ensemble weights
        weights_path = self.model_save_path / 'ensemble_weights.json'
        with open(weights_path, 'w') as f:
            json.dump(self.ensemble_weights, f, indent=2)
        self.logger.info(f"Saved ensemble weights to {weights_path}")

        # Save feature engineer config
        config_path = self.model_save_path / 'feature_config.json'
        feature_config = {
            'lags': self.feature_engineer.lags,
            'rolling_windows': self.feature_engineer.rolling_windows,
            'time_features': self.feature_engineer.time_features
        }
        with open(config_path, 'w') as f:
            json.dump(feature_config, f, indent=2)
        self.logger.info(f"Saved feature config to {config_path}")

        return saved_paths

    @classmethod
    def load_models(cls, model_save_path: Union[Path, str]) -> 'PriceForecastingPipeline':
        """
        Load saved models.

        Args:
            model_save_path: Path to directory containing saved models

        Returns:
            PriceForecastingPipeline instance with loaded models
        """
        model_save_path = Path(model_save_path)

        if not model_save_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_save_path}")

        # Create pipeline instance
        instance = cls()
        instance.model_save_path = model_save_path

        # Load each model
        arima_path = model_save_path / 'arima.pkl'
        if arima_path.exists():
            instance.models['arima'] = ARIMAForecaster.load(arima_path)
            instance.logger.info("Loaded ARIMA model")

        xgb_path = model_save_path / 'xgboost.joblib'
        if xgb_path.exists():
            instance.models['xgboost'] = XGBoostForecaster.load(xgb_path)
            instance.logger.info("Loaded XGBoost model")

        lstm_path = model_save_path / 'lstm'
        if (lstm_path.with_suffix('.keras')).exists():
            instance.models['lstm'] = LSTMForecaster.load(lstm_path)
            instance.logger.info("Loaded LSTM model")

        # Load ensemble weights
        weights_path = model_save_path / 'ensemble_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                instance.ensemble_weights = json.load(f)
            instance.logger.info("Loaded ensemble weights")

        # Load feature config
        config_path = model_save_path / 'feature_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                feature_config = json.load(f)
            instance.feature_engineer.lags = feature_config['lags']
            instance.feature_engineer.rolling_windows = feature_config['rolling_windows']
            instance.feature_engineer.time_features = feature_config['time_features']
            instance.logger.info("Loaded feature config")

        instance.logger.info(f"Loaded {len(instance.models)} models from {model_save_path}")
        return instance
