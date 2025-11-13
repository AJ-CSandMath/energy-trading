"""
Renewable energy generation forecasting module with probabilistic forecasts.

This module provides specialized forecasting for wind and solar generation with
probabilistic forecasts, curtailment modeling, and Monte Carlo simulation for
portfolio planning. Includes ARIMA, XGBoost, LSTM, and quantile regression models
with uncertainty estimation.

Requirements:
    - xgboost>=2.0.0 for quantile regression support (QuantileXGBoostForecaster)
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

# Scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Statistical models
from statsmodels.tsa.arima.model import ARIMA

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
from src.data.data_manager import DataManager


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_generation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_mw: float
) -> Dict[str, float]:
    """
    Calculate evaluation metrics specific to generation forecasting.

    Args:
        y_true: True generation values
        y_pred: Predicted generation values
        capacity_mw: Generator capacity in MW

    Returns:
        Dictionary with MAE, RMSE, MAPE, capacity factor error, ramp rate error
    """
    # MAE: Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE: Mean Absolute Percentage Error
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    # Capacity factor error
    true_cf = np.mean(y_true) / capacity_mw
    pred_cf = np.mean(y_pred) / capacity_mw
    cf_error = np.abs(true_cf - pred_cf)

    # Ramp rate error (average absolute error in generation changes)
    if len(y_true) > 1:
        true_ramps = np.abs(np.diff(y_true))
        pred_ramps = np.abs(np.diff(y_pred))
        ramp_rate_error = np.mean(np.abs(true_ramps - pred_ramps))
    else:
        ramp_rate_error = 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'capacity_factor_error': cf_error,
        'ramp_rate_error': ramp_rate_error
    }


def apply_curtailment(
    generation_forecast: np.ndarray,
    price_forecast: Optional[np.ndarray] = None,
    grid_capacity: float = np.inf,
    negative_price_threshold: float = -5.0,
    max_curtailment_ramp: Optional[float] = None,
    dt_hours: float = 1.0,
    capacity_mw: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply curtailment based on grid constraints and pricing with ramp rate limits.

    Args:
        generation_forecast: Base generation forecast (MW)
        price_forecast: Optional price forecast ($/MWh)
        grid_capacity: Grid injection capacity limit (MW)
        negative_price_threshold: Curtail when price below this ($/MWh)
        max_curtailment_ramp: Maximum curtailment ramp rate (fraction of capacity per minute)
        dt_hours: Time step duration in hours (default: 1.0)
        capacity_mw: Generator capacity in MW (for ramp rate calculation)

    Returns:
        Tuple of (curtailed_generation, curtailment_amount)
    """
    curtailed_generation = generation_forecast.copy()
    curtailment = np.zeros_like(generation_forecast)

    # Apply grid capacity constraint
    grid_curtailment_mask = curtailed_generation > grid_capacity
    curtailment[grid_curtailment_mask] = curtailed_generation[grid_curtailment_mask] - grid_capacity
    curtailed_generation[grid_curtailment_mask] = grid_capacity

    # Apply negative pricing curtailment if price forecast provided
    if price_forecast is not None:
        negative_price_mask = price_forecast < negative_price_threshold
        additional_curtailment = curtailed_generation[negative_price_mask]
        curtailment[negative_price_mask] += additional_curtailment
        curtailed_generation[negative_price_mask] = 0

    # Apply ramp rate limits to curtailment if specified
    if max_curtailment_ramp is not None and len(curtailed_generation) > 1:
        # Use configured capacity if provided, otherwise estimate from forecast
        if capacity_mw is not None:
            capacity = capacity_mw
        else:
            capacity = np.max(generation_forecast) if np.max(generation_forecast) > 0 else 100.0
        max_ramp_mw = max_curtailment_ramp * 60 * dt_hours * capacity

        for t in range(1, len(curtailed_generation)):
            delta = curtailed_generation[t] - curtailed_generation[t-1]
            if abs(delta) > max_ramp_mw:
                if delta > 0:
                    curtailed_generation[t] = curtailed_generation[t-1] + max_ramp_mw
                else:
                    curtailed_generation[t] = curtailed_generation[t-1] - max_ramp_mw
                # Update curtailment amount
                curtailment[t] = generation_forecast[t] - curtailed_generation[t]

    return curtailed_generation, curtailment


# =============================================================================
# Renewable Feature Engineer
# =============================================================================

class RenewableFeatureEngineer:
    """
    Feature engineer specialized for renewable energy generation forecasting.

    Extends base feature engineering with weather-specific features, renewable
    generation patterns, and ramp rate analysis.

    Attributes:
        config: Configuration dictionary
        base_engineer: Base FeatureEngineer instance
        weather_lags: Weather feature lag periods
        weather_rolling_windows: Weather feature rolling windows
        logger: Logger instance
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize renewable feature engineer.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config is None:
            config = get_config()
        self.config = config

        # Initialize base feature engineer
        self.base_engineer = FeatureEngineer(config=config)

        # Extract renewable feature config
        renewable_config = self.config.get('models', {}).get('renewable_features', {})
        self.weather_lags = renewable_config.get('weather_lags', [1, 2, 3, 6, 12, 24])
        self.weather_rolling_windows = renewable_config.get('weather_rolling_windows', [6, 12, 24])
        self.capacity_factor_lags = renewable_config.get('capacity_factor_lags', [1, 6, 24, 168])
        self.include_ramp_features = renewable_config.get('include_ramp_features', True)
        self.include_seasonal = renewable_config.get('include_seasonal', True)
        self.ramp_window = renewable_config.get('ramp_window', 3)

        # Get capacity from wind/solar forecasting configs if available
        wind_config = self.config.get('models', {}).get('wind_forecasting', {})
        solar_config = self.config.get('models', {}).get('solar_forecasting', {})
        self.wind_capacity_mw = wind_config.get('capacity_mw', 100.0)
        self.solar_capacity_mw = solar_config.get('capacity_mw', 100.0)

        # Store feature names (populated during create_renewable_features)
        self._feature_names = []

        self.logger.info(
            f"RenewableFeatureEngineer initialized with {len(self.weather_lags)} weather lags, "
            f"{len(self.weather_rolling_windows)} weather windows"
        )

    def create_renewable_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'generation_mw',
        include_weather: bool = True
    ) -> pd.DataFrame:
        """
        Create features for renewable generation forecasting.

        Args:
            data: DataFrame with generation_mw and weather columns
            target_col: Name of target column (default: 'generation_mw')
            include_weather: Whether to include weather features

        Returns:
            DataFrame with all engineered features
        """
        self.logger.info(f"Creating renewable features for {len(data)} samples")

        # Start with base features (lags, rolling, time)
        features_df = self.base_engineer.create_features(
            data, target_col=target_col, include_target=True
        )

        # Identify weather columns
        weather_cols = []
        if include_weather:
            # Core weather features
            if 'wind_speed_mps' in data.columns:
                weather_cols.append('wind_speed_mps')
            if 'irradiance_w_m2' in data.columns:
                weather_cols.append('irradiance_w_m2')
            # Extended weather features
            if 'wind_direction_deg' in data.columns:
                weather_cols.append('wind_direction_deg')
            if 'temperature_c' in data.columns:
                weather_cols.append('temperature_c')
            if 'cloud_cover_pct' in data.columns:
                weather_cols.append('cloud_cover_pct')

        # Add weather features
        if weather_cols:
            weather_features = self.create_weather_features(data, weather_cols)
            # Merge weather features with base features
            for col in weather_features.columns:
                if col not in features_df.columns:
                    features_df[col] = weather_features[col]

        # Add capacity factor features if generation_mw and capacity_mw available
        if 'capacity_factor' in data.columns:
            cf_features = self._create_capacity_factor_features(data)
            for col in cf_features.columns:
                if col not in features_df.columns:
                    features_df[col] = cf_features[col]

        # Add ramp rate features
        if self.include_ramp_features:
            # Determine capacity based on data source (wind vs solar)
            capacity = None
            if 'wind_speed_mps' in data.columns:
                capacity = self.wind_capacity_mw
            elif 'irradiance_w_m2' in data.columns:
                capacity = self.solar_capacity_mw
            # Check if capacity_mw column exists in data
            elif 'capacity_mw' in data.columns:
                capacity = data['capacity_mw'].iloc[0] if len(data) > 0 else None

            ramp_features = self.create_ramp_features(data, target_col, capacity_mw=capacity)
            for col in ramp_features.columns:
                if col not in features_df.columns:
                    features_df[col] = ramp_features[col]

        # Add seasonal indicators
        if self.include_seasonal:
            seasonal_features = self._create_seasonal_indicators(features_df)
            for col in seasonal_features.columns:
                if col not in features_df.columns:
                    features_df[col] = seasonal_features[col]

        # Handle missing values
        features_df = features_df.dropna()

        # Store feature names (exclude target)
        self._feature_names = [col for col in features_df.columns if col != target_col]

        self.logger.info(
            f"Created {len(self._feature_names)} renewable features, "
            f"resulting in {len(features_df)} samples after dropping NaN"
        )

        return features_df

    def create_weather_features(
        self,
        data: pd.DataFrame,
        weather_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create weather-based features with extended coverage.

        Args:
            data: DataFrame with weather columns
            weather_cols: List of weather column names

        Returns:
            DataFrame with weather features
        """
        df = data.copy()

        for col in weather_cols:
            if col not in df.columns:
                continue

            # Special handling for wind direction (circular feature)
            if col == 'wind_direction_deg':
                # Trigonometric encoding for wind direction
                df['wind_direction_sin'] = np.sin(np.deg2rad(df[col]))
                df['wind_direction_cos'] = np.cos(np.deg2rad(df[col]))

                # Lag features for sin/cos components
                for lag in self.weather_lags:
                    df[f'wind_direction_sin_lag_{lag}'] = df['wind_direction_sin'].shift(lag)
                    df[f'wind_direction_cos_lag_{lag}'] = df['wind_direction_cos'].shift(lag)

                # Rolling statistics for sin/cos components
                for window in self.weather_rolling_windows:
                    df[f'wind_direction_sin_rolling_mean_{window}'] = df['wind_direction_sin'].rolling(window=window).mean()
                    df[f'wind_direction_cos_rolling_mean_{window}'] = df['wind_direction_cos'].rolling(window=window).mean()

                # Rate of change (use raw degrees for this)
                df[f'{col}_rate_of_change'] = df[col].diff()

                continue

            # Standard features for all other weather columns
            # Lag features
            for lag in self.weather_lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

            # Rolling statistics
            for window in self.weather_rolling_windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()

            # Rate of change
            df[f'{col}_rate_of_change'] = df[col].diff()

            # Interaction features
            if 'hour' in df.columns or hasattr(df.index, 'hour'):
                hour = df.index.hour if hasattr(df.index, 'hour') else df['hour']
                df[f'{col}_hour_interaction'] = df[col] * hour

            if 'month' in df.columns or hasattr(df.index, 'month'):
                month = df.index.month if hasattr(df.index, 'month') else df['month']
                df[f'{col}_month_interaction'] = df[col] * month

        self.logger.debug(f"Created weather features for columns: {weather_cols}")
        return df

    def create_ramp_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        capacity_mw: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Create ramp rate features for variability analysis.

        Args:
            data: DataFrame with generation data
            target_col: Target column name
            capacity_mw: Optional capacity for percentage calculation

        Returns:
            DataFrame with ramp features
        """
        df = data.copy()

        if target_col not in df.columns:
            return df

        # Basic ramp rate (MW/hour)
        df['ramp_rate'] = df[target_col].diff()

        # Percentage ramp rate - use capacity if available, otherwise previous generation
        if capacity_mw is not None and capacity_mw > 0:
            df['ramp_rate_pct'] = df['ramp_rate'] / capacity_mw
        else:
            # Fallback to previous generation-based percentage
            df['ramp_rate_pct'] = df['ramp_rate'] / (df[target_col].shift(1) + 1e-6)

        # Rolling ramp statistics - include ramp_window from config
        windows = [3, 6, 12]
        if self.ramp_window not in windows:
            windows.append(self.ramp_window)

        for window in windows:
            df[f'ramp_rate_rolling_max_{window}'] = df['ramp_rate'].rolling(window=window).max()
            df[f'ramp_rate_rolling_min_{window}'] = df['ramp_rate'].rolling(window=window).min()
            df[f'ramp_rate_volatility_{window}'] = df['ramp_rate'].rolling(window=window).std()

        self.logger.debug("Created ramp rate features")
        return df

    def _create_capacity_factor_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create capacity factor lag features."""
        df = data.copy()

        if 'capacity_factor' not in df.columns:
            return df

        for lag in self.capacity_factor_lags:
            df[f'capacity_factor_lag_{lag}'] = df['capacity_factor'].shift(lag)

        return df

    def _create_seasonal_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal indicator features."""
        df = pd.DataFrame(index=data.index)

        # High wind season (winter/spring: Nov-Apr)
        if hasattr(data.index, 'month'):
            month = data.index.month
            df['high_wind_season'] = ((month >= 11) | (month <= 4)).astype(int)
            df['high_solar_season'] = ((month >= 5) & (month <= 8)).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """
        Return list of all renewable feature column names.

        Returns:
            List of feature names including weather, ramp, and seasonal features
        """
        return self._feature_names if self._feature_names else self.base_engineer.get_feature_names()


# =============================================================================
# Quantile XGBoost Forecaster
# =============================================================================

class QuantileXGBoostForecaster:
    """
    XGBoost with quantile regression for probabilistic forecasts.

    Uses quantile loss objective to predict multiple quantiles simultaneously,
    enabling uncertainty quantification without MC dropout.

    Attributes:
        quantiles: List of quantiles to predict
        models: Dictionary mapping quantile to XGBRegressor
        feature_names: List of feature names
        logger: Logger instance
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize quantile XGBoost forecaster.

        Requires xgboost>=2.0.0 for native quantile regression support.

        Args:
            quantiles: Quantiles to predict (default: [0.1, 0.5, 0.9])
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            config: Optional configuration dictionary
            **kwargs: Additional XGBoost parameters

        Raises:
            ImportError: If xgboost version is too old
        """
        self.logger = logging.getLogger(__name__)

        # Check XGBoost version for quantile regression support
        xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
        if xgb_version < (2, 0):
            raise ImportError(
                f"QuantileXGBoostForecaster requires xgboost>=2.0.0 for native quantile "
                f"regression support. Current version: {xgb.__version__}. "
                f"Please upgrade with: pip install 'xgboost>=2.0.0'"
            )

        # Load defaults from config if provided
        if config is not None:
            qxgb_config = config.get('models', {}).get('quantile_xgboost', {})
            n_estimators = qxgb_config.get('n_estimators', n_estimators)
            max_depth = qxgb_config.get('max_depth', max_depth)
            learning_rate = qxgb_config.get('learning_rate', learning_rate)

            # Add other config params to kwargs
            for key in ['subsample', 'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']:
                if key in qxgb_config:
                    kwargs[key] = qxgb_config[key]

        self.quantiles = quantiles
        self.feature_names = None

        # Create separate model for each quantile
        self.models = {}
        for q in quantiles:
            self.models[q] = XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                **kwargs
            )

        self.logger.info(
            f"QuantileXGBoostForecaster initialized with quantiles={quantiles}, "
            f"n_estimators={n_estimators}, max_depth={max_depth}"
        )

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple]] = None,
        verbose: bool = False
    ) -> 'QuantileXGBoostForecaster':
        """
        Fit quantile models.

        Args:
            X_train: Training features
            y_train: Training target
            eval_set: Optional evaluation set
            verbose: Whether to print training progress

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting {len(self.quantiles)} quantile models on {len(X_train)} samples")

        # Store feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        # Fit each quantile model
        for q in self.quantiles:
            self.logger.debug(f"Training quantile {q} model...")
            self.models[q].fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
            self.logger.debug(f"Quantile {q} model training complete")

        self.logger.info("All quantile models trained successfully")
        return self

    def predict(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        return_quantiles: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic forecasts.

        Args:
            X_test: Test features
            return_quantiles: Whether to return all quantiles (default: True)

        Returns:
            Dictionary with 'forecast' (median), 'quantiles', 'lower_ci', 'upper_ci'
        """
        # Point forecast using median (0.5 quantile)
        if 0.5 in self.models:
            forecast = self.models[0.5].predict(X_test)
        else:
            # Use closest quantile to 0.5
            closest_q = min(self.quantiles, key=lambda x: abs(x - 0.5))
            forecast = self.models[closest_q].predict(X_test)

        result = {'forecast': forecast}

        if return_quantiles:
            quantile_predictions = {}
            for q in self.quantiles:
                quantile_predictions[q] = self.models[q].predict(X_test)

            result['quantiles'] = quantile_predictions

            # Add confidence intervals
            if 0.1 in quantile_predictions:
                result['lower_ci'] = quantile_predictions[0.1]
            if 0.9 in quantile_predictions:
                result['upper_ci'] = quantile_predictions[0.9]

        self.logger.debug(f"Generated quantile forecast for {len(X_test)} samples")
        return result

    def save(self, filepath: Union[Path, str]) -> Path:
        """
        Save all quantile models.

        Args:
            filepath: Base path to save models

        Returns:
            Path where models were saved
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save each quantile model
        for q in self.quantiles:
            q_filepath = filepath.parent / f"{filepath.stem}_q{q}{filepath.suffix}"
            joblib.dump(self.models[q], q_filepath)

        # Save metadata
        metadata = {
            'quantiles': self.quantiles,
            'feature_names': self.feature_names
        }
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Quantile XGBoost models saved to {filepath.parent}")
        return filepath

    @classmethod
    def load(cls, filepath: Union[Path, str]) -> 'QuantileXGBoostForecaster':
        """
        Load saved quantile models.

        Args:
            filepath: Base path to saved models

        Returns:
            QuantileXGBoostForecaster instance with loaded models
        """
        filepath = Path(filepath)

        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        quantiles = metadata['quantiles']

        # Create instance
        instance = cls(quantiles=quantiles)
        instance.feature_names = metadata.get('feature_names')

        # Load each quantile model
        for q in quantiles:
            q_filepath = filepath.parent / f"{filepath.stem}_q{q}{filepath.suffix}"
            if not q_filepath.exists():
                raise FileNotFoundError(f"Quantile model file not found: {q_filepath}")
            instance.models[q] = joblib.load(q_filepath)

        instance.logger.info(f"Quantile XGBoost models loaded from {filepath.parent}")
        return instance


# =============================================================================
# Wind Forecaster
# =============================================================================

class WindForecaster:
    """
    Specialized forecaster for wind generation.

    Supports multiple model types with physical constraints and curtailment modeling.

    Attributes:
        model_type: Type of model ('arima', 'xgboost', 'lstm', 'quantile_xgboost')
        model: Underlying forecasting model
        capacity_mw: Wind farm capacity
        config: Configuration dictionary
        logger: Logger instance
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        config: Optional[Dict] = None
    ):
        """
        Initialize wind forecaster.

        Args:
            model_type: Model type ('arima', 'xgboost', 'lstm', 'quantile_xgboost')
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        if config is None:
            config = get_config()
        self.config = config

        wind_config = self.config.get('models', {}).get('wind_forecasting', {})
        self.capacity_mw = wind_config.get('capacity_mw', 100.0)
        self.min_generation = wind_config.get('min_generation', 0.0)
        self.max_generation = wind_config.get('max_generation') or self.capacity_mw
        self.grid_capacity_mw = wind_config.get('grid_capacity_mw', 90.0)
        self.negative_price_threshold = wind_config.get('negative_price_threshold', -5.0)

        self.model_type = model_type
        self.model = None

        # Create model based on type
        if model_type == 'quantile_xgboost':
            quantiles = wind_config.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
            self.model = QuantileXGBoostForecaster(quantiles=quantiles, config=config)
        elif model_type == 'xgboost':
            from src.models.price_forecasting import XGBoostForecaster
            self.model = XGBoostForecaster(config=config)
        elif model_type == 'lstm':
            from src.models.price_forecasting import LSTMForecaster
            self.model = LSTMForecaster(config=config)
        elif model_type == 'arima':
            from src.models.price_forecasting import ARIMAForecaster
            self.model = ARIMAForecaster(config=config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.logger.info(f"WindForecaster initialized with model_type={model_type}, capacity={self.capacity_mw}MW")

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'WindForecaster':
        """
        Fit wind generation model.

        Args:
            X_train: Training features (3D for LSTM, 2D otherwise)
            y_train: Training target (generation_mw)
            X_val: Optional validation features (3D for LSTM, 2D otherwise)
            y_val: Optional validation target

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting wind forecaster ({self.model_type}) on {len(y_train)} samples")

        if self.model_type == 'arima':
            self.model.fit(y_train)
        elif self.model_type in ['xgboost', 'quantile_xgboost']:
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        elif self.model_type == 'lstm':
            # For LSTM, X_train should already be 3D sequences
            # If not, create sequences from 2D data
            if X_train.ndim == 2:
                lookback = self.config.get('models', {}).get('renewable_forecasting', {}).get('lookback_hours', 168)
                from src.models.feature_engineering import FeatureEngineer
                feature_eng = FeatureEngineer(config=self.config)

                X_train_seq, y_train_seq = feature_eng.create_sequences_for_lstm(
                    X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train),
                    y_train if isinstance(y_train, pd.Series) else pd.Series(y_train),
                    lookback=lookback,
                    forecast_horizon=1
                )

                if X_val is not None and y_val is not None:
                    X_val_seq, y_val_seq = feature_eng.create_sequences_for_lstm(
                        X_val if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val),
                        y_val if isinstance(y_val, pd.Series) else pd.Series(y_val),
                        lookback=lookback,
                        forecast_horizon=1
                    )
                    validation_data = (X_val_seq, y_val_seq)
                else:
                    validation_data = None

                self.model.fit(X_train_seq, y_train_seq, validation_data=validation_data, verbose=0)
            else:
                # Already 3D sequences
                validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
                self.model.fit(X_train, y_train, validation_data=validation_data, verbose=0)

        self.logger.info("Wind forecaster training complete")
        return self

    def predict(
        self,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        steps: Optional[int] = None,
        return_conf_int: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate wind generation forecasts.

        Args:
            X_test: Test features (not needed for ARIMA)
            steps: Number of steps to forecast (for ARIMA)
            return_conf_int: Whether to return confidence intervals

        Returns:
            Dictionary with 'forecast', 'lower_ci', 'upper_ci'
        """
        if self.model_type == 'arima':
            pred_dict = self.model.predict(steps=steps, return_conf_int=return_conf_int)
        elif self.model_type == 'quantile_xgboost':
            # Quantile forecaster uses return_quantiles parameter
            pred_dict = self.model.predict(X_test, return_quantiles=return_conf_int)
            # Extract lower_ci and upper_ci from quantiles if available
            if return_conf_int and 'quantiles' in pred_dict:
                quantiles = pred_dict['quantiles']
                if 0.1 in quantiles:
                    pred_dict['lower_ci'] = quantiles[0.1]
                if 0.9 in quantiles:
                    pred_dict['upper_ci'] = quantiles[0.9]
        else:
            pred_dict = self.model.predict(X_test, return_conf_int=return_conf_int)

        # Apply physical constraints
        forecast = pred_dict['forecast']
        forecast = np.clip(forecast, self.min_generation, self.max_generation)
        pred_dict['forecast'] = forecast

        if 'lower_ci' in pred_dict:
            pred_dict['lower_ci'] = np.clip(pred_dict['lower_ci'], self.min_generation, self.max_generation)
        if 'upper_ci' in pred_dict:
            pred_dict['upper_ci'] = np.clip(pred_dict['upper_ci'], self.min_generation, self.max_generation)

        self.logger.debug(f"Generated wind forecast for {len(forecast)} timesteps")
        return pred_dict

    def predict_with_curtailment(
        self,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        steps: Optional[int] = None,
        price_forecast: Optional[np.ndarray] = None,
        dt_hours: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Forecast with curtailment scenarios.

        Args:
            X_test: Test features
            steps: Number of steps to forecast
            price_forecast: Optional price forecast for economic curtailment
            dt_hours: Time step duration in hours (default: 1.0)

        Returns:
            Dictionary with 'forecast', 'curtailed_forecast', 'curtailment_mw'
        """
        # Generate base forecast
        pred_dict = self.predict(X_test, steps, return_conf_int=False)
        base_forecast = pred_dict['forecast']

        # Get curtailment ramp rate from config
        wind_config = self.config.get('models', {}).get('wind_forecasting', {})
        curtailment_ramp_rate = wind_config.get('curtailment_ramp_rate', 0.2)

        # Apply curtailment
        curtailed_forecast, curtailment_mw = apply_curtailment(
            generation_forecast=base_forecast,
            price_forecast=price_forecast,
            grid_capacity=self.grid_capacity_mw,
            negative_price_threshold=self.negative_price_threshold,
            max_curtailment_ramp=curtailment_ramp_rate,
            dt_hours=dt_hours,
            capacity_mw=self.capacity_mw
        )

        result = {
            'forecast': base_forecast,
            'curtailed_forecast': curtailed_forecast,
            'curtailment_mw': curtailment_mw
        }

        self.logger.debug(
            f"Applied curtailment: {curtailment_mw.sum():.2f} MWh curtailed "
            f"({curtailment_mw.sum()/base_forecast.sum()*100:.1f}% of generation)"
        )

        return result

    def analyze_ramp_rates(
        self,
        forecast: np.ndarray,
        dt_hours: float = 1.0
    ) -> Dict[str, float]:
        """
        Analyze forecast ramp rate characteristics.

        Args:
            forecast: Generation forecast array
            dt_hours: Time step duration in hours

        Returns:
            Dictionary with ramp statistics
        """
        ramp_rates = np.diff(forecast) / dt_hours
        abs_ramp_rates = np.abs(ramp_rates)

        return {
            'mean_ramp_rate': np.mean(abs_ramp_rates),
            'max_ramp_up': np.max(ramp_rates),
            'max_ramp_down': np.min(ramp_rates),
            'ramp_rate_std': np.std(ramp_rates),
            'ramp_rate_95th': np.percentile(abs_ramp_rates, 95)
        }

    def save(self, filepath: Union[Path, str]) -> Path:
        """Save fitted model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save underlying model
        model_path = self.model.save(filepath)

        # Save wind-specific metadata
        metadata = {
            'model_type': self.model_type,
            'capacity_mw': self.capacity_mw,
            'min_generation': self.min_generation,
            'max_generation': self.max_generation,
            'grid_capacity_mw': self.grid_capacity_mw,
            'negative_price_threshold': self.negative_price_threshold
        }
        metadata_path = filepath.with_name(f"{filepath.stem}_wind_meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Wind forecaster saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Union[Path, str], model_type: str) -> 'WindForecaster':
        """Load saved model."""
        filepath = Path(filepath)

        # Load metadata
        metadata_path = filepath.with_name(f"{filepath.stem}_wind_meta.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_type = metadata.get('model_type', model_type)

        # Create instance
        instance = cls(model_type=model_type)

        # Load underlying model
        if model_type == 'quantile_xgboost':
            instance.model = QuantileXGBoostForecaster.load(filepath)
        elif model_type == 'xgboost':
            from src.models.price_forecasting import XGBoostForecaster
            instance.model = XGBoostForecaster.load(filepath)
        elif model_type == 'lstm':
            from src.models.price_forecasting import LSTMForecaster
            instance.model = LSTMForecaster.load(filepath)
        elif model_type == 'arima':
            from src.models.price_forecasting import ARIMAForecaster
            instance.model = ARIMAForecaster.load(filepath)

        # Load wind-specific metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            instance.capacity_mw = metadata.get('capacity_mw', instance.capacity_mw)
            instance.min_generation = metadata.get('min_generation', instance.min_generation)
            instance.max_generation = metadata.get('max_generation', instance.max_generation)
            instance.grid_capacity_mw = metadata.get('grid_capacity_mw', instance.grid_capacity_mw)
            instance.negative_price_threshold = metadata.get('negative_price_threshold', instance.negative_price_threshold)

        instance.logger.info(f"Wind forecaster loaded from {filepath}")
        return instance


# =============================================================================
# Solar Forecaster
# =============================================================================

class SolarForecaster:
    """
    Specialized forecaster for solar generation.

    Similar to WindForecaster but with solar-specific constraints like
    nighttime filtering and irradiance-based physical limits.

    Attributes:
        model_type: Type of model
        model: Underlying forecasting model
        capacity_mw: Solar farm capacity
        config: Configuration dictionary
        logger: Logger instance
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        config: Optional[Dict] = None
    ):
        """
        Initialize solar forecaster.

        Args:
            model_type: Model type ('arima', 'xgboost', 'lstm', 'quantile_xgboost')
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        if config is None:
            config = get_config()
        self.config = config

        solar_config = self.config.get('models', {}).get('solar_forecasting', {})
        self.capacity_mw = solar_config.get('capacity_mw', 100.0)
        self.min_generation = solar_config.get('min_generation', 0.0)
        self.max_generation = solar_config.get('max_generation') or self.capacity_mw
        self.grid_capacity_mw = solar_config.get('grid_capacity_mw', 90.0)
        self.negative_price_threshold = solar_config.get('negative_price_threshold', -5.0)
        self.filter_nighttime = solar_config.get('filter_nighttime', True)
        self.min_irradiance_threshold = solar_config.get('min_irradiance_threshold', 10.0)

        self.model_type = model_type
        self.model = None

        # Create model based on type
        if model_type == 'quantile_xgboost':
            quantiles = solar_config.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
            self.model = QuantileXGBoostForecaster(quantiles=quantiles, config=config)
        elif model_type == 'xgboost':
            from src.models.price_forecasting import XGBoostForecaster
            self.model = XGBoostForecaster(config=config)
        elif model_type == 'lstm':
            from src.models.price_forecasting import LSTMForecaster
            self.model = LSTMForecaster(config=config)
        elif model_type == 'arima':
            from src.models.price_forecasting import ARIMAForecaster
            self.model = ARIMAForecaster(config=config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.logger.info(f"SolarForecaster initialized with model_type={model_type}, capacity={self.capacity_mw}MW")

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'SolarForecaster':
        """
        Fit solar generation model.

        Args:
            X_train: Training features
            y_train: Training target (generation_mw)
            X_val: Optional validation features
            y_val: Optional validation target

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting solar forecaster ({self.model_type}) on {len(y_train)} samples")

        # Filter nighttime data if requested - use irradiance if available
        if self.filter_nighttime:
            # Try to use irradiance_w_m2 column if available
            if isinstance(X_train, pd.DataFrame) and 'irradiance_w_m2' in X_train.columns:
                daylight_mask = X_train['irradiance_w_m2'].values > self.min_irradiance_threshold
                self.logger.info(f"Using irradiance-based nighttime filtering (threshold={self.min_irradiance_threshold} W/m²)")
            elif isinstance(y_train, (pd.Series, np.ndarray)):
                # Fallback to generation threshold
                if isinstance(y_train, pd.Series):
                    daylight_mask = y_train.values > 0.01
                else:
                    daylight_mask = y_train > 0.01
                self.logger.info("Using generation-based nighttime filtering (fallback)")
            else:
                daylight_mask = None

            if daylight_mask is not None and self.model_type in ['xgboost', 'quantile_xgboost']:
                X_train_filtered = X_train[daylight_mask] if isinstance(X_train, (pd.DataFrame, np.ndarray)) and len(X_train) == len(daylight_mask) else X_train
                y_train_filtered = y_train[daylight_mask] if isinstance(y_train, (pd.Series, np.ndarray)) and len(y_train) == len(daylight_mask) else y_train

                if X_val is not None and y_val is not None:
                    if isinstance(X_val, pd.DataFrame) and 'irradiance_w_m2' in X_val.columns:
                        val_daylight_mask = X_val['irradiance_w_m2'].values > self.min_irradiance_threshold
                    elif isinstance(y_val, pd.Series):
                        val_daylight_mask = y_val.values > 0.01
                    else:
                        val_daylight_mask = y_val > 0.01 if isinstance(y_val, np.ndarray) else None

                    if val_daylight_mask is not None:
                        X_val_filtered = X_val[val_daylight_mask] if len(X_val) == len(val_daylight_mask) else X_val
                        y_val_filtered = y_val[val_daylight_mask] if isinstance(y_val, (pd.Series, np.ndarray)) and len(y_val) == len(val_daylight_mask) else y_val
                    else:
                        X_val_filtered = X_val
                        y_val_filtered = y_val
                else:
                    X_val_filtered = X_val
                    y_val_filtered = y_val

                self.logger.info(f"Filtered to {len(y_train_filtered)} daylight samples")
            else:
                X_train_filtered = X_train
                y_train_filtered = y_train
                X_val_filtered = X_val
                y_val_filtered = y_val
        else:
            X_train_filtered = X_train
            y_train_filtered = y_train
            X_val_filtered = X_val
            y_val_filtered = y_val

        # Fit model
        if self.model_type == 'arima':
            self.model.fit(y_train_filtered)
        elif self.model_type in ['xgboost', 'quantile_xgboost']:
            eval_set = [(X_val_filtered, y_val_filtered)] if X_val_filtered is not None and y_val_filtered is not None else None
            self.model.fit(X_train_filtered, y_train_filtered, eval_set=eval_set, verbose=False)
        elif self.model_type == 'lstm':
            # For LSTM, X_train should already be 3D sequences
            # If not, create sequences from 2D data
            if isinstance(X_train_filtered, (pd.DataFrame, np.ndarray)) and X_train_filtered.ndim == 2:
                lookback = self.config.get('models', {}).get('renewable_forecasting', {}).get('lookback_hours', 168)
                from src.models.feature_engineering import FeatureEngineer
                feature_eng = FeatureEngineer(config=self.config)

                X_train_seq, y_train_seq = feature_eng.create_sequences_for_lstm(
                    X_train_filtered if isinstance(X_train_filtered, pd.DataFrame) else pd.DataFrame(X_train_filtered),
                    y_train_filtered if isinstance(y_train_filtered, pd.Series) else pd.Series(y_train_filtered),
                    lookback=lookback,
                    forecast_horizon=1
                )

                if X_val_filtered is not None and y_val_filtered is not None:
                    X_val_seq, y_val_seq = feature_eng.create_sequences_for_lstm(
                        X_val_filtered if isinstance(X_val_filtered, pd.DataFrame) else pd.DataFrame(X_val_filtered),
                        y_val_filtered if isinstance(y_val_filtered, pd.Series) else pd.Series(y_val_filtered),
                        lookback=lookback,
                        forecast_horizon=1
                    )
                    validation_data = (X_val_seq, y_val_seq)
                else:
                    validation_data = None

                self.model.fit(X_train_seq, y_train_seq, validation_data=validation_data, verbose=0)
            else:
                # Already 3D sequences
                validation_data = (X_val_filtered, y_val_filtered) if X_val_filtered is not None and y_val_filtered is not None else None
                self.model.fit(X_train_filtered, y_train_filtered, validation_data=validation_data, verbose=0)

        self.logger.info("Solar forecaster training complete")
        return self

    def predict(
        self,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        steps: Optional[int] = None,
        return_conf_int: bool = True,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate solar generation forecasts.

        Args:
            X_test: Test features
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            timestamps: Optional timestamps for nighttime detection

        Returns:
            Dictionary with 'forecast', 'lower_ci', 'upper_ci'
        """
        if self.model_type == 'arima':
            pred_dict = self.model.predict(steps=steps, return_conf_int=return_conf_int)
        elif self.model_type == 'quantile_xgboost':
            # Quantile forecaster uses return_quantiles parameter
            pred_dict = self.model.predict(X_test, return_quantiles=return_conf_int)
            # Extract lower_ci and upper_ci from quantiles if available
            if return_conf_int and 'quantiles' in pred_dict:
                quantiles = pred_dict['quantiles']
                if 0.1 in quantiles:
                    pred_dict['lower_ci'] = quantiles[0.1]
                if 0.9 in quantiles:
                    pred_dict['upper_ci'] = quantiles[0.9]
        else:
            pred_dict = self.model.predict(X_test, return_conf_int=return_conf_int)

        # Apply physical constraints
        forecast = pred_dict['forecast']

        # Zero generation at night - prioritize timestamps when provided for future forecasts
        if timestamps is not None and hasattr(timestamps, 'hour'):
            # Use timestamp-based nighttime detection (for future forecasts)
            night_mask = (timestamps.hour < 6) | (timestamps.hour >= 20)
            forecast[night_mask] = 0
            self.logger.debug(f"Applied timestamp-based nighttime zeroing: {night_mask.sum()} timesteps")
        elif isinstance(X_test, pd.DataFrame) and 'irradiance_w_m2' in X_test.columns:
            # Fallback to irradiance-based nighttime detection (for aligned observed times)
            night_mask = X_test['irradiance_w_m2'].values < self.min_irradiance_threshold
            forecast[night_mask] = 0
            self.logger.debug(f"Applied irradiance-based nighttime zeroing: {night_mask.sum()} timesteps")

        forecast = np.clip(forecast, self.min_generation, self.max_generation)
        pred_dict['forecast'] = forecast

        if 'lower_ci' in pred_dict:
            pred_dict['lower_ci'] = np.clip(pred_dict['lower_ci'], self.min_generation, self.max_generation)
        if 'upper_ci' in pred_dict:
            pred_dict['upper_ci'] = np.clip(pred_dict['upper_ci'], self.min_generation, self.max_generation)

        self.logger.debug(f"Generated solar forecast for {len(forecast)} timesteps")
        return pred_dict

    def predict_with_curtailment(
        self,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        steps: Optional[int] = None,
        price_forecast: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
        dt_hours: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Forecast with curtailment scenarios.

        Args:
            X_test: Test features
            steps: Number of steps to forecast
            price_forecast: Optional price forecast
            timestamps: Optional timestamps
            dt_hours: Time step duration in hours (default: 1.0)

        Returns:
            Dictionary with 'forecast', 'curtailed_forecast', 'curtailment_mw'
        """
        # Generate base forecast
        pred_dict = self.predict(X_test, steps, return_conf_int=False, timestamps=timestamps)
        base_forecast = pred_dict['forecast']

        # Get curtailment ramp rate from config
        solar_config = self.config.get('models', {}).get('solar_forecasting', {})
        curtailment_ramp_rate = solar_config.get('curtailment_ramp_rate', 0.3)

        # Apply curtailment
        curtailed_forecast, curtailment_mw = apply_curtailment(
            generation_forecast=base_forecast,
            price_forecast=price_forecast,
            grid_capacity=self.grid_capacity_mw,
            negative_price_threshold=self.negative_price_threshold,
            max_curtailment_ramp=curtailment_ramp_rate,
            dt_hours=dt_hours,
            capacity_mw=self.capacity_mw
        )

        result = {
            'forecast': base_forecast,
            'curtailed_forecast': curtailed_forecast,
            'curtailment_mw': curtailment_mw
        }

        self.logger.debug(
            f"Applied curtailment: {curtailment_mw.sum():.2f} MWh curtailed "
            f"({curtailment_mw.sum()/(base_forecast.sum()+1e-6)*100:.1f}% of generation)"
        )

        return result

    def analyze_capacity_factor(
        self,
        forecast: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze forecast capacity factor.

        Args:
            forecast: Generation forecast array

        Returns:
            Dictionary with CF statistics
        """
        capacity_factors = forecast / self.capacity_mw

        return {
            'mean_cf': np.mean(capacity_factors),
            'median_cf': np.median(capacity_factors),
            'max_cf': np.max(capacity_factors),
            'cf_std': np.std(capacity_factors),
            'cf_95th': np.percentile(capacity_factors, 95)
        }

    def save(self, filepath: Union[Path, str]) -> Path:
        """Save fitted model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save underlying model
        model_path = self.model.save(filepath)

        # Save solar-specific metadata
        metadata = {
            'model_type': self.model_type,
            'capacity_mw': self.capacity_mw,
            'min_generation': self.min_generation,
            'max_generation': self.max_generation,
            'grid_capacity_mw': self.grid_capacity_mw,
            'negative_price_threshold': self.negative_price_threshold,
            'filter_nighttime': self.filter_nighttime,
            'min_irradiance_threshold': self.min_irradiance_threshold
        }
        metadata_path = filepath.with_name(f"{filepath.stem}_solar_meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Solar forecaster saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Union[Path, str], model_type: str) -> 'SolarForecaster':
        """Load saved model."""
        filepath = Path(filepath)

        # Load metadata
        metadata_path = filepath.with_name(f"{filepath.stem}_solar_meta.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_type = metadata.get('model_type', model_type)

        # Create instance
        instance = cls(model_type=model_type)

        # Load underlying model
        if model_type == 'quantile_xgboost':
            instance.model = QuantileXGBoostForecaster.load(filepath)
        elif model_type == 'xgboost':
            from src.models.price_forecasting import XGBoostForecaster
            instance.model = XGBoostForecaster.load(filepath)
        elif model_type == 'lstm':
            from src.models.price_forecasting import LSTMForecaster
            instance.model = LSTMForecaster.load(filepath)
        elif model_type == 'arima':
            from src.models.price_forecasting import ARIMAForecaster
            instance.model = ARIMAForecaster.load(filepath)

        # Load solar-specific metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            instance.capacity_mw = metadata.get('capacity_mw', instance.capacity_mw)
            instance.min_generation = metadata.get('min_generation', instance.min_generation)
            instance.max_generation = metadata.get('max_generation', instance.max_generation)
            instance.grid_capacity_mw = metadata.get('grid_capacity_mw', instance.grid_capacity_mw)
            instance.negative_price_threshold = metadata.get('negative_price_threshold', instance.negative_price_threshold)
            instance.filter_nighttime = metadata.get('filter_nighttime', instance.filter_nighttime)
            instance.min_irradiance_threshold = metadata.get('min_irradiance_threshold', instance.min_irradiance_threshold)

        instance.logger.info(f"Solar forecaster loaded from {filepath}")
        return instance


# =============================================================================
# Monte Carlo Simulator
# =============================================================================

class MonteCarloSimulator:
    """
    Generate correlated scenarios for portfolio planning.

    Uses multivariate normal distribution to create correlated wind/solar
    scenarios with specified correlation structure.

    Attributes:
        n_scenarios: Number of scenarios to generate
        correlation_matrix: Correlation matrix for scenarios
        random_seed: Random seed for reproducibility
        logger: Logger instance
    """

    def __init__(
        self,
        n_scenarios: int = 1000,
        correlation_matrix: Optional[np.ndarray] = None,
        random_seed: int = 45,
        config: Optional[Dict] = None
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_scenarios: Number of scenarios (default: 1000)
            correlation_matrix: Optional custom correlation matrix
            random_seed: Random seed (default: 45)
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        if config is None:
            config = get_config()
        self.config = config

        mc_config = self.config.get('models', {}).get('monte_carlo', {})
        self.n_scenarios = mc_config.get('n_scenarios', n_scenarios)
        self.random_seed = mc_config.get('random_seed', random_seed)
        self.wind_solar_correlation = mc_config.get('wind_solar_correlation', -0.2)
        self.wind_std_multiplier = mc_config.get('wind_std_multiplier', 0.15)
        self.solar_std_multiplier = mc_config.get('solar_std_multiplier', 0.20)
        self.stress_scenarios = mc_config.get('stress_scenarios', {})

        self.correlation_matrix = correlation_matrix
        self.rng = np.random.default_rng(self.random_seed)

        self.logger.info(
            f"MonteCarloSimulator initialized: n_scenarios={self.n_scenarios}, "
            f"correlation={self.wind_solar_correlation}"
        )

    def generate_correlated_scenarios(
        self,
        base_wind_forecast: np.ndarray,
        base_solar_forecast: np.ndarray,
        wind_std: Optional[float] = None,
        solar_std: Optional[float] = None,
        correlation: Optional[float] = None,
        wind_capacity_mw: Optional[float] = None,
        solar_capacity_mw: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate correlated wind/solar scenarios.

        Args:
            base_wind_forecast: Base wind forecast (MW)
            base_solar_forecast: Base solar forecast (MW)
            wind_std: Optional wind standard deviation (default: from config)
            solar_std: Optional solar standard deviation (default: from config)
            correlation: Optional correlation coefficient (default: from config)
            wind_capacity_mw: Optional wind capacity cap (MW)
            solar_capacity_mw: Optional solar capacity cap (MW)

        Returns:
            Dictionary with 'wind_scenarios', 'solar_scenarios' (n_scenarios x n_timesteps)
        """
        n_timesteps = len(base_wind_forecast)

        if wind_std is None:
            wind_std = self.wind_std_multiplier * np.mean(base_wind_forecast)
        if solar_std is None:
            solar_std = self.solar_std_multiplier * np.mean(base_solar_forecast)
        if correlation is None:
            correlation = self.wind_solar_correlation

        self.logger.info(
            f"Generating {self.n_scenarios} correlated scenarios for {n_timesteps} timesteps"
        )

        # Covariance matrix - use provided correlation_matrix if available
        if self.correlation_matrix is not None:
            # Validate correlation matrix dimensions
            if self.correlation_matrix.shape != (2, 2):
                self.logger.warning(
                    f"Invalid correlation_matrix shape {self.correlation_matrix.shape}, expected (2, 2). "
                    f"Falling back to scalar correlation construction."
                )
                cov_matrix = np.array([
                    [wind_std**2, correlation * wind_std * solar_std],
                    [correlation * wind_std * solar_std, solar_std**2]
                ])
            else:
                # Convert correlation matrix to covariance matrix using provided stds
                # cov[i,j] = corr[i,j] * std[i] * std[j]
                stds = np.array([wind_std, solar_std])
                cov_matrix = self.correlation_matrix * np.outer(stds, stds)
                self.logger.info("Using provided correlation_matrix for scenario generation")
        else:
            # Construct covariance matrix from scalar correlation
            cov_matrix = np.array([
                [wind_std**2, correlation * wind_std * solar_std],
                [correlation * wind_std * solar_std, solar_std**2]
            ])

        # Generate scenarios
        wind_scenarios = np.zeros((self.n_scenarios, n_timesteps))
        solar_scenarios = np.zeros((self.n_scenarios, n_timesteps))

        for i in range(self.n_scenarios):
            # Generate correlated noise for all timesteps
            noise = self.rng.multivariate_normal(
                mean=[0, 0],
                cov=cov_matrix,
                size=n_timesteps
            )

            # Add noise to base forecasts
            wind_scenarios[i] = base_wind_forecast + noise[:, 0]
            solar_scenarios[i] = base_solar_forecast + noise[:, 1]

            # Apply physical constraints (non-negative and capacity caps)
            wind_scenarios[i] = np.maximum(wind_scenarios[i], 0)
            solar_scenarios[i] = np.maximum(solar_scenarios[i], 0)

            if wind_capacity_mw is not None:
                wind_scenarios[i] = np.minimum(wind_scenarios[i], wind_capacity_mw)
            if solar_capacity_mw is not None:
                solar_scenarios[i] = np.minimum(solar_scenarios[i], solar_capacity_mw)

        self.logger.info("Scenario generation complete")

        return {
            'wind_scenarios': wind_scenarios,
            'solar_scenarios': solar_scenarios
        }

    def generate_stress_scenarios(
        self,
        base_forecasts: Dict[str, np.ndarray],
        scenario_types: List[str] = ['low_wind', 'low_solar', 'renewable_drought', 'high_variability']
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate specific stress test scenarios.

        Args:
            base_forecasts: Dictionary with 'wind' and 'solar' base forecasts
            scenario_types: List of scenario types to generate

        Returns:
            Dictionary mapping scenario_name to forecasts
        """
        self.logger.info(f"Generating stress scenarios: {scenario_types}")

        stress_results = {}

        for scenario_type in scenario_types:
            if scenario_type not in self.stress_scenarios:
                self.logger.warning(f"Unknown stress scenario type: {scenario_type}")
                continue

            scenario_config = self.stress_scenarios[scenario_type]

            if scenario_type == 'low_wind':
                reduction = scenario_config.get('reduction_factor', 0.4)
                wind_stressed = base_forecasts['wind'] * reduction
                solar_stressed = base_forecasts['solar'].copy()

            elif scenario_type == 'low_solar':
                reduction = scenario_config.get('reduction_factor', 0.5)
                wind_stressed = base_forecasts['wind'].copy()
                solar_stressed = base_forecasts['solar'] * reduction

            elif scenario_type == 'renewable_drought':
                wind_reduction = scenario_config.get('wind_reduction', 0.5)
                solar_reduction = scenario_config.get('solar_reduction', 0.6)
                wind_stressed = base_forecasts['wind'] * wind_reduction
                solar_stressed = base_forecasts['solar'] * solar_reduction

            elif scenario_type == 'high_variability':
                wind_mult = scenario_config.get('wind_std_multiplier', 0.30)
                solar_mult = scenario_config.get('solar_std_multiplier', 0.40)

                wind_std = wind_mult * np.mean(base_forecasts['wind'])
                solar_std = solar_mult * np.mean(base_forecasts['solar'])

                wind_noise = self.rng.normal(0, wind_std, len(base_forecasts['wind']))
                solar_noise = self.rng.normal(0, solar_std, len(base_forecasts['solar']))

                wind_stressed = base_forecasts['wind'] + wind_noise
                solar_stressed = base_forecasts['solar'] + solar_noise

                wind_stressed = np.maximum(wind_stressed, 0)
                solar_stressed = np.maximum(solar_stressed, 0)

            else:
                continue

            stress_results[scenario_type] = {
                'wind': wind_stressed,
                'solar': solar_stressed
            }

        self.logger.info(f"Generated {len(stress_results)} stress scenarios")
        return stress_results

    def calculate_portfolio_metrics(
        self,
        wind_scenarios: np.ndarray,
        solar_scenarios: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level metrics across scenarios.

        Args:
            wind_scenarios: Wind scenarios (n_scenarios x n_timesteps)
            solar_scenarios: Solar scenarios (n_scenarios x n_timesteps)
            weights: Optional weights dict {'wind': w1, 'solar': w2}

        Returns:
            Dictionary with portfolio statistics
        """
        if weights is None:
            weights = {'wind': 1.0, 'solar': 1.0}

        # Calculate total portfolio generation for each scenario
        portfolio_scenarios = (
            weights['wind'] * wind_scenarios +
            weights['solar'] * solar_scenarios
        )

        # Calculate statistics
        mean_generation = np.mean(portfolio_scenarios)
        median_generation = np.median(portfolio_scenarios)
        std_generation = np.std(portfolio_scenarios)

        # Percentiles
        p10 = np.percentile(portfolio_scenarios, 10)
        p90 = np.percentile(portfolio_scenarios, 90)

        # Coefficient of variation
        cv = std_generation / mean_generation if mean_generation > 0 else 0

        return {
            'mean_generation': mean_generation,
            'median_generation': median_generation,
            'std_generation': std_generation,
            'p10_generation': p10,
            'p90_generation': p90,
            'coefficient_of_variation': cv
        }

    def export_scenarios(
        self,
        scenarios: Dict[str, np.ndarray],
        output_path: Union[Path, str],
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Path:
        """
        Export scenarios for downstream use (optimized without Python loops).

        Args:
            scenarios: Dictionary with 'wind_scenarios', 'solar_scenarios'
            output_path: Path to save scenarios
            timestamps: Optional timestamps for scenario index

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        wind_scenarios = scenarios['wind_scenarios']
        solar_scenarios = scenarios['solar_scenarios']

        n_scenarios, n_timesteps = wind_scenarios.shape

        # Create DataFrame using vectorized operations (no Python loops)
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=n_timesteps, freq='H')

        # Create MultiIndex using repeat and tile
        scenario_ids = np.repeat(np.arange(n_scenarios), n_timesteps)
        timestamp_array = np.tile(timestamps, n_scenarios)

        # Flatten scenario arrays
        wind_flat = wind_scenarios.flatten()
        solar_flat = solar_scenarios.flatten()

        # Create DataFrame directly from arrays
        df = pd.DataFrame({
            'scenario_id': scenario_ids,
            'timestamp': timestamp_array,
            'wind_generation_mw': wind_flat,
            'solar_generation_mw': solar_flat
        })

        # Save as Parquet
        df.to_parquet(output_path, compression='snappy', index=False)

        self.logger.info(f"Exported {n_scenarios} scenarios to {output_path}")
        return output_path


# =============================================================================
# Renewable Forecasting Pipeline
# =============================================================================

class RenewableForecastingPipeline:
    """
    Orchestrate renewable forecasting workflow.

    Handles data preparation, feature engineering, model training, evaluation,
    ensemble forecasting, curtailment modeling, and scenario generation.

    Attributes:
        config: Configuration dictionary
        feature_engineer: RenewableFeatureEngineer instance
        wind_models: Dictionary of trained wind models
        solar_models: Dictionary of trained solar models
        metrics: Dictionary of evaluation metrics
        ensemble_weights: Dictionary of ensemble weights
        mc_simulator: MonteCarloSimulator instance
        logger: Logger instance

    Example:
        >>> from src.models.renewable_forecasting import RenewableForecastingPipeline
        >>> from src.data.renewable_generator import WindGenerator, SolarGenerator
        >>>
        >>> # Generate synthetic data
        >>> wind_gen = WindGenerator()
        >>> wind_data = wind_gen.generate_wind_profile('2023-01-01', '2024-12-31')
        >>> solar_gen = SolarGenerator()
        >>> solar_data = solar_gen.generate_solar_profile('2023-01-01', '2024-12-31')
        >>>
        >>> # Create and train pipeline
        >>> pipeline = RenewableForecastingPipeline()
        >>> pipeline.prepare_data(wind_data, solar_data)
        >>> pipeline.train_wind_models(['xgboost', 'lstm'])
        >>> pipeline.train_solar_models(['xgboost', 'lstm'])
        >>>
        >>> # Evaluate and create ensemble
        >>> metrics = pipeline.evaluate_models()
        >>> pipeline.create_ensemble()
        >>>
        >>> # Generate forecasts with curtailment
        >>> forecasts = pipeline.forecast(
        >>>     wind_data.tail(200), solar_data.tail(200),
        >>>     model_name='ensemble', steps=24, include_curtailment=True
        >>> )
        >>>
        >>> # Generate Monte Carlo scenarios
        >>> scenarios = pipeline.generate_scenarios(
        >>>     wind_data.tail(200), solar_data.tail(200),
        >>>     n_scenarios=1000, include_stress_tests=True
        >>> )
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

        # Extract configuration
        models_config = self.config.get('models', {})
        renewable_config = models_config.get('renewable_forecasting', {})
        self.lookback_hours = renewable_config.get('lookback_hours', 168)
        self.forecast_horizon = renewable_config.get('forecast_horizon', 24)
        self.model_types = renewable_config.get('model_types', ['xgboost', 'lstm'])
        self.model_save_path = Path(models_config.get('model_save_path', 'models/saved'))
        self.random_seed = models_config.get('random_seed', 42)

        # Set random seeds
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        # Initialize feature engineer
        self.feature_engineer = RenewableFeatureEngineer(config=self.config)

        # Initialize storage
        self.wind_models = {}
        self.solar_models = {}
        self.metrics = {}
        self.ensemble_weights = {}

        # Initialize Monte Carlo simulator
        self.mc_simulator = MonteCarloSimulator(config=self.config)

        # Data storage
        self.wind_X_train = None
        self.wind_X_val = None
        self.wind_X_test = None
        self.wind_y_train = None
        self.wind_y_val = None
        self.wind_y_test = None

        self.solar_X_train = None
        self.solar_X_val = None
        self.solar_X_test = None
        self.solar_y_train = None
        self.solar_y_val = None
        self.solar_y_test = None

        self.logger.info(
            f"RenewableForecastingPipeline initialized: lookback={self.lookback_hours}h, "
            f"horizon={self.forecast_horizon}h"
        )

    def prepare_data(
        self,
        wind_data: pd.DataFrame,
        solar_data: pd.DataFrame
    ) -> 'RenewableForecastingPipeline':
        """
        Load and prepare wind/solar data.

        Args:
            wind_data: DataFrame with wind generation data
            solar_data: DataFrame with solar generation data

        Returns:
            Self for method chaining
        """
        self.logger.info(
            f"Preparing data: wind={len(wind_data)} samples, solar={len(solar_data)} samples"
        )

        # Prepare wind data
        wind_features = self.feature_engineer.create_renewable_features(
            wind_data, target_col='generation_mw', include_weather=True
        )

        # Time-based split for wind
        from src.models.price_forecasting import time_series_split
        wind_train, wind_val, wind_test = time_series_split(wind_features, train_ratio=0.7, val_ratio=0.15)

        feature_cols = [col for col in wind_features.columns if col != 'generation_mw']
        self.wind_X_train = wind_train[feature_cols]
        self.wind_y_train = wind_train['generation_mw']
        self.wind_X_val = wind_val[feature_cols]
        self.wind_y_val = wind_val['generation_mw']
        self.wind_X_test = wind_test[feature_cols]
        self.wind_y_test = wind_test['generation_mw']

        # Prepare solar data
        solar_features = self.feature_engineer.create_renewable_features(
            solar_data, target_col='generation_mw', include_weather=True
        )

        # Time-based split for solar
        solar_train, solar_val, solar_test = time_series_split(solar_features, train_ratio=0.7, val_ratio=0.15)

        feature_cols = [col for col in solar_features.columns if col != 'generation_mw']
        self.solar_X_train = solar_train[feature_cols]
        self.solar_y_train = solar_train['generation_mw']
        self.solar_X_val = solar_val[feature_cols]
        self.solar_y_val = solar_val['generation_mw']
        self.solar_X_test = solar_test[feature_cols]
        self.solar_y_test = solar_test['generation_mw']

        self.logger.info(
            f"Data prepared: wind_train={len(self.wind_X_train)}, solar_train={len(self.solar_X_train)}"
        )

        return self

    def train_wind_models(
        self,
        model_types: Optional[List[str]] = None
    ) -> 'RenewableForecastingPipeline':
        """
        Train wind forecasting models.

        Args:
            model_types: Optional list of model types to train

        Returns:
            Self for method chaining
        """
        if model_types is None:
            model_types = self.model_types

        self.logger.info(f"Training wind models: {model_types}")

        for model_type in model_types:
            try:
                self.logger.info(f"Training wind {model_type} model...")
                wind_forecaster = WindForecaster(model_type=model_type, config=self.config)
                wind_forecaster.fit(
                    self.wind_X_train, self.wind_y_train,
                    self.wind_X_val, self.wind_y_val
                )
                self.wind_models[model_type] = wind_forecaster
                self.logger.info(f"Wind {model_type} training complete")
            except Exception as e:
                self.logger.error(f"Wind {model_type} training failed: {e}")

        self.logger.info(f"Wind model training complete: {len(self.wind_models)} models trained")
        return self

    def train_solar_models(
        self,
        model_types: Optional[List[str]] = None
    ) -> 'RenewableForecastingPipeline':
        """
        Train solar forecasting models.

        Args:
            model_types: Optional list of model types to train

        Returns:
            Self for method chaining
        """
        if model_types is None:
            model_types = self.model_types

        self.logger.info(f"Training solar models: {model_types}")

        for model_type in model_types:
            try:
                self.logger.info(f"Training solar {model_type} model...")
                solar_forecaster = SolarForecaster(model_type=model_type, config=self.config)
                solar_forecaster.fit(
                    self.solar_X_train, self.solar_y_train,
                    self.solar_X_val, self.solar_y_val
                )
                self.solar_models[model_type] = solar_forecaster
                self.logger.info(f"Solar {model_type} training complete")
            except Exception as e:
                self.logger.error(f"Solar {model_type} training failed: {e}")

        self.logger.info(f"Solar model training complete: {len(self.solar_models)} models trained")
        return self

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on validation set.

        Returns:
            Dictionary of metrics for each model
        """
        self.logger.info("Evaluating models on validation set")

        # Evaluate wind models
        for model_name, model in self.wind_models.items():
            try:
                pred_dict = model.predict(self.wind_X_val, return_conf_int=False)
                predictions = pred_dict['forecast']
                y_true = self.wind_y_val.values

                metrics = calculate_generation_metrics(
                    y_true, predictions,
                    capacity_mw=model.capacity_mw
                )
                self.metrics[f'wind_{model_name}'] = metrics

                self.logger.info(
                    f"WIND {model_name.upper()} - MAE: {metrics['mae']:.2f}, "
                    f"RMSE: {metrics['rmse']:.2f}, CF_error: {metrics['capacity_factor_error']:.4f}"
                )
            except Exception as e:
                self.logger.error(f"Evaluation failed for wind {model_name}: {e}")

        # Evaluate solar models
        for model_name, model in self.solar_models.items():
            try:
                pred_dict = model.predict(self.solar_X_val, return_conf_int=False)
                predictions = pred_dict['forecast']
                y_true = self.solar_y_val.values

                metrics = calculate_generation_metrics(
                    y_true, predictions,
                    capacity_mw=model.capacity_mw
                )
                self.metrics[f'solar_{model_name}'] = metrics

                self.logger.info(
                    f"SOLAR {model_name.upper()} - MAE: {metrics['mae']:.2f}, "
                    f"RMSE: {metrics['rmse']:.2f}, CF_error: {metrics['capacity_factor_error']:.4f}"
                )
            except Exception as e:
                self.logger.error(f"Evaluation failed for solar {model_name}: {e}")

        return self.metrics

    def create_ensemble(
        self,
        weight_strategy: str = 'inverse_rmse'
    ) -> 'RenewableForecastingPipeline':
        """
        Create ensemble forecasts.

        Args:
            weight_strategy: Strategy for computing weights ('equal', 'inverse_rmse')

        Returns:
            Self for method chaining
        """
        if not self.metrics:
            raise ValueError("Must evaluate models before creating ensemble")

        # Compute weights separately for wind and solar
        wind_metrics = {k.replace('wind_', ''): v for k, v in self.metrics.items() if k.startswith('wind_')}
        solar_metrics = {k.replace('solar_', ''): v for k, v in self.metrics.items() if k.startswith('solar_')}

        if weight_strategy == 'inverse_rmse':
            # Wind ensemble weights
            wind_inverse_rmse = {name: 1.0 / metrics['rmse'] for name, metrics in wind_metrics.items()}
            wind_total = sum(wind_inverse_rmse.values())
            self.ensemble_weights['wind'] = {name: weight / wind_total for name, weight in wind_inverse_rmse.items()}

            # Solar ensemble weights
            solar_inverse_rmse = {name: 1.0 / metrics['rmse'] for name, metrics in solar_metrics.items()}
            solar_total = sum(solar_inverse_rmse.values())
            self.ensemble_weights['solar'] = {name: weight / solar_total for name, weight in solar_inverse_rmse.items()}

        elif weight_strategy == 'equal':
            self.ensemble_weights['wind'] = {name: 1.0 / len(wind_metrics) for name in wind_metrics.keys()}
            self.ensemble_weights['solar'] = {name: 1.0 / len(solar_metrics) for name in solar_metrics.keys()}

        self.logger.info(f"Ensemble created - Wind weights: {self.ensemble_weights['wind']}")
        self.logger.info(f"Ensemble created - Solar weights: {self.ensemble_weights['solar']}")

        return self

    def forecast(
        self,
        wind_data: pd.DataFrame,
        solar_data: pd.DataFrame,
        model_name: str = 'ensemble',
        steps: Optional[int] = None,
        include_curtailment: bool = False,
        price_forecast: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for new data using iterative multi-step forecasting.

        Args:
            wind_data: Wind generation data
            solar_data: Solar generation data
            model_name: Model to use ('ensemble' or specific model)
            steps: Number of steps to forecast
            include_curtailment: Whether to apply curtailment
            price_forecast: Optional price forecast for curtailment

        Returns:
            Dictionary with 'wind_forecast', 'solar_forecast' DataFrames
        """
        if steps is None:
            steps = self.forecast_horizon

        self.logger.info(f"Generating {steps}-step forecast using {model_name}")

        # Generate future timestamps for both wind and solar
        forecast_index = pd.date_range(
            start=wind_data.index[-1] + pd.Timedelta(hours=1),
            periods=steps,
            freq='H'
        )

        # Iteratively generate multi-step forecasts
        wind_forecasts = []
        solar_forecasts = []

        # Working copies of data to build upon iteratively
        working_wind_data = wind_data.copy()
        working_solar_data = solar_data.copy()

        for step in range(steps):
            # Create features from current working data
            wind_features = self.feature_engineer.create_renewable_features(
                working_wind_data, target_col='generation_mw', include_weather=True
            )
            solar_features = self.feature_engineer.create_renewable_features(
                working_solar_data, target_col='generation_mw', include_weather=True
            )

            feature_cols_wind = [col for col in wind_features.columns if col != 'generation_mw']
            feature_cols_solar = [col for col in solar_features.columns if col != 'generation_mw']

            # Get last row of features for this step
            X_wind = wind_features[feature_cols_wind].tail(1)
            X_solar = solar_features[feature_cols_solar].tail(1)

            # Get current timestamp for this forecast step
            current_timestamp = forecast_index[step]

            # Generate single-step wind forecast
            if model_name == 'ensemble' and 'wind' in self.ensemble_weights:
                wind_pred = self._ensemble_predict('wind', X_wind, include_curtailment,
                                                   price_forecast[step:step+1] if price_forecast is not None else None)
            else:
                if model_name in self.wind_models:
                    if include_curtailment:
                        wind_pred = self.wind_models[model_name].predict_with_curtailment(
                            X_wind, price_forecast=price_forecast[step:step+1] if price_forecast is not None else None
                        )
                    else:
                        wind_pred = self.wind_models[model_name].predict(X_wind)
                else:
                    raise ValueError(f"Wind model '{model_name}' not found")

            # Generate single-step solar forecast - pass single timestamp for nighttime handling
            if model_name == 'ensemble' and 'solar' in self.ensemble_weights:
                solar_pred = self._ensemble_predict('solar', X_solar, include_curtailment,
                                                    price_forecast[step:step+1] if price_forecast is not None else None,
                                                    timestamps=pd.DatetimeIndex([current_timestamp]))
            else:
                if model_name in self.solar_models:
                    if include_curtailment:
                        solar_pred = self.solar_models[model_name].predict_with_curtailment(
                            X_solar, price_forecast=price_forecast[step:step+1] if price_forecast is not None else None,
                            timestamps=pd.DatetimeIndex([current_timestamp])
                        )
                    else:
                        solar_pred = self.solar_models[model_name].predict(X_solar, timestamps=pd.DatetimeIndex([current_timestamp]))
                else:
                    raise ValueError(f"Solar model '{model_name}' not found")

            # Extract forecast values
            wind_value = wind_pred.get('forecast', wind_pred.get('curtailed_forecast'))[0]
            solar_value = solar_pred.get('forecast', solar_pred.get('curtailed_forecast'))[0]

            wind_forecasts.append(wind_value)
            solar_forecasts.append(solar_value)

            # Append predictions to working data for next iteration
            # This updates lags and enables proper feature generation for next step
            new_wind_row = pd.DataFrame(
                {'generation_mw': [wind_value]},
                index=[current_timestamp]
            )
            # Preserve weather columns if they exist in the original data
            for col in ['wind_speed_mps', 'capacity_factor']:
                if col in working_wind_data.columns:
                    # Use last known value (in real deployment, would use weather forecast)
                    new_wind_row[col] = working_wind_data[col].iloc[-1]
            working_wind_data = pd.concat([working_wind_data, new_wind_row])

            new_solar_row = pd.DataFrame(
                {'generation_mw': [solar_value]},
                index=[current_timestamp]
            )
            # Preserve weather columns if they exist
            for col in ['irradiance_w_m2', 'capacity_factor']:
                if col in working_solar_data.columns:
                    # Use last known value (in real deployment, would use weather forecast)
                    new_solar_row[col] = working_solar_data[col].iloc[-1]
            working_solar_data = pd.concat([working_solar_data, new_solar_row])

        # Create result DataFrames
        wind_df = pd.DataFrame({
            'forecast': wind_forecasts,
        }, index=forecast_index)

        solar_df = pd.DataFrame({
            'forecast': solar_forecasts,
        }, index=forecast_index)

        return {
            'wind_forecast': wind_df,
            'solar_forecast': solar_df
        }

    def _ensemble_predict(
        self,
        resource_type: str,
        X_test: pd.DataFrame,
        include_curtailment: bool,
        price_forecast: Optional[np.ndarray],
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, np.ndarray]:
        """Generate ensemble prediction for wind or solar."""
        models = self.wind_models if resource_type == 'wind' else self.solar_models
        weights = self.ensemble_weights[resource_type]

        ensemble_forecast = None

        for model_name, weight in weights.items():
            if model_name not in models:
                continue

            if include_curtailment:
                # Pass timestamps for solar forecaster
                if resource_type == 'solar':
                    pred_dict = models[model_name].predict_with_curtailment(
                        X_test, price_forecast=price_forecast, timestamps=timestamps
                    )
                else:
                    pred_dict = models[model_name].predict_with_curtailment(X_test, price_forecast=price_forecast)
                forecast = pred_dict.get('curtailed_forecast', pred_dict['forecast'])
            else:
                # Pass timestamps for solar forecaster
                if resource_type == 'solar':
                    pred_dict = models[model_name].predict(X_test, return_conf_int=False, timestamps=timestamps)
                else:
                    pred_dict = models[model_name].predict(X_test, return_conf_int=False)
                forecast = pred_dict['forecast']

            if ensemble_forecast is None:
                ensemble_forecast = weight * forecast
            else:
                ensemble_forecast += weight * forecast

        return {'forecast': ensemble_forecast}

    def generate_scenarios(
        self,
        base_wind_data: pd.DataFrame,
        base_solar_data: pd.DataFrame,
        n_scenarios: Optional[int] = None,
        include_stress_tests: bool = False,
        model_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate Monte Carlo scenarios.

        Args:
            base_wind_data: Base wind data
            base_solar_data: Base solar data
            n_scenarios: Number of scenarios (default: from config)
            include_stress_tests: Whether to include stress test scenarios
            model_name: Optional model name (default: 'ensemble' if available)

        Returns:
            Dictionary with scenario arrays

        Raises:
            ValueError: If ensemble requested but not available
        """
        if n_scenarios is None:
            n_scenarios = self.mc_simulator.n_scenarios

        # Determine which model to use for base forecasts
        if model_name is None:
            # Check if ensemble weights exist
            if 'wind' in self.ensemble_weights and 'solar' in self.ensemble_weights:
                model_name = 'ensemble'
                self.logger.info("Using ensemble for scenario base forecasts")
            elif self.metrics:
                # Fall back to best single model based on lowest RMSE
                wind_metrics = {k.replace('wind_', ''): v for k, v in self.metrics.items() if k.startswith('wind_')}
                solar_metrics = {k.replace('solar_', ''): v for k, v in self.metrics.items() if k.startswith('solar_')}

                if wind_metrics and solar_metrics:
                    best_wind_model = min(wind_metrics.items(), key=lambda x: x[1]['rmse'])[0]
                    best_solar_model = min(solar_metrics.items(), key=lambda x: x[1]['rmse'])[0]
                    if best_wind_model == best_solar_model:
                        model_name = best_wind_model
                        self.logger.info(f"Using best single model '{model_name}' for scenario base forecasts")
                    else:
                        self.logger.warning(
                            f"No ensemble available. Wind best: {best_wind_model}, Solar best: {best_solar_model}. "
                            f"Please run evaluate_models() and create_ensemble() first. Using {best_wind_model}."
                        )
                        model_name = best_wind_model
                else:
                    raise ValueError(
                        "No models available for scenario generation. "
                        "Please run train_wind_models(), train_solar_models(), "
                        "evaluate_models(), and create_ensemble() first."
                    )
            else:
                raise ValueError(
                    "No ensemble weights or metrics available for scenario generation. "
                    "Please run evaluate_models() and create_ensemble() first."
                )

        self.logger.info(f"Generating {n_scenarios} Monte Carlo scenarios using {model_name}")

        # Generate base forecasts
        forecasts = self.forecast(base_wind_data, base_solar_data, model_name=model_name)
        base_wind_forecast = forecasts['wind_forecast']['forecast'].values
        base_solar_forecast = forecasts['solar_forecast']['forecast'].values

        # Get capacity values for capping scenarios
        wind_capacity = None
        solar_capacity = None
        if self.wind_models:
            first_wind_model = next(iter(self.wind_models.values()))
            wind_capacity = first_wind_model.capacity_mw
        if self.solar_models:
            first_solar_model = next(iter(self.solar_models.values()))
            solar_capacity = first_solar_model.capacity_mw

        # Generate correlated scenarios with capacity caps
        scenarios = self.mc_simulator.generate_correlated_scenarios(
            base_wind_forecast,
            base_solar_forecast,
            wind_capacity_mw=wind_capacity,
            solar_capacity_mw=solar_capacity
        )

        # Add stress test scenarios if requested
        if include_stress_tests:
            stress_scenarios = self.mc_simulator.generate_stress_scenarios({
                'wind': base_wind_forecast,
                'solar': base_solar_forecast
            })
            scenarios['stress_scenarios'] = stress_scenarios

        self.logger.info("Scenario generation complete")
        return scenarios

    def analyze_variability(
        self,
        forecasts: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze forecast variability and uncertainty.

        Args:
            forecasts: Dictionary with forecast DataFrames

        Returns:
            Dictionary with variability metrics
        """
        variability_metrics = {}

        for resource_type, forecast_df in forecasts.items():
            forecast = forecast_df['forecast'].values

            # Calculate spread and uncertainty
            forecast_std = np.std(forecast)
            forecast_mean = np.mean(forecast)
            cv = forecast_std / forecast_mean if forecast_mean > 0 else 0

            # Ramp rates
            ramp_rates = np.abs(np.diff(forecast))
            mean_ramp = np.mean(ramp_rates)
            max_ramp = np.max(ramp_rates)

            variability_metrics[resource_type] = {
                'forecast_std': forecast_std,
                'forecast_mean': forecast_mean,
                'coefficient_of_variation': cv,
                'mean_ramp_rate': mean_ramp,
                'max_ramp_rate': max_ramp
            }

        return variability_metrics

    def save_models(self) -> Dict[str, Path]:
        """
        Save all trained models.

        Returns:
            Dictionary of model names to saved file paths
        """
        self.logger.info(f"Saving models to {self.model_save_path}")

        self.model_save_path.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        # Save wind models
        for model_name, model in self.wind_models.items():
            try:
                filepath = self.model_save_path / f'wind_{model_name}.joblib'
                saved_path = model.save(filepath)
                saved_paths[f'wind_{model_name}'] = saved_path
                self.logger.info(f"Saved wind {model_name} to {saved_path}")
            except Exception as e:
                self.logger.error(f"Failed to save wind {model_name}: {e}")

        # Save solar models
        for model_name, model in self.solar_models.items():
            try:
                filepath = self.model_save_path / f'solar_{model_name}.joblib'
                saved_path = model.save(filepath)
                saved_paths[f'solar_{model_name}'] = saved_path
                self.logger.info(f"Saved solar {model_name} to {saved_path}")
            except Exception as e:
                self.logger.error(f"Failed to save solar {model_name}: {e}")

        # Save ensemble weights
        weights_path = self.model_save_path / 'renewable_ensemble_weights.json'
        with open(weights_path, 'w') as f:
            json.dump(self.ensemble_weights, f, indent=2)
        self.logger.info(f"Saved ensemble weights to {weights_path}")

        # Save feature engineer config
        config_path = self.model_save_path / 'renewable_feature_config.json'
        feature_config = {
            'weather_lags': self.feature_engineer.weather_lags,
            'weather_rolling_windows': self.feature_engineer.weather_rolling_windows,
            'capacity_factor_lags': self.feature_engineer.capacity_factor_lags,
            'include_ramp_features': self.feature_engineer.include_ramp_features,
            'include_seasonal': self.feature_engineer.include_seasonal
        }
        with open(config_path, 'w') as f:
            json.dump(feature_config, f, indent=2)
        self.logger.info(f"Saved feature config to {config_path}")

        return saved_paths

    @classmethod
    def load_models(cls, model_save_path: Union[Path, str]) -> 'RenewableForecastingPipeline':
        """
        Load saved models.

        Args:
            model_save_path: Path to directory containing saved models

        Returns:
            RenewableForecastingPipeline instance with loaded models
        """
        model_save_path = Path(model_save_path)

        if not model_save_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_save_path}")

        # Create pipeline instance
        instance = cls()
        instance.model_save_path = model_save_path

        # Load wind models
        for model_type in ['xgboost', 'lstm', 'arima', 'quantile_xgboost']:
            wind_path = model_save_path / f'wind_{model_type}.joblib'
            if wind_path.exists() or (model_save_path / f'wind_{model_type}.keras').exists():
                instance.wind_models[model_type] = WindForecaster.load(wind_path, model_type=model_type)
                instance.logger.info(f"Loaded wind {model_type} model")

        # Load solar models
        for model_type in ['xgboost', 'lstm', 'arima', 'quantile_xgboost']:
            solar_path = model_save_path / f'solar_{model_type}.joblib'
            if solar_path.exists() or (model_save_path / f'solar_{model_type}.keras').exists():
                instance.solar_models[model_type] = SolarForecaster.load(solar_path, model_type=model_type)
                instance.logger.info(f"Loaded solar {model_type} model")

        # Load ensemble weights
        weights_path = model_save_path / 'renewable_ensemble_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                instance.ensemble_weights = json.load(f)
            instance.logger.info("Loaded ensemble weights")

        # Load feature config
        config_path = model_save_path / 'renewable_feature_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                feature_config = json.load(f)
            instance.feature_engineer.weather_lags = feature_config['weather_lags']
            instance.feature_engineer.weather_rolling_windows = feature_config['weather_rolling_windows']
            instance.feature_engineer.capacity_factor_lags = feature_config['capacity_factor_lags']
            instance.feature_engineer.include_ramp_features = feature_config['include_ramp_features']
            instance.feature_engineer.include_seasonal = feature_config['include_seasonal']
            instance.logger.info("Loaded feature config")

        instance.logger.info(f"Loaded models from {model_save_path}")
        return instance
