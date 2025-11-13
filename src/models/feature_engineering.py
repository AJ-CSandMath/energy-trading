"""
Feature engineering module for time-series price forecasting.

This module provides the FeatureEngineer class for transforming raw time-series
price data into ML-ready features including lag features, rolling statistics,
time-based features, and LSTM sequences.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.config.load_config import get_config


class FeatureEngineer:
    """
    Transform time-series price data into machine learning features.

    Creates lag features, rolling window statistics, time-based features, and
    sequences for LSTM models. Follows configuration from config.yaml.

    Attributes:
        config (Dict): Configuration dictionary
        lags (List[int]): Lag periods to create
        rolling_windows (List[int]): Window sizes for rolling statistics
        time_features (List[str]): Time-based features to create
        logger (logging.Logger): Logger instance

    Example:
        >>> from src.models.feature_engineering import FeatureEngineer
        >>> from src.data.data_manager import DataManager
        >>>
        >>> # Load price data
        >>> manager = DataManager()
        >>> prices = manager.load_data('synthetic', 'prices', 'processed')
        >>>
        >>> # Create features
        >>> engineer = FeatureEngineer()
        >>> features_df = engineer.create_features(prices, target_col='price')
        >>>
        >>> # Get feature names
        >>> feature_cols = engineer.get_feature_names()
        >>> X = features_df[feature_cols]
        >>> y = features_df['price']
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the FeatureEngineer.

        Args:
            config: Optional configuration dictionary. If not provided, loads from config.yaml
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config is None:
            config = get_config()
        self.config = config

        # Extract feature engineering parameters
        features_config = self.config.get('models', {}).get('features', {})
        self.lags = features_config.get('lags', [1, 2, 3, 6, 12, 24, 48, 168])
        self.rolling_windows = features_config.get('rolling_windows', [24, 48, 168])
        self.time_features = features_config.get('time_features',
                                                  ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday'])

        # Store feature names (populated during create_features)
        self._feature_names = []

        self.logger.info(
            f"FeatureEngineer initialized with {len(self.lags)} lags, "
            f"{len(self.rolling_windows)} rolling windows, {len(self.time_features)} time features"
        )

    def create_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'price',
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Create all features from raw price data.

        Args:
            data: DataFrame with DatetimeIndex and price column
            target_col: Name of the target column (default: 'price')
            include_target: Whether to include target column in output (default: True)

        Returns:
            DataFrame with all engineered features and optionally target column

        Raises:
            ValueError: If data validation fails
        """
        self.logger.info(f"Creating features for {len(data)} samples")

        # Validate input data
        self._validate_data(data, target_col)

        # Create a copy to avoid modifying original
        features_df = data.copy()

        # Create lag features
        features_df = self.create_lag_features(features_df, target_col, self.lags)

        # Create rolling features
        features_df = self.create_rolling_features(features_df, target_col, self.rolling_windows)

        # Create time features
        features_df = self.create_time_features(features_df, self.time_features)

        # Handle missing values (drop rows with NaN)
        features_df = self._handle_missing_values(features_df)

        # Store feature names (exclude target)
        if include_target:
            self._feature_names = [col for col in features_df.columns if col != target_col]
        else:
            self._feature_names = list(features_df.columns)

        self.logger.info(
            f"Created {len(self._feature_names)} features, "
            f"resulting in {len(features_df)} samples after dropping NaN"
        )

        # Optionally exclude target
        if not include_target and target_col in features_df.columns:
            features_df = features_df.drop(columns=[target_col])

        return features_df

    def create_lag_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged price features.

        Args:
            data: DataFrame with price data
            target_col: Name of the target column
            lags: List of lag periods to create

        Returns:
            DataFrame with additional lag columns named like 'price_lag_1', 'price_lag_24', etc.
        """
        df = data.copy()

        for lag in lags:
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df[target_col].shift(lag)

        self.logger.debug(f"Created {len(lags)} lag features")
        return df

    def create_rolling_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.

        Args:
            data: DataFrame with price data
            target_col: Name of the target column
            windows: List of window sizes for rolling statistics

        Returns:
            DataFrame with rolling features named like 'price_rolling_mean_24',
            'price_rolling_std_24', etc.
        """
        df = data.copy()

        for window in windows:
            # Rolling mean (average price)
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()

            # Rolling standard deviation (volatility)
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()

            # Rolling minimum (support level)
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()

            # Rolling maximum (resistance level)
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()

        self.logger.debug(f"Created {len(windows) * 4} rolling features")
        return df

    def create_time_features(
        self,
        data: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Create time-based features from DatetimeIndex.

        Args:
            data: DataFrame with DatetimeIndex
            features: List of time features to create

        Returns:
            DataFrame with time features
        """
        df = data.copy()

        # Hour of day (0-23)
        if 'hour' in features:
            df['hour'] = df.index.hour
            # Cyclical encoding for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (0=Monday, 6=Sunday)
        if 'day_of_week' in features:
            df['day_of_week'] = df.index.dayofweek

        # Month (1-12)
        if 'month' in features:
            df['month'] = df.index.month
            # Cyclical encoding for month
            df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
            df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

        # Weekend indicator
        if 'is_weekend' in features:
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Holiday indicator
        if 'is_holiday' in features:
            cal = USFederalHolidayCalendar()
            holidays = cal.holidays(start=df.index.min(), end=df.index.max())
            # Normalize index to dates and check if date is in holiday set
            df['is_holiday'] = df.index.normalize().isin(holidays).astype(int)

        self.logger.debug(f"Created time-based features: {features}")
        return df

    def create_sequences_for_lstm(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        lookback: int,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D sequences for LSTM input.

        Args:
            features: DataFrame with feature columns
            target: Series with target values
            lookback: Number of timesteps to look back
            forecast_horizon: Number of timesteps to forecast (default: 1)

        Returns:
            Tuple of (X_sequences, y_targets) as numpy arrays
            - X_sequences shape: (n_samples, lookback, n_features)
            - y_targets shape: (n_samples, forecast_horizon) or (n_samples,) for single-step
        """
        # Convert to numpy arrays
        X = features.values
        y = target.values

        n_samples = len(X) - lookback - forecast_horizon + 1

        if n_samples <= 0:
            raise ValueError(
                f"Insufficient data: need at least {lookback + forecast_horizon} samples, "
                f"got {len(X)}"
            )

        # Initialize arrays
        X_sequences = np.zeros((n_samples, lookback, X.shape[1]))

        if forecast_horizon == 1:
            y_targets = np.zeros(n_samples)
        else:
            y_targets = np.zeros((n_samples, forecast_horizon))

        # Create sequences using sliding window
        for i in range(n_samples):
            X_sequences[i] = X[i:i + lookback]

            if forecast_horizon == 1:
                y_targets[i] = y[i + lookback]
            else:
                y_targets[i] = y[i + lookback:i + lookback + forecast_horizon]

        self.logger.info(
            f"Created {n_samples} LSTM sequences with lookback={lookback}, "
            f"forecast_horizon={forecast_horizon}"
        )

        return X_sequences, y_targets

    def get_feature_names(self) -> List[str]:
        """
        Return list of all feature column names.

        Returns:
            List of feature names (excluding target)
        """
        return self._feature_names

    def inverse_transform_target(
        self,
        scaled_values: np.ndarray,
        scaler: Optional[object] = None
    ) -> np.ndarray:
        """
        Inverse transform scaled target values.

        Args:
            scaled_values: Scaled values to transform
            scaler: Scikit-learn scaler object (if None, returns values as-is)

        Returns:
            Original scale values
        """
        if scaler is None:
            return scaled_values

        # Reshape for scaler if needed
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)

        return scaler.inverse_transform(scaled_values).flatten()

    def _validate_data(self, data: pd.DataFrame, target_col: str) -> None:
        """
        Validate input data.

        Args:
            data: DataFrame to validate
            target_col: Name of target column

        Raises:
            ValueError: If validation fails
        """
        # Check DataFrame has DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        # Check target column exists
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Check for sufficient data
        forecasting_config = self.config.get('models', {}).get('forecasting', {})
        lookback_hours = forecasting_config.get('lookback_hours', 168)
        forecast_horizon = forecasting_config.get('forecast_horizon', 24)
        min_required = lookback_hours + forecast_horizon

        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} samples, got {len(data)}"
            )

        self.logger.debug(
            f"Data validation passed: {len(data)} samples "
            f"(lookback={lookback_hours}h, horizon={forecast_horizon}h)"
        )

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values after feature creation.

        Args:
            data: DataFrame potentially containing NaN values

        Returns:
            DataFrame with NaN values removed
        """
        initial_rows = len(data)

        # Drop rows with any NaN (conservative approach for time-series)
        df = data.dropna()

        rows_dropped = initial_rows - len(df)

        if rows_dropped > 0:
            self.logger.info(f"Dropped {rows_dropped} rows with NaN values")

        return df
