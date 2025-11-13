"""
Unit tests for FeatureEngineer class.

Tests feature creation including lag features, rolling statistics,
time-based features, and LSTM sequence generation.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test feature engineering."""

    def test_init(self, sample_config):
        """Test initialization with config."""
        engineer = FeatureEngineer(config=sample_config)
        assert engineer.lags is not None
        assert engineer.rolling_windows is not None

    def test_create_features(self, sample_price_data, sample_config):
        """Test complete feature creation."""
        engineer = FeatureEngineer(config=sample_config)
        features_df = engineer.create_features(
            sample_price_data,
            target_col="price",
        )

        # Verify all feature types created
        assert "price_lag_1" in features_df.columns
        assert "price_rolling_mean_24" in features_df.columns or any(
            "rolling" in col for col in features_df.columns
        )
        assert "hour" in features_df.columns
        assert len(features_df) <= len(sample_price_data)  # NaN rows dropped

    def test_create_lag_features(self, sample_price_data, sample_config):
        """Test lag feature creation."""
        # Configure lags via config
        config = sample_config.copy()
        config['models']['features']['lags'] = [1, 2, 24]
        engineer = FeatureEngineer(config=config)
        lagged_df = engineer.create_lag_features(
            sample_price_data,
            target_col="price",
            lags=[1, 2, 24],
        )

        # Verify lag columns created
        assert "price_lag_1" in lagged_df.columns
        assert "price_lag_2" in lagged_df.columns
        assert "price_lag_24" in lagged_df.columns

        # Check lag values match shifted original
        np.testing.assert_array_equal(
            lagged_df["price_lag_1"].values[1:],
            sample_price_data["price"].values[:-1],
        )

    def test_create_rolling_features(self, sample_price_data, sample_config):
        """Test rolling statistics."""
        # Configure rolling windows via config
        config = sample_config.copy()
        config['models']['features']['rolling_windows'] = [12, 24]
        engineer = FeatureEngineer(config=config)
        rolling_df = engineer.create_rolling_features(
            sample_price_data,
            target_col="price",
            windows=[12, 24],
        )

        # Verify rolling mean, std created
        assert any("rolling_mean" in col for col in rolling_df.columns)
        assert any("rolling_std" in col for col in rolling_df.columns)

    def test_create_time_features(self, sample_price_data, sample_config):
        """Test time-based features."""
        engineer = FeatureEngineer(config=sample_config)
        time_df = engineer.create_time_features(sample_price_data, features=["hour", "day_of_week", "month", "is_weekend"])

        # Verify hour, day_of_week, month created
        assert "hour" in time_df.columns
        assert "day_of_week" in time_df.columns
        assert "month" in time_df.columns

        # Check cyclical encoding
        assert "hour_sin" in time_df.columns
        assert "hour_cos" in time_df.columns

        # Verify is_weekend flag
        assert "is_weekend" in time_df.columns
        assert set(time_df["is_weekend"].unique()).issubset({0, 1})

    def test_create_sequences_for_lstm(self, sample_price_data, sample_config):
        """Test LSTM sequence creation."""
        engineer = FeatureEngineer(config=sample_config)
        features_df = engineer.create_features(sample_price_data, target_col="price")

        # Split features and target
        feature_cols = [col for col in features_df.columns if col != "price"]
        X_features = features_df[feature_cols]
        y_target = features_df["price"]

        X_seq, y_targets = engineer.create_sequences_for_lstm(
            features=X_features,
            target=y_target,
            lookback=24,
            forecast_horizon=1,
        )

        # Verify X_sequences shape (n_samples, lookback, n_features)
        assert X_seq.ndim == 3
        assert X_seq.shape[1] == 24  # lookback
        assert X_seq.shape[2] == len(feature_cols)  # n_features

        # Verify y_targets shape (n_samples,) for single-step forecast
        assert y_targets.ndim == 1

    def test_create_sequences_insufficient_data(self, sample_config):
        """Test ValueError for insufficient data."""
        engineer = FeatureEngineer(config=sample_config)
        small_df = pd.DataFrame(
            {"price": [50, 51, 52]},
            index=pd.date_range("2023-01-01", periods=3, freq="H"),
        )

        # Create dummy features (just use price as feature)
        features = small_df[["price"]]
        target = small_df["price"]

        with pytest.raises(ValueError):
            engineer.create_sequences_for_lstm(
                features=features,
                target=target,
                lookback=10,  # More than available data
            )

    def test_get_feature_names(self, sample_price_data, sample_config):
        """Test feature name retrieval."""
        engineer = FeatureEngineer(config=sample_config)
        features_df = engineer.create_features(sample_price_data, target_col="price")
        feature_names = engineer.get_feature_names()

        # Verify returns list of feature names
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "price" not in feature_names  # Target excluded

    def test_validate_data_valid(self, sample_price_data, sample_config):
        """Test validation passes for valid data."""
        engineer = FeatureEngineer(config=sample_config)
        # _validate_data is a private method, call it via the public API
        engineer._validate_data(sample_price_data, target_col="price")
        # If no exception raised, validation passed
        assert True

    def test_validate_data_no_datetime_index(self, sample_config):
        """Test ValueError for non-DatetimeIndex."""
        engineer = FeatureEngineer(config=sample_config)
        df = pd.DataFrame({"price": [50, 51, 52]})

        with pytest.raises(ValueError, match="DatetimeIndex"):
            engineer._validate_data(df, target_col="price")

    def test_validate_data_missing_target_column(self, sample_price_data, sample_config):
        """Test ValueError for missing target."""
        engineer = FeatureEngineer(config=sample_config)

        with pytest.raises(ValueError, match="target"):
            engineer._validate_data(sample_price_data, target_col="missing_col")

    def test_validate_data_insufficient_data(self, sample_config):
        """Test ValueError for too few samples."""
        engineer = FeatureEngineer(config=sample_config)
        small_df = pd.DataFrame(
            {"price": [50]},
            index=pd.date_range("2023-01-01", periods=1, freq="H"),
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            engineer._validate_data(small_df, target_col="price")

    @pytest.mark.parametrize("lags", [[1, 2, 3], [1, 6, 24], [1, 24, 48]])
    def test_different_lag_configurations(self, sample_price_data, sample_config, lags):
        """Test with different lag configurations."""
        # Configure lags via config
        config = sample_config.copy()
        config['models']['features']['lags'] = lags
        engineer = FeatureEngineer(config=config)
        lagged_df = engineer.create_lag_features(sample_price_data, target_col="price", lags=lags)

        # Verify all lag columns created
        for lag in lags:
            assert f"price_lag_{lag}" in lagged_df.columns

    @pytest.mark.parametrize("windows", [[12, 24], [24, 48]])
    def test_different_rolling_windows(self, sample_price_data, sample_config, windows):
        """Test with different rolling windows."""
        # Configure rolling windows via config
        config = sample_config.copy()
        config['models']['features']['rolling_windows'] = windows
        engineer = FeatureEngineer(config=config)
        rolling_df = engineer.create_rolling_features(sample_price_data, target_col="price", windows=windows)

        # Verify rolling features for each window
        for window in windows:
            assert any(f"{window}" in col for col in rolling_df.columns)
