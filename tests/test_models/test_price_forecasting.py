"""
Unit tests for price forecasting models and pipeline.

Tests ARIMA, XGBoost, LSTM models, and the complete forecasting pipeline
including training, evaluation, ensemble creation, and predictions.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.feature_engineering import FeatureEngineer
from src.models.price_forecasting import (
    ARIMAForecaster,
    LSTMForecaster,
    PriceForecastingPipeline,
    XGBoostForecaster,
)


class TestARIMAForecaster:
    """Test ARIMA model."""

    def test_init(self):
        """Test initialization with order parameter."""
        forecaster = ARIMAForecaster(order=(1, 1, 1))
        assert forecaster.order == (1, 1, 1)
        assert forecaster.fitted_model is None

    def test_fit(self, sample_price_data):
        """Test model fitting."""
        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(sample_price_data["price"].values)

        # Verify fitted_model is not None
        assert forecaster.fitted_model is not None

    def test_predict(self, sample_price_data):
        """Test prediction."""
        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(sample_price_data["price"].values)

        forecast = forecaster.predict(steps=10)

        # Verify forecast length matches steps parameter
        assert "forecast" in forecast
        assert len(forecast["forecast"]) == 10

    def test_predict_not_fitted(self):
        """Test error when predicting before fitting."""
        forecaster = ARIMAForecaster()
        with pytest.raises(Exception):  # NotFittedError or similar
            forecaster.predict(steps=10)

    def test_save_load(self, tmp_path, sample_price_data):
        """Test model persistence."""
        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(sample_price_data["price"].values)

        # Get predictions before save
        forecast_before = forecaster.predict(steps=5)

        # Save model
        save_path = tmp_path / "arima_model.pkl"
        forecaster.save(str(save_path))

        # Load model
        loaded_forecaster = ARIMAForecaster()
        loaded_forecaster.load(str(save_path))

        # Verify predictions match
        forecast_after = loaded_forecaster.predict(steps=5)
        np.testing.assert_allclose(
            forecast_before["forecast"],
            forecast_after["forecast"],
            rtol=0.01,
        )


class TestXGBoostForecaster:
    """Test XGBoost model."""

    def test_init(self):
        """Test initialization with hyperparameters."""
        forecaster = XGBoostForecaster(n_estimators=50, max_depth=5)
        assert forecaster.params["n_estimators"] == 50
        assert forecaster.params["max_depth"] == 5

    def test_fit(self, sample_price_data, sample_config):
        """Test model fitting with features."""
        config = sample_config.copy()
        config['models']['features']['lags'] = [1, 2, 3]
        engineer = FeatureEngineer(config=config)
        features_df = engineer.create_features(sample_price_data, target_col="price")

        X = features_df.drop(columns=["price"])
        y = features_df["price"]

        forecaster = XGBoostForecaster(n_estimators=10)
        forecaster.fit(X.values, y.values)

        # Verify model is fitted
        assert forecaster.model is not None

    def test_predict(self, sample_price_data, sample_config):
        """Test prediction."""
        config = sample_config.copy()
        config['models']['features']['lags'] = [1, 2, 3]
        engineer = FeatureEngineer(config=config)
        features_df = engineer.create_features(sample_price_data, target_col="price")

        X = features_df.drop(columns=["price"])
        y = features_df["price"]

        forecaster = XGBoostForecaster(n_estimators=10)
        forecaster.fit(X.values, y.values)

        forecast = forecaster.predict(X.values[:10])

        # Verify forecast shape
        assert "forecast" in forecast
        assert len(forecast["forecast"]) == 10

    def test_save_load(self, tmp_path, sample_price_data, sample_config):
        """Test model persistence with joblib."""
        config = sample_config.copy()
        config['models']['features']['lags'] = [1, 2, 3]
        engineer = FeatureEngineer(config=config)
        features_df = engineer.create_features(sample_price_data, target_col="price")

        X = features_df.drop(columns=["price"])
        y = features_df["price"]

        forecaster = XGBoostForecaster(n_estimators=10)
        forecaster.fit(X.values, y.values)

        # Save and load
        save_path = tmp_path / "xgb_model.pkl"
        forecaster.save(str(save_path))

        loaded_forecaster = XGBoostForecaster()
        loaded_forecaster.load(str(save_path))

        # Verify predictions match
        forecast_original = forecaster.predict(X.values[:5])
        forecast_loaded = loaded_forecaster.predict(X.values[:5])

        np.testing.assert_allclose(
            forecast_original["forecast"],
            forecast_loaded["forecast"],
            rtol=0.01,
        )


@pytest.mark.slow
class TestLSTMForecaster:
    """Test LSTM model (mark as slow)."""

    def test_init(self):
        """Test initialization with architecture parameters."""
        forecaster = LSTMForecaster(units=32, layers=2, epochs=5)
        assert forecaster.units == 32
        assert forecaster.layers == 2
        assert forecaster.epochs == 5

    def test_build_model(self):
        """Test model architecture building."""
        forecaster = LSTMForecaster(units=10, layers=1)
        model = forecaster.build_model(input_shape=(24, 5))

        # Verify model layers
        assert model is not None
        assert len(model.layers) > 0

    def test_fit(self, sample_price_data, sample_config):
        """Test model training."""
        engineer = FeatureEngineer(config=sample_config)
        features_df = engineer.create_features(sample_price_data, target_col="price")

        # Split features and target for LSTM
        feature_cols = [col for col in features_df.columns if col != "price"]
        X_features = features_df[feature_cols]
        y_target = features_df["price"]

        X_seq, y_targets = engineer.create_sequences_for_lstm(
            features=X_features,
            target=y_target,
            lookback=10,
            forecast_horizon=1,
        )

        # Use small architecture for speed
        forecaster = LSTMForecaster(units=10, layers=1, epochs=2, batch_size=16)
        forecaster.fit(X_seq, y_targets)

        # Verify training history
        assert forecaster.model is not None

    def test_predict(self, sample_price_data, sample_config):
        """Test prediction."""
        engineer = FeatureEngineer(config=sample_config)
        features_df = engineer.create_features(sample_price_data, target_col="price")

        # Split features and target for LSTM
        feature_cols = [col for col in features_df.columns if col != "price"]
        X_features = features_df[feature_cols]
        y_target = features_df["price"]

        X_seq, y_targets = engineer.create_sequences_for_lstm(
            features=X_features,
            target=y_target,
            lookback=10,
            forecast_horizon=1,
        )

        forecaster = LSTMForecaster(units=10, layers=1, epochs=2)
        forecaster.fit(X_seq, y_targets)

        forecast = forecaster.predict(X_seq[:5])

        # Verify forecast shape
        assert "forecast" in forecast
        assert len(forecast["forecast"]) == 5


class TestPriceForecastingPipeline:
    """Test forecasting pipeline."""

    def test_init(self):
        """Test initialization."""
        pipeline = PriceForecastingPipeline()
        assert pipeline.config is not None

    def test_prepare_data(self, sample_price_data):
        """Test data preparation."""
        pipeline = PriceForecastingPipeline()
        pipeline.prepare_data(sample_price_data, target_col="price")

        # Verify train/val/test split
        assert pipeline.X_train is not None
        assert pipeline.X_val is not None
        assert pipeline.X_test is not None

    def test_train_models(self, sample_price_data):
        """Test model training."""
        pipeline = PriceForecastingPipeline()
        pipeline.prepare_data(sample_price_data, target_col="price")

        # Train only ARIMA and XGBoost (skip LSTM for speed)
        pipeline.train_models(model_types=["arima", "xgboost"])

        # Verify models stored
        assert "arima" in pipeline.models
        assert "xgboost" in pipeline.models

    def test_evaluate_models(self, sample_price_data):
        """Test model evaluation."""
        pipeline = PriceForecastingPipeline()
        pipeline.prepare_data(sample_price_data, target_col="price")
        pipeline.train_models(model_types=["arima"])

        metrics = pipeline.evaluate_models()

        # Verify metrics dict returned
        assert "arima" in metrics
        assert "mae" in metrics["arima"]
        assert "rmse" in metrics["arima"]
        assert "mape" in metrics["arima"]

        # Check metrics are reasonable
        assert metrics["arima"]["rmse"] > 0
        assert metrics["arima"]["mape"] < 100

    def test_create_ensemble(self, sample_price_data):
        """Test ensemble creation."""
        pipeline = PriceForecastingPipeline()
        pipeline.prepare_data(sample_price_data, target_col="price")
        pipeline.train_models(model_types=["arima", "xgboost"])
        pipeline.evaluate_models()
        pipeline.create_ensemble()

        # Verify ensemble_weights dict created
        assert pipeline.ensemble_weights is not None
        assert "arima" in pipeline.ensemble_weights
        assert "xgboost" in pipeline.ensemble_weights

        # Check weights sum to 1.0
        weight_sum = sum(pipeline.ensemble_weights.values())
        np.testing.assert_almost_equal(weight_sum, 1.0, decimal=5)

    def test_predict(self, sample_price_data):
        """Test prediction."""
        pipeline = PriceForecastingPipeline()
        pipeline.prepare_data(sample_price_data, target_col="price")
        pipeline.train_models(model_types=["arima"])
        pipeline.evaluate_models()
        pipeline.create_ensemble()

        forecast_df = pipeline.predict(
            sample_price_data.tail(20),
            model_name="arima",
            steps=5,
        )

        # Verify forecast DataFrame structure
        assert isinstance(forecast_df, pd.DataFrame)
        assert "forecast" in forecast_df.columns
        assert len(forecast_df) == 5

    def test_save_load_models(self, tmp_path, sample_price_data):
        """Test pipeline persistence."""
        pipeline = PriceForecastingPipeline()
        pipeline.prepare_data(sample_price_data, target_col="price")
        pipeline.train_models(model_types=["arima"])

        # Save models
        pipeline.save_models(str(tmp_path))

        # Load models using classmethod
        loaded_pipeline = PriceForecastingPipeline.load_models(str(tmp_path))

        # Verify models loaded
        assert "arima" in loaded_pipeline.models
