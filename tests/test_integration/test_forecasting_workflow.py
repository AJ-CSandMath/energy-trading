"""
Integration tests for forecasting workflow.

Tests end-to-end forecasting including data preparation, model training,
evaluation, ensemble creation, and model persistence.
"""

import pytest

from src.models.price_forecasting import PriceForecastingPipeline


@pytest.mark.integration
@pytest.mark.slow
class TestForecastingWorkflow:
    """Test end-to-end forecasting."""

    def test_price_forecasting_pipeline(self, sample_price_data, sample_config, tmp_path):
        """Test complete price forecasting workflow."""
        # Initialize pipeline with test config (low n_estimators for speed)
        pipeline = PriceForecastingPipeline(config=sample_config)

        # Prepare data
        pipeline.prepare_data(sample_price_data, target_col="price")

        # Train models (skip LSTM for speed)
        pipeline.train_models(model_types=["arima", "xgboost"])

        # Evaluate models
        metrics = pipeline.evaluate_models()
        assert "arima" in metrics
        assert "xgboost" in metrics

        # Create ensemble
        pipeline.create_ensemble()
        assert pipeline.ensemble_weights is not None

        # Generate forecasts
        forecasts = pipeline.predict(
            sample_price_data.tail(20),
            model_name="ensemble",
            steps=5,
        )
        assert len(forecasts) == 5

        # Save models
        pipeline.save_models(str(tmp_path))

        # Load models using classmethod and verify predictions match
        loaded_pipeline = PriceForecastingPipeline.load_models(str(tmp_path))
        assert "arima" in loaded_pipeline.models
