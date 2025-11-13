"""
Unit tests for data fetcher classes (EIAFetcher and CAISOFetcher).

Tests API initialization, data fetching, pagination, error handling,
and retry logic with mocked HTTP requests.
"""

import io
import zipfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from src.data.data_fetcher import CAISOFetcher, EIAFetcher


class TestEIAFetcher:
    """Test EIA API fetcher."""

    def test_init_with_api_key(self):
        """Test initialization with API key parameter."""
        fetcher = EIAFetcher(api_key="test_key_123")
        assert fetcher.api_key == "test_key_123"
        assert fetcher.base_url is not None

    def test_init_from_env(self, monkeypatch):
        """Test initialization from EIA_API_KEY environment variable."""
        monkeypatch.setenv("EIA_API_KEY", "env_key_456")
        fetcher = EIAFetcher()
        assert fetcher.api_key == "env_key_456"

    def test_init_from_config(self, sample_config):
        """Test initialization from config dictionary."""
        config = sample_config.copy()
        config["api"] = {"eia": {"api_key": "config_key_789"}}
        fetcher = EIAFetcher(config=config)
        assert fetcher.api_key == "config_key_789"

    def test_init_no_api_key_raises(self, monkeypatch):
        """Test ValueError raised when no API key provided."""
        monkeypatch.delenv("EIA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            EIAFetcher()

    @patch("requests.Session.get")
    def test_fetch_electricity_data_success(self, mock_get):
        """Test successful data fetch with mocked response."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "data": [
                    {"period": "2023-01-01T00:00", "value": 50.5},
                    {"period": "2023-01-01T01:00", "value": 52.3},
                ],
                "total": 2,
            }
        }
        mock_get.return_value = mock_response

        fetcher = EIAFetcher(api_key="test_key")
        df = fetcher.fetch_electricity_data(
            endpoint="retail-sales",
            state=["CA"],
            start_date="2023-01-01",
            end_date="2023-01-31",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert pd.api.types.is_datetime64_any_dtype(df.index)
        assert "value" in df.columns

    @patch("requests.Session.get")
    def test_fetch_electricity_data_pagination(self, mock_get):
        """Test pagination with >5000 rows."""
        # Mock multiple responses for pagination
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            "response": {
                "data": [{"period": f"2023-{i:04d}", "value": float(i)} for i in range(5000)],
                "total": 6000,
            }
        }

        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            "response": {
                "data": [
                    {"period": f"2023-{i:04d}", "value": float(i)} for i in range(5000, 6000)
                ],
                "total": 6000,
            }
        }

        mock_get.side_effect = [mock_response_1, mock_response_2]

        fetcher = EIAFetcher(api_key="test_key")
        df = fetcher.fetch_electricity_data(
            endpoint="retail-sales",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert len(df) == 6000
        assert mock_get.call_count == 2

    @patch("requests.Session.get")
    def test_fetch_electricity_data_api_error(self, mock_get):
        """Test APIError raised on API error response."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Bad Request")
        mock_get.return_value = mock_response

        fetcher = EIAFetcher(api_key="test_key")
        with pytest.raises(Exception):  # APIError or similar
            fetcher.fetch_electricity_data(
                endpoint="invalid-endpoint",
                start_date="2023-01-01",
                end_date="2023-01-31",
            )

    @patch("requests.Session.get")
    def test_fetch_electricity_data_network_error(self, mock_get):
        """Test APIError raised on network failure."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        fetcher = EIAFetcher(api_key="test_key")
        with pytest.raises(Exception):
            fetcher.fetch_electricity_data(
                endpoint="retail-sales",
                start_date="2023-01-01",
                end_date="2023-01-31",
            )

    def test_date_validation(self):
        """Test ValueError raised for invalid date formats."""
        fetcher = EIAFetcher(api_key="test_key")
        with pytest.raises(ValueError):
            fetcher.fetch_electricity_data(
                endpoint="retail-sales",
                start_date="invalid-date",
                end_date="2023-01-31",
            )

    @patch("requests.Session.get")
    def test_retry_logic(self, mock_get):
        """Test retry mechanism on 429/500 errors."""
        # First call fails with 429 (HTTPError), second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 429
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.HTTPError("Rate limit")

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status.return_value = None
        mock_response_success.json.return_value = {
            "response": {"data": [{"period": "2023-01-01T00:00", "value": 50.0}], "total": 1}
        }

        mock_get.side_effect = [mock_response_fail, mock_response_success]

        fetcher = EIAFetcher(api_key="test_key", max_retries=3)
        df = fetcher.fetch_electricity_data(
            endpoint="retail-sales",
            start_date="2023-01-01",
            end_date="2023-01-31",
        )

        assert len(df) == 1
        assert mock_get.call_count == 2


class TestCAISOFetcher:
    """Test CAISO API fetcher."""

    def test_init(self, sample_config):
        """Test initialization with config."""
        fetcher = CAISOFetcher(config=sample_config)
        assert fetcher.base_url is not None
        assert fetcher.timeout > 0

    def test_format_caiso_datetime_valid(self):
        """Test datetime format validation with valid input."""
        fetcher = CAISOFetcher()
        formatted = fetcher._format_caiso_datetime("20230101T00:00-0000")
        assert formatted == "20230101T00:00-0000"

    def test_format_caiso_datetime_invalid(self):
        """Test ValueError raised for invalid formats."""
        fetcher = CAISOFetcher()
        with pytest.raises(ValueError):
            fetcher._format_caiso_datetime("invalid")

    @patch("requests.Session.get")
    def test_fetch_lmp_data_success(self, mock_get):
        """Test successful LMP data fetch."""
        # Create mock ZIP file with CSV content
        csv_content = "INTERVALSTARTTIME_GMT,NODE,LMP\n2023-01-01T00:00,NP15,45.50\n"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("lmp_data.csv", csv_content)
        zip_buffer.seek(0)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = zip_buffer.read()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        fetcher = CAISOFetcher()
        df = fetcher.fetch_lmp_data(
            start_date="20230101T00:00-0000",
            end_date="20230102T00:00-0000",
            market_run_id="DAM",
            nodes=["NP15"],
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "LMP" in df.columns

    def test_fetch_lmp_data_invalid_market_query_combo(self):
        """Test ValueError for invalid market_run_id/query_name combination."""
        fetcher = CAISOFetcher()
        with pytest.raises(ValueError, match="combination"):
            fetcher.fetch_lmp_data(
                start_date="20230101T00:00-0000",
                end_date="20230102T00:00-0000",
                market_run_id="INVALID",
                query_name="INVALID_QUERY",
            )

    def test_fetch_lmp_data_date_order_validation(self):
        """Test ValueError when start_date > end_date."""
        fetcher = CAISOFetcher()
        with pytest.raises(ValueError, match="start_date.*end_date"):
            fetcher.fetch_lmp_data(
                start_date="20230110T00:00-0000",
                end_date="20230101T00:00-0000",
                market_run_id="DAM",
            )

    @patch("requests.Session.get")
    def test_fetch_demand_forecast_success(self, mock_get):
        """Test demand forecast fetch."""
        csv_content = "INTERVALSTARTTIME_GMT,DEMAND_FORECAST\n2023-01-01T00:00,35000\n"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("forecast.csv", csv_content)
        zip_buffer.seek(0)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = zip_buffer.read()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        fetcher = CAISOFetcher()
        df = fetcher.fetch_demand_forecast(
            start_date="20230101T00:00-0000",
            end_date="20230102T00:00-0000",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @patch("requests.Session.get")
    def test_fetch_lmp_data_bad_zip(self, mock_get):
        """Test APIError on invalid ZIP response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"NOT A ZIP FILE"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        fetcher = CAISOFetcher()
        with pytest.raises(Exception):  # APIError or ZipFile error
            fetcher.fetch_lmp_data(
                start_date="20230101T00:00-0000",
                end_date="20230102T00:00-0000",
                market_run_id="DAM",
            )
