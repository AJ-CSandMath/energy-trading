"""
Unit tests for DataManager class.

Tests data persistence, retrieval, partitioning, date filtering,
validation, and various data operations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.data_manager import DataManager


class TestDataManager:
    """Test data persistence and retrieval."""

    def test_init_with_paths(self, tmp_path):
        """Test initialization with custom paths."""
        raw_path = tmp_path / "raw"
        processed_path = tmp_path / "processed"

        manager = DataManager(
            raw_data_path=str(raw_path),
            processed_data_path=str(processed_path),
        )

        # Verify directories are created
        assert raw_path.exists()
        assert processed_path.exists()

        # Check paths are set correctly
        assert manager.raw_data_path == raw_path
        assert manager.processed_data_path == processed_path

    def test_init_from_config(self, sample_config):
        """Test initialization from config."""
        manager = DataManager(config=sample_config)
        assert manager.raw_data_path is not None
        assert manager.processed_data_path is not None

    def test_save_raw_data(self, tmp_path, sample_price_data):
        """Test saving raw data."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        filepath = manager.save_raw_data(
            data=sample_price_data,
            source="synthetic",
            dataset="prices",
            start_date="2023-01-01",
            end_date="2023-01-05",
        )

        # Verify Parquet file created
        assert filepath.exists()
        assert filepath.suffix == ".parquet"
        assert "synthetic" in filepath.name
        assert "prices" in filepath.name

        # Verify file size > 0
        assert filepath.stat().st_size > 0

    def test_save_raw_data_empty_dataframe(self, tmp_path):
        """Test returns None for empty DataFrame."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        empty_df = pd.DataFrame()
        result = manager.save_raw_data(
            data=empty_df,
            source="test",
            dataset="empty",
        )

        assert result is None

    def test_save_processed_data(self, tmp_path, sample_price_data):
        """Test saving processed data."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        # Test Parquet format
        filepath_parquet = manager.save_processed_data(
            data=sample_price_data,
            source="synthetic",
            dataset="prices",
            format="parquet",
        )
        assert filepath_parquet.suffix == ".parquet"

        # Test CSV format
        filepath_csv = manager.save_processed_data(
            data=sample_price_data,
            source="synthetic",
            dataset="prices_csv",
            format="csv",
        )
        assert filepath_csv.suffix == ".csv"

    def test_save_partitioned(self, tmp_path, sample_price_data):
        """Test partitioned save."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
            partition_by_date=True,
        )

        filepath = manager.save_raw_data(
            data=sample_price_data,
            source="synthetic",
            dataset="prices",
        )

        # Verify directory structure created
        assert filepath.is_dir()  # Returns directory for partitioned save

        # Check for year/month subdirectories
        partitions = list(filepath.glob("*/*/*.parquet"))
        assert len(partitions) > 0

    def test_load_data(self, tmp_path, sample_price_data):
        """Test loading data."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        # Save data first
        manager.save_raw_data(
            data=sample_price_data,
            source="synthetic",
            dataset="prices",
        )

        # Load data
        loaded_df = manager.load_data(
            source="synthetic",
            dataset="prices",
            data_type="raw",
        )

        # Verify loaded DataFrame matches original
        assert len(loaded_df) == len(sample_price_data)
        assert list(loaded_df.columns) == list(sample_price_data.columns)
        pd.testing.assert_frame_equal(
            loaded_df.sort_index(),
            sample_price_data.sort_index(),
            check_dtype=False,
        )

    def test_load_data_with_date_filter(self, tmp_path, sample_price_data):
        """Test date filtering."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        # Save data
        manager.save_raw_data(
            data=sample_price_data,
            source="synthetic",
            dataset="prices",
        )

        # Load with date filter
        loaded_df = manager.load_data(
            source="synthetic",
            dataset="prices",
            data_type="raw",
            start_date="2023-01-02",
            end_date="2023-01-03",
            date_filter=True,
        )

        # Verify only filtered data returned
        assert len(loaded_df) < len(sample_price_data)
        assert loaded_df.index.min() >= pd.Timestamp("2023-01-02")
        assert loaded_df.index.max() <= pd.Timestamp("2023-01-04")

    def test_load_data_not_found(self, tmp_path):
        """Test FileNotFoundError for missing data."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        with pytest.raises(FileNotFoundError):
            manager.load_data(
                source="nonexistent",
                dataset="missing",
                data_type="raw",
            )

    def test_merge_datasets(self, tmp_path, sample_price_data):
        """Test merging multiple files."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        # Save multiple datasets with different date ranges
        df1 = sample_price_data.iloc[:30]
        df2 = sample_price_data.iloc[30:60]

        manager.save_raw_data(data=df1, source="test", dataset="prices_1")
        manager.save_raw_data(data=df2, source="test", dataset="prices_2")

        # Merge datasets (using glob pattern in dataset name)
        merged = manager.merge_datasets(
            source="test",
            dataset="prices",
            data_type="raw",
        )

        # Verify all data combined
        assert len(merged) >= len(df1) + len(df2) - 10  # Allow for some overlap

    def test_remove_duplicates(self, sample_price_data):
        """Test duplicate removal."""
        manager = DataManager()

        # Create DataFrame with duplicate timestamps
        duplicated_df = pd.concat([sample_price_data, sample_price_data.head(10)])

        cleaned_df = manager.remove_duplicates(duplicated_df)

        # Verify duplicates removed
        assert len(cleaned_df) == len(sample_price_data)
        assert not cleaned_df.index.duplicated().any()

    def test_fill_missing_timestamps(self, sample_price_data):
        """Test filling missing timestamps."""
        manager = DataManager()

        # Create data with gaps
        gapped_df = sample_price_data.iloc[::2]  # Take every other row

        # Fill with interpolation
        filled_df = manager.fill_missing_timestamps(
            gapped_df,
            freq="H",
            method="interpolate",
        )

        # Verify gaps filled
        assert len(filled_df) > len(gapped_df)
        assert not filled_df.isna().any().any()

    def test_resample_frequency(self, sample_price_data):
        """Test resampling."""
        manager = DataManager()

        # Resample hourly to daily
        daily_df = manager.resample_frequency(
            sample_price_data,
            freq="D",
            agg_func="mean",
        )

        # Verify aggregation
        assert len(daily_df) < len(sample_price_data)
        assert daily_df.index.freq == pd.DateOffset(days=1) or daily_df.index.freq is None

    def test_validate_data(self):
        """Test data validation."""
        manager = DataManager()

        # Valid data
        valid_df = pd.DataFrame(
            {"price": [50, 51, 52]},
            index=pd.date_range("2023-01-01", periods=3, freq="H"),
        )
        validated = manager.validate_data(valid_df)
        assert len(validated) == 3

        # Empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            manager.validate_data(empty_df, raise_on_empty=True)

        # All-null columns
        null_df = pd.DataFrame(
            {"price": [50, 51, 52], "null_col": [None, None, None]},
            index=pd.date_range("2023-01-01", periods=3, freq="H"),
        )
        validated_null = manager.validate_data(null_df)
        assert "null_col" not in validated_null.columns  # Dropped

    def test_get_available_datasets(self, tmp_path, sample_price_data):
        """Test dataset listing."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        # Save multiple datasets
        manager.save_raw_data(data=sample_price_data, source="test", dataset="prices_1")
        manager.save_raw_data(data=sample_price_data, source="test", dataset="prices_2")

        # List datasets
        datasets = manager.get_available_datasets(data_type="raw")

        assert isinstance(datasets, list)
        assert len(datasets) >= 2
        # Check first dataset has expected keys
        if len(datasets) > 0:
            assert "name" in datasets[0]
            assert "rows" in datasets[0]
            assert "size_mb" in datasets[0]

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli", "none"])
    def test_compression_formats(self, tmp_path, sample_price_data, compression):
        """Test different compression codecs."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
            compression=compression,
        )

        # Save with compression
        filepath = manager.save_raw_data(
            data=sample_price_data,
            source="test",
            dataset=f"prices_{compression}",
        )

        # Verify can be saved and loaded
        assert filepath.exists()

        loaded_df = manager.load_data(
            source="test",
            dataset=f"prices_{compression}",
            data_type="raw",
        )

        assert len(loaded_df) == len(sample_price_data)
