"""
Data management module for handling Parquet storage and retrieval.

This module provides the DataManager class for efficient storage and retrieval
of time-series energy market data using Apache Parquet format with PyArrow engine.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


# Get logger (no basicConfig - central config handles logging)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manager for storing and retrieving energy market data in Parquet format.

    Uses Apache Parquet columnar storage format for efficient time-series data
    operations with PyArrow engine for optimal performance and compression.

    Features:
    - Automatic partitioning by date (year/month)
    - Configurable compression (snappy, gzip, brotli)
    - Data validation and deduplication
    - Metadata tracking for datasets

    Example:
        >>> manager = DataManager(
        ...     raw_data_path="data/raw",
        ...     processed_data_path="data/processed"
        ... )
        >>> manager.save_raw_data(
        ...     data=df,
        ...     source="eia",
        ...     dataset="electricity",
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31"
        ... )
    """

    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        processed_data_path: Optional[str] = None,
        compression: Optional[str] = None,
        partition_by_date: Optional[bool] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize DataManager.

        Args:
            raw_data_path: Path to raw data directory. If None, reads from config.
            processed_data_path: Path to processed data directory. If None, reads from config.
            compression: Compression codec ("snappy", "gzip", "brotli", "none"). If None, reads from config.
            partition_by_date: Whether to partition data by year/month. If None, reads from config.
            config: Configuration dictionary. If None, loads from config.yaml.
        """
        # Load config if not provided
        if config is None:
            from src.config.load_config import get_config
            config = get_config()

        # Get data storage settings from config
        data_config = config.get("data", {})
        parquet_config = data_config.get("parquet", {})

        # Use provided values or fall back to config
        self.raw_data_path = Path(raw_data_path if raw_data_path is not None else data_config.get("raw_data_path", "data/raw"))
        self.processed_data_path = Path(processed_data_path if processed_data_path is not None else data_config.get("processed_data_path", "data/processed"))
        self.compression = compression if compression is not None else parquet_config.get("compression", "snappy")
        self.partition_by_date = partition_by_date if partition_by_date is not None else parquet_config.get("partition_by_date", True)

        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DataManager initialized: raw={self.raw_data_path}, "
            f"processed={self.processed_data_path}, compression={self.compression}"
        )

    def save_raw_data(
        self,
        data: pd.DataFrame,
        source: str,
        dataset: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True
    ) -> Optional[Path]:
        """
        Save raw data fetched from APIs to Parquet format.

        Args:
            data: DataFrame to save
            source: Data source identifier (e.g., "eia", "caiso")
            dataset: Dataset name (e.g., "electricity", "lmp", "demand")
            start_date: Start date string (YYYY-MM-DD) for filename
            end_date: End date string (YYYY-MM-DD) for filename
            validate: Whether to validate data before saving

        Returns:
            Path to saved file or directory (if partitioned), or None if data is empty

        Raises:
            ValueError: If data validation fails
        """
        if data.empty:
            logger.warning("Empty DataFrame provided, skipping save")
            return None

        logger.info(
            f"Saving raw data: source={source}, dataset={dataset}, "
            f"rows={len(data)}, start={start_date}, end={end_date}"
        )

        # Validate data if requested
        if validate:
            data = self._validate_data(data)

        # Generate filename
        if start_date and end_date:
            # Remove hyphens from dates for cleaner filenames
            start_str = start_date.replace("-", "")
            end_str = end_date.replace("-", "")
            filename = f"{source}_{dataset}_{start_str}_{end_str}"
        else:
            # Use current timestamp if no dates provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{source}_{dataset}_{timestamp}"

        # Save based on partitioning strategy
        if self.partition_by_date and not data.index.empty:
            return self._save_partitioned(
                data=data,
                base_path=self.raw_data_path,
                filename_prefix=filename
            )
        else:
            filepath = self.raw_data_path / f"{filename}.parquet"
            return self._save_single_file(data, filepath)

    def save_processed_data(
        self,
        data: pd.DataFrame,
        source: str,
        dataset: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True,
        format: str = 'parquet'
    ) -> Optional[Path]:
        """
        Save processed/cleaned data to Parquet or CSV format.

        Args:
            data: DataFrame to save
            source: Data source identifier (e.g., "eia", "caiso")
            dataset: Dataset name (e.g., "electricity", "lmp", "demand")
            start_date: Start date string (YYYY-MM-DD) for filename
            end_date: End date string (YYYY-MM-DD) for filename
            validate: Whether to validate data before saving
            format: Output format ('parquet' or 'csv'). Default is 'parquet'.

        Returns:
            Path to saved file or directory (if partitioned), or None if data is empty

        Raises:
            ValueError: If data validation fails or unsupported format specified
        """
        if data.empty:
            logger.warning("Empty DataFrame provided, skipping save")
            return None

        logger.info(
            f"Saving processed data: source={source}, dataset={dataset}, "
            f"rows={len(data)}, start={start_date}, end={end_date}, format={format}"
        )

        # Validate format
        if format not in ['parquet', 'csv']:
            raise ValueError(f"Unsupported format: {format}. Must be 'parquet' or 'csv'.")

        # Validate data if requested
        if validate:
            data = self._validate_data(data)

        # Generate filename
        if start_date and end_date:
            start_str = start_date.replace("-", "")
            end_str = end_date.replace("-", "")
            filename = f"{source}_{dataset}_{start_str}_{end_str}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{source}_{dataset}_{timestamp}"

        # Save based on format
        if format == 'csv':
            # CSV format - always single file
            filepath = self.processed_data_path / f"{filename}.csv"
            return self._save_single_file_csv(data, filepath)
        else:
            # Parquet format - use partitioning strategy if applicable
            if self.partition_by_date and not data.index.empty:
                return self._save_partitioned(
                    data=data,
                    base_path=self.processed_data_path,
                    filename_prefix=filename
                )
            else:
                filepath = self.processed_data_path / f"{filename}.parquet"
                return self._save_single_file(data, filepath)

    def _save_single_file(self, data: pd.DataFrame, filepath: Path) -> Path:
        """
        Save DataFrame to a single Parquet file.

        Args:
            data: DataFrame to save
            filepath: Full path to output file

        Returns:
            Path to saved file
        """
        try:
            # Convert to PyArrow Table for better control
            table = pa.Table.from_pandas(data)

            # Write with specified compression
            pq.write_table(
                table,
                filepath,
                compression=self.compression,
                use_dictionary=True,
                write_statistics=True
            )

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(
                f"Saved {len(data)} rows to {filepath} "
                f"({file_size_mb:.2f} MB, compression={self.compression})"
            )

            return filepath

        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {str(e)}")
            raise

    def _save_single_file_csv(self, data: pd.DataFrame, filepath: Path) -> Path:
        """
        Save DataFrame to a single CSV file.

        Args:
            data: DataFrame to save
            filepath: Full path to output file

        Returns:
            Path to saved file
        """
        try:
            # Save to CSV with index (preserves timestamp if datetime index)
            data.to_csv(filepath, index=True)

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(
                f"Saved {len(data)} rows to {filepath} "
                f"({file_size_mb:.2f} MB, CSV format)"
            )

            return filepath

        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {str(e)}")
            raise

    def _save_partitioned(
        self,
        data: pd.DataFrame,
        base_path: Path,
        filename_prefix: str
    ) -> Path:
        """
        Save DataFrame partitioned by year/month.

        Args:
            data: DataFrame to save (must have datetime index)
            base_path: Base directory path
            filename_prefix: Prefix for partition directory name

        Returns:
            Path to partition directory
        """
        try:
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.warning("Index is not DatetimeIndex, converting...")
                data.index = pd.to_datetime(data.index)

            # Add partition columns
            data_copy = data.copy()
            data_copy['year'] = data_copy.index.year
            data_copy['month'] = data_copy.index.month

            # Create partition directory
            partition_path = base_path / filename_prefix

            # Convert to PyArrow Table
            table = pa.Table.from_pandas(data_copy)

            # Write partitioned dataset
            pq.write_to_dataset(
                table,
                root_path=str(partition_path),
                partition_cols=['year', 'month'],
                compression=self.compression,
                use_dictionary=True,
                existing_data_behavior='overwrite_or_ignore'
            )

            logger.info(
                f"Saved {len(data)} rows to partitioned dataset at {partition_path} "
                f"(compression={self.compression})"
            )

            return partition_path

        except Exception as e:
            logger.error(f"Failed to save partitioned data: {str(e)}")
            raise

    def load_data(
        self,
        source: str,
        dataset: str,
        data_type: str = "raw",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_filter: bool = True
    ) -> pd.DataFrame:
        """
        Load data from Parquet files with optional date filtering.

        Args:
            source: Data source identifier (e.g., "eia", "caiso")
            dataset: Dataset name (e.g., "electricity", "lmp")
            data_type: "raw" or "processed"
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)
            date_filter: Whether to apply date filtering

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If no matching files found
        """
        base_path = self.raw_data_path if data_type == "raw" else self.processed_data_path

        logger.info(
            f"Loading data: source={source}, dataset={dataset}, type={data_type}, "
            f"start={start_date}, end={end_date}"
        )

        # Find matching files/directories
        pattern = f"{source}_{dataset}_*"
        matching_paths = list(base_path.glob(pattern))

        if not matching_paths:
            raise FileNotFoundError(
                f"No data found matching: {pattern} in {base_path}"
            )

        # Load data from all matching paths
        dataframes = []

        for path in matching_paths:
            try:
                if path.is_dir():
                    # Load partitioned dataset
                    df = pd.read_parquet(path, engine='pyarrow')
                elif path.suffix == '.parquet':
                    # Load single file
                    df = pd.read_parquet(path, engine='pyarrow')
                else:
                    continue

                # Restore DatetimeIndex if timestamp column exists
                if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    df = df.sort_index()

                dataframes.append(df)

            except Exception as e:
                logger.warning(f"Failed to load {path}: {str(e)}")
                continue

        if not dataframes:
            raise ValueError("No valid data loaded from matching files")

        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=False)

        # Remove partition columns if present
        partition_cols = ['year', 'month']
        combined_df = combined_df.drop(
            columns=[col for col in partition_cols if col in combined_df.columns],
            errors='ignore'
        )

        # Apply date filtering if requested
        if date_filter and start_date and end_date:
            if isinstance(combined_df.index, pd.DatetimeIndex):
                combined_df = combined_df.loc[start_date:end_date]
            else:
                logger.warning("Cannot apply date filter: index is not DatetimeIndex")

        # Remove duplicates and sort
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()

        logger.info(f"Loaded {len(combined_df)} rows from {len(matching_paths)} file(s)")

        return combined_df

    def merge_datasets(
        self,
        source: str,
        dataset: str,
        data_type: str = "raw",
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        Merge multiple Parquet files into a single DataFrame.

        Args:
            source: Data source identifier
            dataset: Dataset name
            data_type: "raw" or "processed"
            remove_duplicates: Whether to remove duplicate rows

        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging datasets: source={source}, dataset={dataset}, type={data_type}")

        # Load all matching data
        df = self.load_data(source, dataset, data_type, date_filter=False)

        # Remove duplicates if requested
        if remove_duplicates:
            original_len = len(df)
            df = self.remove_duplicates(df)
            removed = original_len - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")

        return df

    def get_available_datasets(self, data_type: str = "raw") -> List[Dict]:
        """
        List all available datasets with metadata using PyArrow metadata only.

        This method efficiently reads metadata without loading full datasets into memory.

        Args:
            data_type: "raw" or "processed"

        Returns:
            List of dictionaries with dataset metadata
        """
        base_path = self.raw_data_path if data_type == "raw" else self.processed_data_path

        datasets = []

        # Find all parquet files and directories
        for path in base_path.iterdir():
            if path.is_file() and path.suffix == '.parquet':
                # Single file - use ParquetFile for metadata-only access
                try:
                    file_size = path.stat().st_size / (1024 * 1024)  # MB

                    # Read metadata without loading data
                    parquet_file = pq.ParquetFile(path)
                    schema = parquet_file.schema_arrow
                    metadata = parquet_file.metadata

                    # Get row count from metadata
                    num_rows = metadata.num_rows

                    # Get columns from schema
                    columns = schema.names

                    # Try to extract date range from metadata if timestamp column exists
                    date_range = None
                    if 'timestamp' in columns or any('date' in col.lower() for col in columns):
                        # For single files, we need to peek at first/last rows for date range
                        # Read only first and last row groups
                        try:
                            first_batch = parquet_file.read_row_group(0, columns=['timestamp'] if 'timestamp' in columns else None)
                            last_batch = parquet_file.read_row_group(metadata.num_row_groups - 1, columns=['timestamp'] if 'timestamp' in columns else None)

                            df_first = first_batch.to_pandas()
                            df_last = last_batch.to_pandas()

                            if 'timestamp' in df_first.columns:
                                min_date = df_first['timestamp'].min()
                                max_date = df_last['timestamp'].max()
                                date_range = (
                                    pd.to_datetime(min_date).strftime("%Y-%m-%d"),
                                    pd.to_datetime(max_date).strftime("%Y-%m-%d")
                                )
                        except Exception:
                            pass  # Date range extraction failed, leave as None

                    datasets.append({
                        'name': path.stem,
                        'type': data_type,
                        'path': str(path),
                        'rows': num_rows,
                        'size_mb': round(file_size, 2),
                        'date_range': date_range,
                        'columns': columns
                    })
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {path}: {str(e)}")

            elif path.is_dir():
                # Partitioned dataset - use pyarrow.dataset for efficient metadata access
                try:
                    # Calculate total size of all files in directory
                    total_size = sum(
                        f.stat().st_size for f in path.rglob('*.parquet')
                    ) / (1024 * 1024)

                    # Use pyarrow dataset API for metadata
                    dataset = ds.dataset(path, format='parquet')
                    schema = dataset.schema

                    # Get columns from schema
                    columns = schema.names

                    # Count rows efficiently using dataset API
                    num_rows = dataset.count_rows()

                    # Extract date range from partition keys and sample first/last files
                    date_range = None
                    try:
                        # Get all partition directories (year/month structure)
                        partition_dirs = sorted([d for d in path.rglob('*.parquet')])

                        if partition_dirs and 'timestamp' in columns:
                            # Read only first and last parquet files for min/max dates
                            first_file = partition_dirs[0]
                            last_file = partition_dirs[-1]

                            # Read minimal data from first file
                            first_pf = pq.ParquetFile(first_file)
                            first_batch = first_pf.read_row_group(0, columns=['timestamp'])
                            min_date = first_batch.to_pandas()['timestamp'].min()

                            # Read minimal data from last file
                            last_pf = pq.ParquetFile(last_file)
                            last_rg_idx = last_pf.metadata.num_row_groups - 1
                            last_batch = last_pf.read_row_group(last_rg_idx, columns=['timestamp'])
                            max_date = last_batch.to_pandas()['timestamp'].max()

                            date_range = (
                                pd.to_datetime(min_date).strftime("%Y-%m-%d"),
                                pd.to_datetime(max_date).strftime("%Y-%m-%d")
                            )
                    except Exception as e:
                        # Date range extraction failed, leave as None
                        logger.debug(f"Could not extract date range from partitioned dataset: {e}")

                    datasets.append({
                        'name': path.name,
                        'type': data_type,
                        'path': str(path),
                        'rows': num_rows,
                        'size_mb': round(total_size, 2),
                        'date_range': date_range,
                        'columns': columns,
                        'partitioned': True
                    })
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {path}: {str(e)}")

        logger.info(f"Found {len(datasets)} datasets in {base_path}")

        return datasets

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.

        Args:
            data: DataFrame to deduplicate

        Returns:
            DataFrame with duplicates removed
        """
        original_len = len(data)

        # Remove duplicates based on index
        if isinstance(data.index, pd.DatetimeIndex):
            data = data[~data.index.duplicated(keep='first')]
        else:
            data = data.drop_duplicates()

        removed = original_len - len(data)

        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")

        return data

    def fill_missing_timestamps(
        self,
        data: pd.DataFrame,
        freq: str = 'H',
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        Fill missing timestamps in time-series data.

        Args:
            data: DataFrame with datetime index
            freq: Frequency string (e.g., 'H' for hourly, 'D' for daily)
            method: Fill method ('interpolate', 'ffill', 'bfill')

        Returns:
            DataFrame with filled timestamps
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex to fill missing timestamps")

        original_len = len(data)

        # Reindex to regular frequency
        data = data.asfreq(freq)

        # Fill missing values
        if method == 'interpolate':
            data = data.interpolate(method='time')
        elif method == 'ffill':
            data = data.fillna(method='ffill')
        elif method == 'bfill':
            data = data.fillna(method='bfill')
        else:
            raise ValueError(f"Unknown fill method: {method}")

        filled = len(data) - original_len

        if filled > 0:
            logger.info(f"Filled {filled} missing timestamps using {method}")

        return data

    def resample_frequency(
        self,
        data: pd.DataFrame,
        freq: str,
        agg_func: Union[str, Dict] = 'mean'
    ) -> pd.DataFrame:
        """
        Resample time-series data to different frequency.

        Args:
            data: DataFrame with datetime index
            freq: Target frequency (e.g., 'H', 'D', 'M')
            agg_func: Aggregation function or dict of column: function

        Returns:
            Resampled DataFrame
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex to resample")

        logger.info(f"Resampling data to frequency: {freq}")

        resampled = data.resample(freq).agg(agg_func)

        logger.info(f"Resampled from {len(data)} to {len(resampled)} rows")

        return resampled

    def _validate_data(
        self,
        data: pd.DataFrame,
        validate_time_index: bool = False,
        expected_freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Validate data before saving.

        Performs validation checks including:
        - Empty dataframe check
        - All-null columns detection
        - Duplicate indices detection
        - Timestamp gap detection (optional)
        - Numeric dtype coercion

        Args:
            data: DataFrame to validate
            validate_time_index: If True, check for gaps in time-indexed data
            expected_freq: Expected frequency for time index (e.g., 'H', 'D')

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If validation fails
        """
        if data.empty:
            raise ValueError("Cannot save empty DataFrame")

        # Check for all-null columns
        null_cols = data.columns[data.isnull().all()].tolist()
        if null_cols:
            logger.warning(f"Dropping columns with all null values: {null_cols}")
            data = data.drop(columns=null_cols)

        # Check for duplicate indices
        if data.index.duplicated().any():
            dup_count = data.index.duplicated().sum()
            logger.warning(f"Found {dup_count} duplicate indices")

        # Validate time index and check for gaps
        if isinstance(data.index, pd.DatetimeIndex):
            if validate_time_index and expected_freq:
                # Detect gaps by comparing actual vs expected date range
                expected_index = pd.date_range(
                    start=data.index.min(),
                    end=data.index.max(),
                    freq=expected_freq
                )
                actual_count = len(data)
                expected_count = len(expected_index)

                if actual_count < expected_count:
                    gap_count = expected_count - actual_count
                    gap_pct = (gap_count / expected_count) * 100
                    logger.warning(
                        f"Detected {gap_count} missing timestamps "
                        f"({gap_pct:.1f}% of expected range with freq={expected_freq})"
                    )

            # Check if index is sorted
            if not data.index.is_monotonic_increasing:
                logger.warning("DateTime index is not sorted, sorting now")
                data = data.sort_index()

        # Coerce expected numeric columns to numeric dtypes
        # Common numeric column patterns in energy data
        numeric_column_patterns = [
            'price', 'lmp', 'mcc', 'mlc', 'mce',  # Price components
            'mw', 'load', 'demand', 'generation',  # Power/energy
            'value', 'quantity', 'revenue',  # Generic numeric
            'volume', 'cost', 'total'
        ]

        for col in data.columns:
            # Check if column name matches any numeric pattern
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in numeric_column_patterns):
                # Try to convert to numeric if not already
                if data[col].dtype == 'object':
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        null_count = data[col].isnull().sum()
                        if null_count > 0:
                            logger.warning(
                                f"Column '{col}' coerced to numeric: {null_count} values became null"
                            )
                    except Exception as e:
                        logger.warning(f"Could not coerce column '{col}' to numeric: {e}")

        return data

    def _get_date_range(self, data: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """
        Get date range from DataFrame index.

        Args:
            data: DataFrame with potential datetime index

        Returns:
            Tuple of (start_date, end_date) as strings, or None
        """
        if isinstance(data.index, pd.DatetimeIndex) and not data.index.empty:
            start = data.index.min().strftime("%Y-%m-%d")
            end = data.index.max().strftime("%Y-%m-%d")
            return (start, end)
        return None

    def save_trades_to_db(
        self,
        trades: pd.DataFrame,
        table: str = 'trades'
    ) -> bool:
        """
        Save trades to database.

        Optional database persistence for trades controlled by configuration.
        Requires SQLAlchemy and database connection settings in config.

        Args:
            trades: DataFrame with trade records
            table: Database table name (default: 'trades')

        Returns:
            True if successful, False otherwise
        """
        # Import SQLAlchemy only when needed
        try:
            from sqlalchemy import create_engine, Table, Column, Float, String, DateTime, MetaData, Integer
            from sqlalchemy.dialects.postgresql import insert as pg_insert
        except ImportError:
            logger.warning(
                "SQLAlchemy not installed. Database logging skipped. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )
            return False

        try:
            # Read DB connection settings from config
            from src.config.load_config import get_config
            config = get_config()
            db_config = config.get('data', {}).get('database', {})

            if not db_config:
                logger.debug("No database configuration found, skipping DB save")
                return False

            # Get connection parameters
            url = db_config.get('url')
            if not url:
                logger.warning("Database URL not configured, skipping DB save")
                return False

            # Ensure timestamp column exists and is datetime
            if 'timestamp' not in trades.columns:
                if isinstance(trades.index, pd.DatetimeIndex):
                    trades = trades.reset_index()
                else:
                    logger.error("No timestamp column or DatetimeIndex found in trades")
                    return False

            trades = trades.copy()
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])

            # Create engine
            engine = create_engine(url)

            # Use pandas to_sql for simplicity (handles schema creation automatically)
            trades.to_sql(
                name=table,
                con=engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )

            logger.info(f"Successfully saved {len(trades)} trades to database table '{table}'")
            engine.dispose()
            return True

        except Exception as e:
            logger.error(f"Failed to save trades to database: {e}")
            return False


if __name__ == "__main__":
    # Setup logging first
    from src.config.load_config import setup_logging
    setup_logging()

    # Example usage
    print("DataManager Example")
    print("=" * 50)

    # Initialize manager (will load config automatically)
    manager = DataManager()

    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'price': [100 + i * 0.1 for i in range(len(dates))],
        'volume': [1000 + i * 10 for i in range(len(dates))]
    }, index=dates)

    print(f"\nSample data shape: {sample_data.shape}")
    print(sample_data.head())

    # Save raw data
    print("\nSaving raw data...")
    path = manager.save_raw_data(
        data=sample_data,
        source="example",
        dataset="test",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    print(f"Saved to: {path}")

    # List available datasets
    print("\nAvailable datasets:")
    datasets = manager.get_available_datasets(data_type="raw")
    for ds in datasets:
        print(f"  - {ds['name']}: {ds['rows']} rows, {ds['size_mb']} MB")

    # Load data back
    print("\nLoading data...")
    loaded_data = manager.load_data(
        source="example",
        dataset="test",
        data_type="raw",
        start_date="2023-06-01",
        end_date="2023-06-30"
    )
    print(f"Loaded data shape: {loaded_data.shape}")
    print(loaded_data.head())
