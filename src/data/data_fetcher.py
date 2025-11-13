"""
Data fetching module for EIA API v2 and CAISO OASIS API.

This module provides classes to fetch electricity market data from:
- EIA (Energy Information Administration) API v2
- CAISO (California Independent System Operator) OASIS API
"""

import os
import io
import time
import zipfile
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Get logger (no basicConfig - central config handles logging)
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom exception for API errors."""
    pass


class EIAFetcher:
    """
    Fetcher for EIA API v2 electricity market data.

    The EIA API provides access to electricity retail sales, RTO data,
    generation capacity, and pricing data. Requires a free API key
    from https://www.eia.gov/opendata/register.php

    Example:
        >>> fetcher = EIAFetcher(api_key="your_api_key")
        >>> df = fetcher.fetch_electricity_data(
        ...     endpoint="retail-sales",
        ...     frequency="monthly",
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     state="CA"
        ... )
    """

    MAX_ROWS_PER_REQUEST = 5000
    DEFAULT_BASE_URL = "https://api.eia.gov/v2/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize EIA API fetcher.

        Args:
            api_key: EIA API key. If None, reads from config or EIA_API_KEY environment variable.
            timeout: Request timeout in seconds. If None, reads from config.
            max_retries: Maximum retry attempts. If None, reads from config.
            retry_backoff: Exponential backoff multiplier. If None, reads from config.
            config: Configuration dictionary. If None, loads from config.yaml.

        Raises:
            ValueError: If no API key is provided or found in environment/config.
        """
        # Load config if not provided
        if config is None:
            from src.config.load_config import get_config
            config = get_config()

        # Get API settings from config
        api_config = config.get("api", {}).get("eia", {})

        # Get API key: priority is parameter > environment variable > config
        self.api_key = api_key or os.getenv('EIA_API_KEY') or api_config.get("api_key")
        if not self.api_key:
            raise ValueError(
                "EIA API key is required. Set EIA_API_KEY environment variable, "
                "configure api.eia.api_key in config.yaml, or pass api_key parameter."
            )

        # Get base URL from config with fallback to default
        self.base_url = api_config.get("base_url", self.DEFAULT_BASE_URL)

        # Get timeout/retry settings from config or use provided/default values
        self.timeout = timeout if timeout is not None else api_config.get("timeout", 120)
        self.max_retries = max_retries if max_retries is not None else api_config.get("max_retries", 3)
        self.retry_backoff = retry_backoff if retry_backoff is not None else api_config.get("retry_backoff", 2.0)

        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def fetch_electricity_data(
        self,
        endpoint: str = "retail-sales",
        frequency: str = "monthly",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        state: Optional[str] = None,
        sector: Optional[str] = None,
        facets: Optional[Dict[str, Any]] = None,
        data_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch electricity market data from EIA API v2.

        Args:
            endpoint: Data endpoint (e.g., "retail-sales", "rto", "operating-generator-capacity")
            frequency: Data frequency ("hourly", "daily", "monthly", "annual")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            state: State code filter (e.g., "CA", "TX")
            sector: Sector filter (e.g., "RES", "COM", "IND", "TRA")
            facets: Additional facet filters as dictionary
            data_columns: Specific data columns to retrieve

        Returns:
            DataFrame with timestamp index and requested data columns.

        Raises:
            APIError: If API request fails or returns error.
        """
        logger.info(
            f"Fetching EIA data: endpoint={endpoint}, frequency={frequency}, "
            f"start={start_date}, end={end_date}, state={state}, sector={sector}"
        )

        # Validate date range
        if start_date:
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start_date must be in YYYY-MM-DD format")

        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("end_date must be in YYYY-MM-DD format")

        # Build URL
        url = f"{self.base_url}electricity/{endpoint}/data/"

        # Build query parameters as list of tuples to support multi-value facets
        params: List[Tuple[str, Any]] = [
            ("api_key", self.api_key),
            ("frequency", frequency),
            ("data[]", "value"),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("offset", 0),
            ("length", self.MAX_ROWS_PER_REQUEST)
        ]

        # Add optional filters
        if start_date:
            params.append(("start", start_date))
        if end_date:
            params.append(("end", end_date))
        if state:
            # Support both single string and list of states
            if isinstance(state, list):
                for s in state:
                    params.append(("facets[stateid][]", s))
            else:
                params.append(("facets[stateid][]", state))
        if sector:
            # Support both single string and list of sectors
            if isinstance(sector, list):
                for s in sector:
                    params.append(("facets[sectorid][]", s))
            else:
                params.append(("facets[sectorid][]", sector))

        # Add custom facets
        if facets:
            for key, value in facets.items():
                if isinstance(value, list):
                    # Add each value as a separate parameter with the same key
                    for v in value:
                        params.append((f"facets[{key}][]", v))
                else:
                    params.append((f"facets[{key}][]", value))

        # Add custom data columns
        if data_columns:
            if isinstance(data_columns, list):
                for col in data_columns:
                    params.append(("data[]", col))
            else:
                params.append(("data[]", data_columns))

        # Fetch data with pagination
        all_data = []
        offset = 0

        while True:
            # Update offset in params list (find and replace the offset tuple)
            params_with_offset = [(k, v) for k, v in params if k != "offset"]
            params_with_offset.append(("offset", offset))

            try:
                response = self.session.get(url, params=params_with_offset, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()

                # Check for API errors
                if "error" in data:
                    raise APIError(f"EIA API error: {data['error']}")

                # Extract data rows
                if "response" in data and "data" in data["response"]:
                    rows = data["response"]["data"]
                    if not rows:
                        break

                    all_data.extend(rows)

                    # Check if more data available
                    total = data["response"].get("total", 0)
                    if offset + len(rows) >= total:
                        break

                    offset += self.MAX_ROWS_PER_REQUEST

                    logger.info(f"Fetched {len(all_data)} / {total} rows")

                    # Rate limiting courtesy delay
                    time.sleep(0.5)
                else:
                    break

            except requests.exceptions.RequestException as e:
                raise APIError(f"Failed to fetch EIA data: {str(e)}")

        if not all_data:
            logger.warning("No data returned from EIA API")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Convert period to datetime
        if "period" in df.columns:
            df["timestamp"] = pd.to_datetime(df["period"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

        # Convert string values to numeric
        numeric_columns = ["value", "price", "quantity", "revenue"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Successfully fetched {len(df)} rows of EIA data")

        return df

    def close(self):
        """Close the session."""
        self.session.close()


class CAISOFetcher:
    """
    Fetcher for CAISO OASIS API data.

    The CAISO OASIS API provides access to California ISO market data including
    locational marginal prices (LMPs), demand forecasts, and generation data.

    Example:
        >>> fetcher = CAISOFetcher()
        >>> df = fetcher.fetch_lmp_data(
        ...     market_run_id="DAM",
        ...     nodes=["TH_NP15_GEN-APND"],
        ...     start_date="20230101T00:00-0000",
        ...     end_date="20230131T23:59-0000"
        ... )
    """

    DEFAULT_BASE_URL = "https://oasis.caiso.com/oasisapi/SingleZip"

    def __init__(
        self,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CAISO OASIS API fetcher.

        Args:
            timeout: Request timeout in seconds. If None, reads from config.
            max_retries: Maximum retry attempts. If None, reads from config.
            retry_backoff: Exponential backoff multiplier. If None, reads from config.
            config: Configuration dictionary. If None, loads from config.yaml.
        """
        # Load config if not provided
        if config is None:
            from src.config.load_config import get_config
            config = get_config()

        # Get API settings from config
        api_config = config.get("api", {}).get("caiso", {})

        # Get base URL from config with fallback to default
        self.base_url = api_config.get("base_url", self.DEFAULT_BASE_URL)

        # Get timeout/retry settings from config or use provided/default values
        self.timeout = timeout if timeout is not None else api_config.get("timeout", 120)
        self.max_retries = max_retries if max_retries is not None else api_config.get("max_retries", 3)
        self.retry_backoff = retry_backoff if retry_backoff is not None else api_config.get("retry_backoff", 2.0)

        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _format_caiso_datetime(self, dt: str) -> str:
        """
        Validate and format datetime for CAISO API.

        Args:
            dt: Datetime string in format YYYYMMDDTHH:MM-0000

        Returns:
            Validated datetime string.

        Raises:
            ValueError: If datetime format is invalid.
        """
        import re

        # Validate format using regex: YYYYMMDDTHH:MM-HHMM or +HHMM
        pattern = r'^\d{8}T\d{2}:\d{2}[-+]\d{4}$'
        if not re.match(pattern, dt):
            raise ValueError(
                f"Invalid CAISO datetime format: {dt}. "
                "Expected format: YYYYMMDDTHH:MM-HHMM (e.g., 20230101T00:00-0800 or 20230101T00:00-0000)"
            )

        # Validate date/time portion (first 16 characters: YYYYMMDDTHH:MM)
        try:
            datetime.strptime(dt[:16], "%Y%m%dT%H:%M")
        except ValueError:
            raise ValueError(
                f"Invalid CAISO datetime format: {dt}. "
                "Expected format: YYYYMMDDTHH:MM-HHMM (date/time portion invalid)"
            )

        return dt

    def fetch_lmp_data(
        self,
        market_run_id: str = "DAM",
        nodes: Optional[List[str]] = None,
        start_date: str = None,
        end_date: str = None,
        query_name: str = "PRC_LMP"
    ) -> pd.DataFrame:
        """
        Fetch Locational Marginal Price (LMP) data from CAISO.

        Args:
            market_run_id: Market run type ("DAM" for day-ahead, "RTM" for real-time)
            nodes: List of pricing node IDs (e.g., ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND"])
            start_date: Start datetime in format YYYYMMDDTmm:HH-0000
            end_date: End datetime in format YYYYMMDDTmm:HH-0000
            query_name: Query name ("PRC_LMP" for day-ahead, "PRC_INTVL_LMP" for real-time 5-min)

        Returns:
            DataFrame with LMP data including timestamp, node, and price components.

        Raises:
            APIError: If API request fails.
        """
        logger.info(
            f"Fetching CAISO LMP data: market={market_run_id}, query={query_name}, "
            f"start={start_date}, end={end_date}, nodes={nodes}"
        )

        # Validate query_name and market_run_id consistency
        if query_name == "PRC_LMP" and market_run_id != "DAM":
            raise ValueError(
                f"query_name 'PRC_LMP' requires market_run_id='DAM', got '{market_run_id}'. "
                "Use 'PRC_INTVL_LMP' for real-time market (RTM)."
            )
        elif query_name == "PRC_INTVL_LMP" and market_run_id != "RTM":
            raise ValueError(
                f"query_name 'PRC_INTVL_LMP' requires market_run_id='RTM', got '{market_run_id}'. "
                "Use 'PRC_LMP' for day-ahead market (DAM)."
            )

        # Validate dates
        if start_date:
            start_date = self._format_caiso_datetime(start_date)
        if end_date:
            end_date = self._format_caiso_datetime(end_date)

        # Validate date order
        if start_date and end_date:
            start_dt = datetime.strptime(start_date[:16], "%Y%m%dT%H:%M")
            end_dt = datetime.strptime(end_date[:16], "%Y%m%dT%H:%M")
            if start_dt > end_dt:
                raise ValueError(
                    f"start_date ({start_date}) must be before or equal to end_date ({end_date})"
                )

        # Build query parameters as list of tuples to support multiple nodes
        params: List[Tuple[str, Any]] = [
            ("queryname", query_name),
            ("market_run_id", market_run_id),
            ("startdatetime", start_date),
            ("enddatetime", end_date),
            ("version", "1"),
            ("resultformat", "6")  # CSV format
        ]

        # Add nodes if specified (each node as a separate parameter)
        if nodes:
            for node in nodes:
                params.append(("node", node))

        try:
            # Fetch data (returns ZIP file)
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Extract CSV from ZIP
            zip_data = io.BytesIO(response.content)

            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                # Get first CSV file from ZIP
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]

                if not csv_files:
                    raise APIError("No CSV file found in CAISO response ZIP")

                # Read CSV
                with zip_ref.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file)

            # Convert timestamp columns to datetime
            timestamp_cols = ["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "OPR_DATE"]
            for col in timestamp_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            # Convert numeric columns
            numeric_cols = ["MW", "LMP", "MCC", "MLC", "MCE"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info(f"Successfully fetched {len(df)} rows of CAISO LMP data")

            return df

        except requests.exceptions.RequestException as e:
            raise APIError(f"Failed to fetch CAISO LMP data: {str(e)}")
        except zipfile.BadZipFile as e:
            raise APIError(f"Invalid ZIP response from CAISO API: {str(e)}")

    def fetch_demand_forecast(
        self,
        start_date: str = None,
        end_date: str = None,
        query_name: str = "SLD_FCST"
    ) -> pd.DataFrame:
        """
        Fetch demand forecast data from CAISO.

        Args:
            start_date: Start datetime in format YYYYMMDDTmm:HH-0000
            end_date: End datetime in format YYYYMMDDTmm:HH-0000
            query_name: Query name (default: "SLD_FCST" for system load forecast)

        Returns:
            DataFrame with demand forecast data.

        Raises:
            APIError: If API request fails.
        """
        logger.info(
            f"Fetching CAISO demand forecast: query={query_name}, "
            f"start={start_date}, end={end_date}"
        )

        # Validate dates
        if start_date:
            start_date = self._format_caiso_datetime(start_date)
        if end_date:
            end_date = self._format_caiso_datetime(end_date)

        # Validate date order
        if start_date and end_date:
            start_dt = datetime.strptime(start_date[:16], "%Y%m%dT%H:%M")
            end_dt = datetime.strptime(end_date[:16], "%Y%m%dT%H:%M")
            if start_dt > end_dt:
                raise ValueError(
                    f"start_date ({start_date}) must be before or equal to end_date ({end_date})"
                )

        # Build query parameters
        params = {
            "queryname": query_name,
            "startdatetime": start_date,
            "enddatetime": end_date,
            "version": "1",
            "resultformat": "6"  # CSV format
        }

        try:
            # Fetch data (returns ZIP file)
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Extract CSV from ZIP
            zip_data = io.BytesIO(response.content)

            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                # Get first CSV file from ZIP
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]

                if not csv_files:
                    raise APIError("No CSV file found in CAISO response ZIP")

                # Read CSV
                with zip_ref.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file)

            # Convert timestamp columns to datetime
            timestamp_cols = ["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "OPR_DATE"]
            for col in timestamp_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            # Convert numeric columns
            numeric_cols = ["MW", "LOAD"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info(f"Successfully fetched {len(df)} rows of CAISO demand forecast data")

            return df

        except requests.exceptions.RequestException as e:
            raise APIError(f"Failed to fetch CAISO demand forecast: {str(e)}")
        except zipfile.BadZipFile as e:
            raise APIError(f"Invalid ZIP response from CAISO API: {str(e)}")

    def close(self):
        """Close the session."""
        self.session.close()


if __name__ == "__main__":
    # Setup logging first
    from src.config.load_config import setup_logging
    setup_logging()

    # Example usage
    print("EIA API Fetcher Example")
    print("-" * 50)

    try:
        # Initialize EIA fetcher
        eia = EIAFetcher()

        # Fetch sample data (last 3 months of California retail sales)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)

        df_eia = eia.fetch_electricity_data(
            endpoint="retail-sales",
            frequency="monthly",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            state="CA"
        )

        print(f"\nFetched {len(df_eia)} rows from EIA API")
        print(df_eia.head())

        eia.close()

    except Exception as e:
        print(f"EIA API Error: {e}")

    print("\n" + "=" * 50)
    print("CAISO OASIS API Fetcher Example")
    print("-" * 50)

    try:
        # Initialize CAISO fetcher
        caiso = CAISOFetcher()

        # Fetch sample LMP data (last 7 days day-ahead market)
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=7)

        df_caiso = caiso.fetch_lmp_data(
            market_run_id="DAM",
            nodes=["TH_NP15_GEN-APND"],
            start_date=start_dt.strftime("%Y%m%dT00:00-0000"),
            end_date=end_dt.strftime("%Y%m%dT23:59-0000")
        )

        print(f"\nFetched {len(df_caiso)} rows from CAISO API")
        print(df_caiso.head())

        caiso.close()

    except Exception as e:
        print(f"CAISO API Error: {e}")
