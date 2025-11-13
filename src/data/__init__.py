"""
Data fetching and management module.

This module provides:
- API data fetchers (EIA, CAISO)
- Data storage and retrieval (DataManager)
- Synthetic data generation (prices, wind, solar)
- Data utilities (blending, correlation, validation)
"""

# Data fetchers
from src.data.data_fetcher import EIAFetcher, CAISOFetcher, APIError

# Data management
from src.data.data_manager import DataManager

# Synthetic data generation
from src.data.synthetic_generator import SyntheticPriceGenerator
from src.data.renewable_generator import WindGenerator, SolarGenerator

# Data utilities
from src.data.data_utils import (
    blend_real_and_synthetic,
    align_timestamps,
    apply_renewable_price_correlation,
    validate_data_consistency,
    generate_correlated_scenarios,
    create_stress_test_suite,
    resample_to_frequency,
    calculate_net_demand,
    export_for_modeling,
    add_scenario_metadata
)

__all__ = [
    # Data fetchers
    'EIAFetcher',
    'CAISOFetcher',
    'APIError',

    # Data management
    'DataManager',

    # Synthetic data generation
    'SyntheticPriceGenerator',
    'WindGenerator',
    'SolarGenerator',

    # Data utilities
    'blend_real_and_synthetic',
    'align_timestamps',
    'apply_renewable_price_correlation',
    'validate_data_consistency',
    'generate_correlated_scenarios',
    'create_stress_test_suite',
    'resample_to_frequency',
    'calculate_net_demand',
    'export_for_modeling',
    'add_scenario_metadata'
]
