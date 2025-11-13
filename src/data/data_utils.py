"""
Data utilities module for blending real and synthetic data.

This module provides utility functions for:
- Blending real and synthetic datasets
- Timestamp alignment
- Implementing correlations between renewable generation and prices
- Data quality validation
- Scenario generation for Monte Carlo analysis
- Stress testing
"""

import logging
from typing import Optional, Dict, List, Tuple, Union, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import linalg


# Get logger (no basicConfig - central config handles logging)
logger = logging.getLogger(__name__)


def blend_real_and_synthetic(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    blend_strategy: str = 'concatenate',
    add_metadata: bool = True,
    preserve_scenarios: bool = False,
    freq: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge real and synthetic datasets seamlessly.

    Args:
        real_data: DataFrame from DataManager or API fetchers
        synthetic_data: DataFrame from synthetic generators
        blend_strategy: One of:
            - 'concatenate': Append synthetic after real
            - 'fill_gaps': Use synthetic only for missing timestamps
            - 'replace': Replace real with synthetic for specified period
        add_metadata: Add 'data_source' column with 'real' or 'synthetic' tags
        preserve_scenarios: If True and either dataset has 'scenario_id' column,
            creates MultiIndex ['scenario_id', timestamp] to preserve all scenarios.
            Real data without scenario_id gets scenario_id=0. Default False.
        freq: Frequency string for 'fill_gaps' strategy (e.g., 'H', 'D').
            If None, will attempt to infer. Required if inference fails.

    Returns:
        Blended DataFrame with consistent schema

    Raises:
        ValueError: If blend_strategy is invalid, data is incompatible, or
            freq cannot be inferred for 'fill_gaps' strategy
    """
    valid_strategies = ['concatenate', 'fill_gaps', 'replace']
    if blend_strategy not in valid_strategies:
        raise ValueError(
            f"blend_strategy must be one of {valid_strategies}, got '{blend_strategy}'"
        )

    if real_data.empty and synthetic_data.empty:
        raise ValueError("Both real_data and synthetic_data are empty")

    logger.info(
        f"Blending data: strategy={blend_strategy}, "
        f"real_rows={len(real_data)}, synthetic_rows={len(synthetic_data)}"
    )

    # Normalize column names
    real_data = _normalize_column_names(real_data.copy())
    synthetic_data = _normalize_column_names(synthetic_data.copy())

    # Check schema compatibility
    issues = _check_schema_compatibility(real_data, synthetic_data)
    if issues:
        logger.warning(f"Schema compatibility issues: {issues}")

    # Handle multi-scenario preservation
    has_scenarios = False
    if preserve_scenarios:
        has_real_scenarios = 'scenario_id' in real_data.columns
        has_synthetic_scenarios = 'scenario_id' in synthetic_data.columns

        if has_real_scenarios or has_synthetic_scenarios:
            has_scenarios = True
            logger.info("Preserving scenarios with MultiIndex")

            # Add scenario_id to real data if missing
            if not has_real_scenarios:
                real_data['scenario_id'] = 0

            # Add scenario_id to synthetic data if missing (shouldn't happen, but be defensive)
            if not has_synthetic_scenarios:
                synthetic_data['scenario_id'] = 0

            # Create MultiIndex for both DataFrames
            real_data = real_data.set_index('scenario_id', append=True)
            real_data = real_data.reorder_levels(['scenario_id', real_data.index.names[0]])

            synthetic_data = synthetic_data.set_index('scenario_id', append=True)
            synthetic_data = synthetic_data.reorder_levels(['scenario_id', synthetic_data.index.names[0]])

    # Add metadata tags if requested
    if add_metadata:
        real_data['data_source'] = 'real'
        synthetic_data['data_source'] = 'synthetic'

    # Apply blending strategy
    if blend_strategy == 'concatenate':
        # Simple concatenation
        blended = pd.concat([real_data, synthetic_data], ignore_index=False)
        blended = blended.sort_index()

    elif blend_strategy == 'fill_gaps':
        # Use synthetic only where real data is missing
        # Create a full date range
        if isinstance(real_data.index, pd.DatetimeIndex) and isinstance(synthetic_data.index, pd.DatetimeIndex):
            # Determine frequency
            inferred_freq = freq
            if inferred_freq is None:
                # Try to infer frequency from real data
                inferred_freq = pd.infer_freq(real_data.index)
                if inferred_freq is None:
                    # Try to infer from synthetic data
                    inferred_freq = pd.infer_freq(synthetic_data.index)

            # Raise error if frequency cannot be determined
            if inferred_freq is None:
                raise ValueError(
                    "Cannot infer frequency for 'fill_gaps' strategy. "
                    "Please provide 'freq' parameter (e.g., 'H' for hourly, 'D' for daily). "
                    "Example: blend_real_and_synthetic(..., blend_strategy='fill_gaps', freq='H')"
                )

            # Create complete index
            start_date = min(real_data.index.min(), synthetic_data.index.min())
            end_date = max(real_data.index.max(), synthetic_data.index.max())
            full_index = pd.date_range(start=start_date, end=end_date, freq=inferred_freq)

            # Reindex both datasets
            real_reindexed = real_data.reindex(full_index)
            synthetic_reindexed = synthetic_data.reindex(full_index)

            # Fill gaps in real data with synthetic data
            blended = real_reindexed.combine_first(synthetic_reindexed)
        else:
            logger.warning("Data does not have DatetimeIndex, using simple concatenation")
            blended = pd.concat([real_data, synthetic_data], ignore_index=False).sort_index()

    elif blend_strategy == 'replace':
        # Replace overlapping real data with synthetic (synthetic takes precedence)
        blended = real_data.copy()

        # For overlapping indices, use synthetic data
        synthetic_indices = synthetic_data.index
        overlap_mask = blended.index.isin(synthetic_indices)

        if overlap_mask.any():
            # Remove overlapping real data
            blended = blended[~overlap_mask]

        # Concatenate with synthetic
        blended = pd.concat([blended, synthetic_data], ignore_index=False).sort_index()

    # Remove duplicate timestamps (only if not preserving scenarios)
    if not has_scenarios:
        if not blended.index.is_unique:
            logger.warning(f"Removing {blended.index.duplicated().sum()} duplicate timestamps")
            blended = blended[~blended.index.duplicated(keep='first')]

    # Calculate blending statistics
    if add_metadata and 'data_source' in blended.columns:
        n_real = (blended['data_source'] == 'real').sum()
        n_synthetic = (blended['data_source'] == 'synthetic').sum()
        pct_real = n_real / len(blended) * 100
        pct_synthetic = n_synthetic / len(blended) * 100

        if has_scenarios:
            # Report per-scenario statistics
            if isinstance(blended.index, pd.MultiIndex):
                scenario_ids = blended.index.get_level_values(0).unique()
                logger.info(
                    f"Blended {len(blended)} rows across {len(scenario_ids)} scenario(s): "
                    f"{pct_real:.1f}% real, {pct_synthetic:.1f}% synthetic"
                )
                # Log per-scenario row counts
                for scenario_id in scenario_ids:
                    scenario_mask = blended.index.get_level_values(0) == scenario_id
                    n_scenario_rows = scenario_mask.sum()
                    n_scenario_real = ((blended['data_source'] == 'real') & scenario_mask).sum()
                    n_scenario_synthetic = ((blended['data_source'] == 'synthetic') & scenario_mask).sum()
                    logger.info(
                        f"  Scenario {scenario_id}: {n_scenario_rows} rows "
                        f"({n_scenario_real} real, {n_scenario_synthetic} synthetic)"
                    )
            else:
                logger.info(
                    f"Blended {len(blended)} rows: {pct_real:.1f}% real, {pct_synthetic:.1f}% synthetic"
                )
        else:
            logger.info(
                f"Blended {len(blended)} rows: {pct_real:.1f}% real, {pct_synthetic:.1f}% synthetic"
            )

    return blended


def align_timestamps(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: str = 'inner',
    freq: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two DataFrames to common timestamp index.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        method: Alignment method ('inner', 'outer', 'left', 'right')
        freq: Target frequency for resampling (e.g., 'H', 'D'). If None, no resampling

    Returns:
        Tuple of (aligned_df1, aligned_df2)

    Raises:
        ValueError: If DataFrames don't have DatetimeIndex
    """
    if not isinstance(df1.index, pd.DatetimeIndex):
        raise ValueError("df1 must have DatetimeIndex")
    if not isinstance(df2.index, pd.DatetimeIndex):
        raise ValueError("df2 must have DatetimeIndex")

    # Resample if frequency specified
    if freq is not None:
        df1 = _resample_with_direction(df1, freq)
        df2 = _resample_with_direction(df2, freq)

    # Align based on method
    if method == 'inner':
        # Only keep common timestamps
        common_index = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_index]
        df2_aligned = df2.loc[common_index]

    elif method == 'outer':
        # Keep all timestamps from both
        all_index = df1.index.union(df2.index)
        df1_aligned = df1.reindex(all_index)
        df2_aligned = df2.reindex(all_index)

    elif method == 'left':
        # Keep all timestamps from df1
        df1_aligned = df1
        df2_aligned = df2.reindex(df1.index)

    elif method == 'right':
        # Keep all timestamps from df2
        df1_aligned = df1.reindex(df2.index)
        df2_aligned = df2

    else:
        raise ValueError(f"method must be one of ['inner', 'outer', 'left', 'right'], got '{method}'")

    logger.info(
        f"Aligned timestamps: method={method}, "
        f"df1: {len(df1)} → {len(df1_aligned)}, df2: {len(df2)} → {len(df2_aligned)}"
    )

    return df1_aligned, df2_aligned


def apply_renewable_price_correlation(
    price_series: pd.DataFrame,
    generation_series: pd.DataFrame,
    correlation_strength: float = -0.6,
    method: str = 'linear',
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> pd.DataFrame:
    """
    Implement negative correlation between renewable generation and prices.

    High renewable generation typically leads to lower prices (merit order effect).

    Args:
        price_series: DataFrame with 'price' column
        generation_series: DataFrame with 'generation_mw' or 'capacity_factor' column
        correlation_strength: Correlation coefficient (-1.0 to 0.0, default -0.6)
        method: 'linear' (simple adjustment) or 'supply_curve' (supply-demand model)
        min_price: Minimum allowed price ($/MWh). Defaults to 1.0 if None.
        max_price: Maximum allowed price ($/MWh). Defaults to 500.0 if None.

    Returns:
        Adjusted price_series DataFrame

    Raises:
        ValueError: If required columns missing or correlation_strength invalid
    """
    if 'price' not in price_series.columns:
        raise ValueError("price_series must have 'price' column")

    gen_col = None
    if 'generation_mw' in generation_series.columns:
        gen_col = 'generation_mw'
    elif 'capacity_factor' in generation_series.columns:
        gen_col = 'capacity_factor'
    else:
        raise ValueError("generation_series must have 'generation_mw' or 'capacity_factor' column")

    if not -1.0 <= correlation_strength <= 0.0:
        raise ValueError(f"correlation_strength must be between -1.0 and 0.0, got {correlation_strength}")

    logger.info(
        f"Applying renewable-price correlation: strength={correlation_strength}, method={method}"
    )

    # Align timestamps
    price_aligned, gen_aligned = align_timestamps(price_series, generation_series, method='inner')

    adjusted_prices = price_aligned.copy()
    prices = adjusted_prices['price'].values
    generation = gen_aligned[gen_col].values

    if method == 'linear':
        # Linear adjustment: price_adjusted = price_base * (1 + correlation * normalized_generation)
        mean_gen = generation.mean()
        std_gen = generation.std()

        if std_gen > 0:
            normalized_gen = (generation - mean_gen) / std_gen
            adjustment_factor = 1.0 + correlation_strength * normalized_gen
            # Ensure adjustment factor is reasonable (between 0.5 and 2.0)
            adjustment_factor = np.clip(adjustment_factor, 0.5, 2.0)
            adjusted_prices['price'] = prices * adjustment_factor
        else:
            logger.warning("Generation has zero variance, no correlation applied")

    elif method == 'supply_curve':
        # Supply curve method: model price as exponential function of net demand
        # P = P_base * exp(-α * generation/capacity)
        # This creates more realistic price suppression at high renewable penetration

        max_gen = generation.max()
        if max_gen > 0:
            normalized_gen = generation / max_gen
            # Alpha controls the strength of the effect
            alpha = -correlation_strength * 2.0  # Scale to reasonable range
            price_multiplier = np.exp(-alpha * normalized_gen)
            adjusted_prices['price'] = prices * price_multiplier
        else:
            logger.warning("Maximum generation is zero, no correlation applied")

    else:
        raise ValueError(f"method must be 'linear' or 'supply_curve', got '{method}'")

    # Apply price bounds
    if min_price is None:
        min_price = 1.0
    if max_price is None:
        max_price = 500.0

    # Count clipped values before clamping
    n_below_min = np.sum(adjusted_prices['price'] < min_price)
    n_above_max = np.sum(adjusted_prices['price'] > max_price)

    # Clamp prices to bounds
    adjusted_prices['price'] = np.clip(adjusted_prices['price'], min_price, max_price)

    # Log clipping if it occurred
    if n_below_min > 0:
        logger.warning(
            f"Clipped {n_below_min} prices below minimum (${min_price:.2f}/MWh)"
        )
    if n_above_max > 0:
        logger.warning(
            f"Clipped {n_above_max} prices above maximum (${max_price:.2f}/MWh)"
        )

    # Log correlation statistics
    original_mean = prices.mean()
    adjusted_mean = adjusted_prices['price'].mean()
    logger.info(
        f"Price adjustment: original_mean=${original_mean:.2f}/MWh, "
        f"adjusted_mean=${adjusted_mean:.2f}/MWh, "
        f"bounds=[${min_price:.2f}, ${max_price:.2f}]"
    )

    return adjusted_prices


def validate_data_consistency(
    data: pd.DataFrame,
    data_type: str = 'price'
) -> Tuple[bool, List[str]]:
    """
    Check data quality and consistency.

    Args:
        data: DataFrame to validate
        data_type: Type of data ('price', 'generation', 'demand')

    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []

    # Check if empty
    if data.empty:
        issues.append("DataFrame is empty")
        return False, issues

    # Check for datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        issues.append("Index is not DatetimeIndex")

    # Check for monotonic timestamp index
    if isinstance(data.index, pd.DatetimeIndex):
        if not data.index.is_monotonic_increasing:
            issues.append("Timestamp index is not monotonically increasing")

    # Data-type specific checks
    if data_type == 'price':
        if 'price' in data.columns:
            prices = data['price']
            # Check for negative values
            if (prices < 0).any():
                n_negative = (prices < 0).sum()
                issues.append(f"Found {n_negative} negative price values")

            # Check for unreasonable values (> $500/MWh is very high)
            if (prices > 500).any():
                n_high = (prices > 500).sum()
                issues.append(f"Found {n_high} prices above $500/MWh")

    elif data_type == 'generation':
        gen_cols = [col for col in data.columns if 'generation' in col.lower() or 'mw' in col.lower()]
        for col in gen_cols:
            # Check for negative values
            if (data[col] < 0).any():
                n_negative = (data[col] < 0).sum()
                issues.append(f"Found {n_negative} negative generation values in {col}")

    elif data_type == 'demand':
        demand_cols = [col for col in data.columns if 'demand' in col.lower() or 'load' in col.lower()]
        for col in demand_cols:
            # Check for negative values
            if (data[col] < 0).any():
                n_negative = (data[col] < 0).sum()
                issues.append(f"Found {n_negative} negative demand values in {col}")

    # Check for excessive gaps in timestamps
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        freq = pd.infer_freq(data.index)
        if freq:
            expected_length = len(pd.date_range(data.index.min(), data.index.max(), freq=freq))
            missing_pct = (expected_length - len(data)) / expected_length * 100
            if missing_pct > 10:
                issues.append(f"Missing {missing_pct:.1f}% of expected timestamps")

    # Check for numeric dtypes in value columns
    numeric_patterns = ['price', 'generation', 'mw', 'capacity', 'demand', 'load']
    for col in data.columns:
        if any(pattern in col.lower() for pattern in numeric_patterns):
            if not np.issubdtype(data[col].dtype, np.number):
                issues.append(f"Column '{col}' is not numeric (dtype: {data[col].dtype})")

    # Determine if valid
    is_valid = len(issues) == 0

    if is_valid:
        logger.info(f"Data validation passed for {data_type} data")
    else:
        logger.warning(f"Data validation found {len(issues)} issues for {data_type} data")

    return is_valid, issues


def generate_correlated_scenarios(
    base_price_series: pd.DataFrame,
    base_generation_series: pd.DataFrame,
    n_scenarios: int = 100,
    correlation_matrix: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
    volatility_multiplier: float = 1.0,
    preserve_mean: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple correlated scenarios for Monte Carlo analysis.

    Uses Cholesky decomposition to generate correlated residuals, then reconstructs
    time series preserving base patterns (mean, seasonality).

    Args:
        base_price_series: Base price DataFrame
        base_generation_series: Base generation DataFrame
        n_scenarios: Number of scenarios to generate
        correlation_matrix: 2x2 correlation matrix. If None, uses [[1, -0.6], [-0.6, 1]]
        random_seed: Random seed for reproducibility
        volatility_multiplier: Multiplier for residual volatility (default 1.0)
        preserve_mean: If True, scenarios have same mean as base series

    Returns:
        Dictionary with keys 'prices' and 'generation', each containing DataFrame with scenario columns
    """
    if correlation_matrix is None:
        # Default: moderate negative correlation between prices and generation
        correlation_matrix = np.array([[1.0, -0.6], [-0.6, 1.0]])

    # Validate correlation matrix
    if correlation_matrix.shape != (2, 2):
        raise ValueError("correlation_matrix must be 2x2")

    if not np.allclose(correlation_matrix, correlation_matrix.T):
        raise ValueError("correlation_matrix must be symmetric")

    logger.info(f"Generating {n_scenarios} correlated scenarios with volatility_multiplier={volatility_multiplier}")

    rng = np.random.default_rng(random_seed)

    # Align timestamps
    price_aligned, gen_aligned = align_timestamps(base_price_series, base_generation_series, method='inner')

    n_steps = len(price_aligned)

    # Extract values
    base_prices = price_aligned['price'].values if 'price' in price_aligned.columns else price_aligned.iloc[:, 0].values
    base_gen = gen_aligned['generation_mw'].values if 'generation_mw' in gen_aligned.columns else gen_aligned.iloc[:, 0].values

    # Compute residuals/deviations from mean
    # For prices: use log returns to preserve multiplicative structure
    price_mean = base_prices.mean()
    price_log_returns = np.diff(np.log(base_prices + 1e-10))  # Add small constant for numerical stability
    price_log_return_std = price_log_returns.std()

    # For generation: use additive residuals (de-meaned)
    gen_mean = base_gen.mean()
    gen_residuals = base_gen - gen_mean
    gen_residual_std = gen_residuals.std()

    # Guard against zero std
    if price_log_return_std < 1e-10:
        price_log_return_std = 0.01
    if gen_residual_std < 1e-10:
        gen_residual_std = 1.0

    # Cholesky decomposition of correlation matrix
    try:
        L = linalg.cholesky(correlation_matrix, lower=True)
    except linalg.LinAlgError:
        logger.error("Correlation matrix is not positive definite")
        raise ValueError("Invalid correlation matrix")

    # Generate scenarios
    price_scenarios = np.zeros((n_steps, n_scenarios))
    gen_scenarios = np.zeros((n_steps, n_scenarios))

    for i in range(n_scenarios):
        # Generate independent standard normal innovations
        innovations = rng.standard_normal((2, n_steps))

        # Apply Cholesky to get correlated innovations
        correlated_innovations = L @ innovations

        # Scale by residual standard deviations
        price_innovations = correlated_innovations[0, :] * price_log_return_std * volatility_multiplier
        gen_innovations = correlated_innovations[1, :] * gen_residual_std * volatility_multiplier

        # Reconstruct price series using log-additive approach
        # Start from first price, accumulate log returns
        price_path = np.zeros(n_steps)
        price_path[0] = base_prices[0]
        for t in range(1, n_steps):
            # Apply innovation as log return
            price_path[t] = price_path[t-1] * np.exp(price_innovations[t-1])

        # Optionally preserve mean
        if preserve_mean:
            current_mean = price_path.mean()
            if current_mean > 0:
                price_path = price_path * (price_mean / current_mean)

        price_scenarios[:, i] = price_path

        # Reconstruct generation series using additive approach
        # Apply base pattern plus correlated innovations
        gen_path = base_gen + gen_innovations

        # Optionally preserve mean
        if preserve_mean:
            gen_path = gen_path - gen_path.mean() + gen_mean

        gen_scenarios[:, i] = gen_path

        # Ensure non-negative values
        price_scenarios[:, i] = np.maximum(price_scenarios[:, i], 1.0)
        gen_scenarios[:, i] = np.maximum(gen_scenarios[:, i], 0.0)

    # Create DataFrames
    price_df = pd.DataFrame(
        price_scenarios,
        index=price_aligned.index,
        columns=[f'scenario_{i}' for i in range(n_scenarios)]
    )

    gen_df = pd.DataFrame(
        gen_scenarios,
        index=gen_aligned.index,
        columns=[f'scenario_{i}' for i in range(n_scenarios)]
    )

    logger.info(
        f"Generated {n_scenarios} correlated scenarios: "
        f"price_mean={price_df.values.mean():.2f} (base={price_mean:.2f}), "
        f"gen_mean={gen_df.values.mean():.2f} (base={gen_mean:.2f})"
    )

    return {'prices': price_df, 'generation': gen_df}


def create_stress_test_suite(
    base_price: pd.DataFrame,
    base_wind: Optional[pd.DataFrame] = None,
    base_solar: Optional[pd.DataFrame] = None,
    scenario_types: Optional[List[str]] = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate comprehensive stress test scenarios.

    Stress parameters are read from config.yaml under 'synthetic.stress_scenarios'.
    Hard-coded fallback values are used if config values are not available.

    Args:
        base_price: Base price series
        base_wind: Base wind generation series (optional)
        base_solar: Base solar generation series (optional)
        scenario_types: List of scenario types to generate. If None, generates all.

    Returns:
        Dictionary mapping scenario_name → {'prices', 'wind', 'solar'}
    """
    if scenario_types is None:
        scenario_types = [
            'high_volatility',
            'low_wind',
            'low_solar',
            'price_spike',
            'renewable_drought',
            'extreme_weather'
        ]

    # Load stress parameters from config
    from src.config.load_config import get_config
    config = get_config()
    stress_config = config.get('synthetic', {}).get('stress_scenarios', {})

    logger.info(f"Creating stress test suite with {len(scenario_types)} scenarios (config-driven)")

    stress_scenarios = {}

    for scenario_type in scenario_types:
        scenario_data = {}

        if scenario_type == 'high_volatility':
            # Increase price volatility
            volatility_mult = stress_config.get('high_volatility', {}).get('volatility_multiplier', 2.5)
            stressed_price = base_price.copy()
            prices = stressed_price['price'].values
            mean_price = prices.mean()
            deviations = prices - mean_price
            stressed_price['price'] = mean_price + deviations * volatility_mult
            scenario_data['prices'] = stressed_price

            if base_wind is not None:
                scenario_data['wind'] = base_wind.copy()
            if base_solar is not None:
                scenario_data['solar'] = base_solar.copy()

        elif scenario_type == 'low_wind' and base_wind is not None:
            # Reduce wind generation
            reduction_factor = stress_config.get('low_wind', {}).get('reduction_factor', 0.5)
            stressed_wind = base_wind.copy()
            stressed_wind['generation_mw'] *= reduction_factor
            stressed_wind['capacity_factor'] *= reduction_factor
            scenario_data['wind'] = stressed_wind
            scenario_data['prices'] = base_price.copy()

            if base_solar is not None:
                scenario_data['solar'] = base_solar.copy()

        elif scenario_type == 'low_solar' and base_solar is not None:
            # Reduce solar generation
            reduction_factor = stress_config.get('low_solar', {}).get('reduction_factor', 0.3)
            stressed_solar = base_solar.copy()
            stressed_solar['generation_mw'] *= reduction_factor
            stressed_solar['capacity_factor'] *= reduction_factor
            scenario_data['solar'] = stressed_solar
            scenario_data['prices'] = base_price.copy()

            if base_wind is not None:
                scenario_data['wind'] = base_wind.copy()

        elif scenario_type == 'price_spike':
            # Add extreme price events
            spike_magnitude = stress_config.get('price_spike', {}).get('spike_magnitude', 200.0)
            stressed_price = base_price.copy()
            prices = stressed_price['price'].values
            # Add spikes to 1% of observations
            n_spikes = max(1, len(prices) // 100)
            spike_indices = np.random.choice(len(prices), size=n_spikes, replace=False)
            stressed_price.loc[stressed_price.index[spike_indices], 'price'] += spike_magnitude
            scenario_data['prices'] = stressed_price

            if base_wind is not None:
                scenario_data['wind'] = base_wind.copy()
            if base_solar is not None:
                scenario_data['solar'] = base_solar.copy()

        elif scenario_type == 'renewable_drought':
            # Simultaneous low wind and solar
            wind_reduction = stress_config.get('renewable_drought', {}).get('wind_reduction', 0.6)
            solar_reduction = stress_config.get('renewable_drought', {}).get('solar_reduction', 0.4)

            if base_wind is not None:
                stressed_wind = base_wind.copy()
                stressed_wind['generation_mw'] *= wind_reduction
                stressed_wind['capacity_factor'] *= wind_reduction
                scenario_data['wind'] = stressed_wind

            if base_solar is not None:
                stressed_solar = base_solar.copy()
                stressed_solar['generation_mw'] *= solar_reduction
                stressed_solar['capacity_factor'] *= solar_reduction
                scenario_data['solar'] = stressed_solar

            scenario_data['prices'] = base_price.copy()

        elif scenario_type == 'extreme_weather':
            # Combined high volatility + low generation
            # Use extreme_weather config if available, otherwise use higher multipliers
            extreme_config = stress_config.get('extreme_weather', {})
            volatility_mult = extreme_config.get('volatility_multiplier', 3.0)
            wind_reduction = extreme_config.get('wind_reduction', 0.3)
            solar_reduction = extreme_config.get('solar_reduction', 0.5)

            stressed_price = base_price.copy()
            prices = stressed_price['price'].values
            mean_price = prices.mean()
            deviations = prices - mean_price
            stressed_price['price'] = mean_price + deviations * volatility_mult
            scenario_data['prices'] = stressed_price

            if base_wind is not None:
                stressed_wind = base_wind.copy()
                stressed_wind['generation_mw'] *= wind_reduction
                stressed_wind['capacity_factor'] *= wind_reduction
                scenario_data['wind'] = stressed_wind

            if base_solar is not None:
                stressed_solar = base_solar.copy()
                stressed_solar['generation_mw'] *= solar_reduction
                stressed_solar['capacity_factor'] *= solar_reduction
                scenario_data['solar'] = stressed_solar

        if scenario_data:
            stress_scenarios[scenario_type] = scenario_data

    logger.info(f"Created {len(stress_scenarios)} stress test scenarios")

    return stress_scenarios


def resample_to_frequency(
    data: pd.DataFrame,
    target_freq: str,
    agg_method: Union[str, Dict] = 'mean'
) -> pd.DataFrame:
    """
    Resample time series to different frequency.

    Handles both upsampling and downsampling:
    - Upsampling: uses asfreq() with forward fill for prices, interpolation for generation
    - Downsampling: uses mean aggregation (or specified agg_method)

    Args:
        data: DataFrame with DatetimeIndex
        target_freq: Target frequency ('H', 'D', 'M', etc.)
        agg_method: Aggregation method or dict of column: method (used only for downsampling)

    Returns:
        Resampled DataFrame

    Raises:
        ValueError: If data doesn't have DatetimeIndex
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex for resampling")

    logger.info(f"Resampling data to frequency: {target_freq}")

    if len(data) < 2:
        return data.asfreq(target_freq)

    # Determine if upsampling or downsampling
    current_delta = (data.index[1] - data.index[0]).total_seconds()
    target_index = pd.date_range(data.index[0], periods=2, freq=target_freq)
    target_delta = (target_index[1] - target_index[0]).total_seconds()

    is_upsampling = target_delta < current_delta

    if is_upsampling:
        # Upsampling: use asfreq with column-aware filling
        resampled = data.asfreq(target_freq)

        for col in resampled.columns:
            col_lower = col.lower()
            if 'price' in col_lower or 'lmp' in col_lower:
                # Forward fill for price-like columns
                resampled[col] = resampled[col].fillna(method='ffill')
            elif 'generation' in col_lower or 'capacity' in col_lower or 'mw' in col_lower:
                # Time interpolation for generation/capacity columns
                resampled[col] = resampled[col].interpolate(method='time')
            else:
                # Default: forward fill
                resampled[col] = resampled[col].fillna(method='ffill')

    else:
        # Downsampling: use aggregation
        # Apply sensible defaults based on column names if agg_method is string
        if isinstance(agg_method, str):
            agg_dict = {}
            for col in data.columns:
                if 'price' in col.lower():
                    agg_dict[col] = 'mean'
                elif 'generation' in col.lower() or 'mw' in col.lower():
                    agg_dict[col] = 'mean'
                elif 'capacity_factor' in col.lower():
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = agg_method
            agg_method = agg_dict

        resampled = data.resample(target_freq).agg(agg_method)

    logger.info(f"Resampled from {len(data)} to {len(resampled)} rows")

    return resampled


def calculate_net_demand(
    demand_series: pd.DataFrame,
    wind_series: Optional[pd.DataFrame] = None,
    solar_series: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate net demand (demand - renewable generation).

    Args:
        demand_series: DataFrame with demand data
        wind_series: DataFrame with wind generation (optional)
        solar_series: DataFrame with solar generation (optional)

    Returns:
        DataFrame with columns ['demand', 'wind', 'solar', 'net_demand']
    """
    result = demand_series.copy()

    # Find demand column
    demand_col = None
    for col in ['demand', 'load', 'MW', 'mw']:
        if col in result.columns:
            demand_col = col
            break

    if demand_col is None:
        raise ValueError("Could not find demand column in demand_series")

    # Initialize output DataFrame
    output = pd.DataFrame(index=result.index)
    output['demand'] = result[demand_col]

    # Add wind generation if provided
    if wind_series is not None:
        _, wind_aligned = align_timestamps(result, wind_series, method='left')
        wind_col = 'generation_mw' if 'generation_mw' in wind_aligned.columns else wind_aligned.columns[0]
        output['wind'] = wind_aligned[wind_col].fillna(0)
    else:
        output['wind'] = 0

    # Add solar generation if provided
    if solar_series is not None:
        _, solar_aligned = align_timestamps(result, solar_series, method='left')
        solar_col = 'generation_mw' if 'generation_mw' in solar_aligned.columns else solar_aligned.columns[0]
        output['solar'] = solar_aligned[solar_col].fillna(0)
    else:
        output['solar'] = 0

    # Calculate net demand
    output['net_demand'] = output['demand'] - output['wind'] - output['solar']

    # Ensure net demand is non-negative
    output['net_demand'] = np.maximum(output['net_demand'], 0)

    logger.info(
        f"Calculated net demand: mean_demand={output['demand'].mean():.2f}, "
        f"mean_net_demand={output['net_demand'].mean():.2f}"
    )

    return output


def export_for_modeling(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    output_path: Path,
    train_test_split: float = 0.8
) -> Dict[str, Path]:
    """
    Export blended data in format ready for ML models.

    Args:
        data: DataFrame to export
        feature_columns: List of feature column names
        target_column: Target column name
        output_path: Base output path
        train_test_split: Train/test split ratio (default 0.8)

    Returns:
        Dictionary with paths to train and test files
    """
    if not 0 < train_test_split < 1:
        raise ValueError(f"train_test_split must be between 0 and 1, got {train_test_split}")

    logger.info(
        f"Exporting data for modeling: features={len(feature_columns)}, "
        f"target={target_column}, split={train_test_split}"
    )

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Time-based split (not random)
    split_idx = int(len(data) * train_test_split)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Select columns
    all_columns = feature_columns + [target_column]
    train_export = train_data[all_columns]
    test_export = test_data[all_columns]

    # Save to Parquet
    train_path = output_path / 'train.parquet'
    test_path = output_path / 'test.parquet'

    train_export.to_parquet(train_path, compression='snappy')
    test_export.to_parquet(test_path, compression='snappy')

    logger.info(
        f"Exported: train={len(train_export)} rows → {train_path}, "
        f"test={len(test_export)} rows → {test_path}"
    )

    return {
        'train': train_path,
        'test': test_path
    }


def add_scenario_metadata(
    data: pd.DataFrame,
    metadata: Dict[str, Any]
) -> pd.DataFrame:
    """
    Add metadata tags to scenario DataFrames.

    Args:
        data: DataFrame to add metadata to
        metadata: Dictionary with metadata keys

    Returns:
        DataFrame with metadata columns added
    """
    result = data.copy()

    for key, value in metadata.items():
        result[f'meta_{key}'] = value

    logger.info(f"Added {len(metadata)} metadata fields to DataFrame")

    return result


# Helper functions

def _resample_with_direction(
    data: pd.DataFrame,
    target_freq: str
) -> pd.DataFrame:
    """
    Resample DataFrame handling upsampling and downsampling appropriately.

    For downsampling: uses mean aggregation
    For upsampling: uses asfreq() with column-aware filling

    Args:
        data: DataFrame with DatetimeIndex
        target_freq: Target frequency string

    Returns:
        Resampled DataFrame
    """
    if len(data) < 2:
        return data.asfreq(target_freq)

    # Infer current frequency
    current_freq = pd.infer_freq(data.index)

    # Determine if upsampling or downsampling
    # Compare time deltas - upsampling means smaller target delta
    current_delta = (data.index[1] - data.index[0]).total_seconds()
    target_index = pd.date_range(data.index[0], periods=2, freq=target_freq)
    target_delta = (target_index[1] - target_index[0]).total_seconds()

    is_upsampling = target_delta < current_delta

    if is_upsampling:
        # Upsampling: use asfreq and fill appropriately
        resampled = data.asfreq(target_freq)

        # Fill columns based on type
        for col in resampled.columns:
            col_lower = col.lower()
            if 'price' in col_lower or 'lmp' in col_lower:
                # Forward fill for price-like columns
                resampled[col] = resampled[col].fillna(method='ffill')
            elif 'generation' in col_lower or 'capacity' in col_lower or 'mw' in col_lower:
                # Time interpolation for generation-like columns
                resampled[col] = resampled[col].interpolate(method='time')
            else:
                # Default: forward fill
                resampled[col] = resampled[col].fillna(method='ffill')

    else:
        # Downsampling: use mean aggregation
        resampled = data.resample(target_freq).mean()

    return resampled


def _normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across real and synthetic data.

    Args:
        data: DataFrame with potentially inconsistent column names

    Returns:
        DataFrame with normalized column names
    """
    # Mapping of common variations
    column_mapping = {
        'LMP': 'price',
        'lmp': 'price',
        'Price': 'price',
        'MW': 'generation_mw',
        'mw': 'generation_mw',
        'generation': 'generation_mw',
        'LOAD': 'demand',
        'load': 'demand',
        'Load': 'demand'
    }

    # Rename columns
    renamed = data.rename(columns=column_mapping)

    return renamed


def _check_schema_compatibility(
    df1: pd.DataFrame,
    df2: pd.DataFrame
) -> List[str]:
    """
    Verify two DataFrames have compatible schemas.

    Args:
        df1: First DataFrame
        df2: Second DataFrame

    Returns:
        List of incompatibility issues (empty if compatible)
    """
    issues = []

    # Check if both have datetime index
    if isinstance(df1.index, pd.DatetimeIndex) != isinstance(df2.index, pd.DatetimeIndex):
        issues.append("Inconsistent index types (one is DatetimeIndex, other is not)")

    # Check for common columns
    common_cols = set(df1.columns) & set(df2.columns)
    if not common_cols:
        issues.append("No common columns between DataFrames")
    else:
        # Check dtypes of common columns
        for col in common_cols:
            if df1[col].dtype != df2[col].dtype:
                issues.append(f"Column '{col}' has different dtypes: {df1[col].dtype} vs {df2[col].dtype}")

    return issues


def _detect_outliers(
    series: pd.Series,
    method: str = 'iqr',
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers using IQR or z-score method.

    Args:
        series: Series to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection

    Returns:
        Boolean mask of outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold

    else:
        raise ValueError(f"method must be 'iqr' or 'zscore', got '{method}'")

    return outliers.values


if __name__ == "__main__":
    # Setup logging
    from src.config.load_config import setup_logging
    setup_logging()

    print("Data Utilities Example")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='H')

    real_data = pd.DataFrame({
        'price': np.random.uniform(40, 60, len(dates[:500]))
    }, index=dates[:500])

    synthetic_data = pd.DataFrame({
        'price': np.random.uniform(45, 65, len(dates[500:]))
    }, index=dates[500:])

    print("\n1. Blending real and synthetic data")
    print("-" * 50)

    blended = blend_real_and_synthetic(
        real_data=real_data,
        synthetic_data=synthetic_data,
        blend_strategy='concatenate',
        add_metadata=True
    )

    print(f"Blended data: {len(blended)} rows")
    print(f"Data sources:\n{blended['data_source'].value_counts()}")

    # Validation
    print("\n2. Data validation")
    print("-" * 50)

    is_valid, issues = validate_data_consistency(blended, data_type='price')
    print(f"Valid: {is_valid}")
    if issues:
        print(f"Issues found: {issues}")
    else:
        print("No issues found")

    # Resampling
    print("\n3. Resampling to daily frequency")
    print("-" * 50)

    daily_data = resample_to_frequency(blended, target_freq='D', agg_method='mean')
    print(f"Resampled to {len(daily_data)} daily observations")
    print(daily_data.head())
