import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def get_dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    """
    Extracts basic information about the CMAPSS dataset.
    """
    engines = df["engine_id"].unique()
    num_engines = len(engines)
    total_cycles = len(df)

    lifetimes = df.groupby("engine_id")["cycle"].max().values
    avg_lifetime = np.mean(lifetimes) if len(lifetimes) > 0 else 0.0
    max_lifetime = np.max(lifetimes) if len(lifetimes) > 0 else 0.0
    min_lifetime = np.min(lifetimes) if len(lifetimes) > 0 else 0.0
    std_lifetime = np.std(lifetimes) if len(lifetimes) > 0 else 0.0

    return {
        "num_engines": num_engines,
        "total_cycles": total_cycles,
        "avg_lifetime": avg_lifetime,
        "max_lifetime": max_lifetime,
        "min_lifetime": min_lifetime,
        "std_lifetime": std_lifetime,
        "lifetime_distribution": lifetimes.tolist(),
    }


def get_sensor_statistics(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """
    Calculate statistics for each sensor across all engines.
    """
    stats_dict = {}
    for col in sensor_cols:
        if col in df.columns:
            data_for_stats = df[col].dropna().astype(float)
            skewness, kurtosis = np.nan, np.nan
            if not data_for_stats.empty:
                if np.all(
                    data_for_stats == data_for_stats.iloc[0]
                ):  # Check if all values are the same
                    skewness = (
                        0.0  # Or np.nan, skewness is ill-defined or 0 for constant
                    )
                    kurtosis = (
                        -3.0
                    )  # Kurtosis of a constant is -3 (Fisher), or 0 (Pearson)
                # Scipy's default is Fisher. For a single point, it's 0.
                # Let's be consistent or choose np.nan if values are truly constant.
                # For nearly constant values, scipy might give RuntimeWarning.
                elif (
                    len(data_for_stats) > 1
                ):  # Skew and Kurtosis need more than 1 point generally
                    skewness = stats.skew(data_for_stats)
                    kurtosis = stats.kurtosis(data_for_stats)

            stats_dict[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
                "25%": df[col].quantile(0.25),
                "75%": df[col].quantile(0.75),
                "skewness": skewness,
                "kurtosis": kurtosis,
            }
    return pd.DataFrame.from_dict(stats_dict, orient="index")


def analyze_feature_correlations(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix between specified columns.
    """
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("None of the specified columns exist for correlation analysis.")
        return pd.DataFrame()
    return df[valid_columns].corr()


def _get_single_engine_sensor_trend_and_fluctuation(
    engine_sensor_data: pd.Series, window_size: int
) -> tuple[str, bool]:
    """
    Determines trend and fluctuation for a single engine's sensor data.
    Returns: (trend_type: str, is_fluctuating: bool)
             trend_type can be 'increasing', 'decreasing', 'stable', 'short_series'.
    """
    # Ensure there are enough data points for regression and rolling window
    min_points_for_trend = max(2, window_size // 2, 1)  # Heuristic for minimum points
    if len(engine_sensor_data) < min_points_for_trend:
        return "short_series", False

    valid_data = engine_sensor_data.dropna()
    if len(valid_data) < 2:  # Need at least 2 points for linear regression
        return "short_series", False

    x_coords = np.arange(len(valid_data))
    try:
        slope, _, _, _, _ = stats.linregress(x_coords, valid_data.values)
    except (
        ValueError
    ):  # Can happen if x_coords and y have different lengths after dropna etc.
        return "stable", False  # Default or error state

    # Determine trend type based on slope
    # Threshold for slope significance can be adjusted (e.g., relative to data std)
    trend_slope_threshold = 1e-3
    trend_type = "stable"
    if abs(slope) > trend_slope_threshold:
        trend_type = "increasing" if slope > 0 else "decreasing"

    # Determine fluctuation
    is_fluctuating = False
    if len(valid_data) >= window_size:
        rolling_std = valid_data.rolling(
            window=window_size, center=True, min_periods=1
        ).std()
        mean_abs_value = abs(valid_data.mean())
        # Fluctuation is significant if average rolling std is a notable fraction of the mean
        # or if absolute std dev is high when mean is near zero
        fluctuation_std_threshold_abs = (
            0.1  # Example absolute threshold for std when mean is ~0
        )
        fluctuation_std_threshold_rel = 0.05  # Example relative threshold (5% of mean)

        if mean_abs_value > 1e-6:  # If mean is not near zero, use relative threshold
            if rolling_std.mean() > fluctuation_std_threshold_rel * mean_abs_value:
                is_fluctuating = True
        elif (
            rolling_std.mean() > fluctuation_std_threshold_abs
        ):  # If mean is near zero, use absolute threshold
            is_fluctuating = True

    return trend_type, is_fluctuating


def _aggregate_engine_trends_to_overall(
    engine_level_trends: list[str],
    engine_level_fluctuations: list[bool],
) -> str:
    """
    Aggregates per-engine trends and fluctuations to determine an overall trend for a sensor.
    """
    if not engine_level_trends:
        return "undetermined"  # No valid per-engine trends found

    num_valid_engines = len(engine_level_trends)

    # Count fluctuations among valid engines
    num_fluctuating = sum(engine_level_fluctuations)
    fluctuation_ratio = (
        num_fluctuating / num_valid_engines if num_valid_engines > 0 else 0.0
    )

    # Count trend types
    trend_counts = pd.Series(engine_level_trends).value_counts()

    # Determine overall trend based on heuristics
    if fluctuation_ratio > 0.4:  # More than 40% of engines fluctuate significantly
        return "fluctuating"

    if not trend_counts.empty:
        dominant_trend_type = trend_counts.idxmax()
        dominant_trend_ratio = trend_counts.max() / num_valid_engines

        if dominant_trend_type == "increasing" and dominant_trend_ratio > 0.5:
            return "increasing"
        if dominant_trend_type == "decreasing" and dominant_trend_ratio > 0.5:
            return "decreasing"
        if (
            dominant_trend_type == "stable" and dominant_trend_ratio > 0.4
        ):  # Stable needs less dominance
            return "stable"
        # If no single trend is strongly dominant, or mixed signals
        return "mixed"  # Or could be dominant_trend_type if preferred

    return "stable"  # Default if trend_counts was empty (should be caught earlier)


def identify_trend_patterns(
    df: pd.DataFrame, sensor_cols: list[str], window_size: int = 5
) -> dict[str, str]:
    """
    Identifies overall trend patterns for each sensor based on per-engine behavior.
    """
    overall_sensor_trends: dict[str, str] = {}
    unique_engine_ids = df["engine_id"].unique()
    num_total_engines = len(unique_engine_ids)

    if num_total_engines == 0:
        logger.warning(
            "No engines found in DataFrame for trend pattern identification."
        )
        return {sensor: "no_data" for sensor in sensor_cols}

    for sensor in sensor_cols:
        if sensor not in df.columns:
            logger.warning(
                f"Sensor {sensor} not found in DataFrame for trend pattern identification."
            )
            overall_sensor_trends[sensor] = "sensor_not_found"
            continue

        engine_level_trends: list[str] = []
        engine_level_fluctuations: list[bool] = []

        for engine_id in unique_engine_ids:
            engine_sensor_data = df[df["engine_id"] == engine_id][sensor]
            trend, is_fluctuating = _get_single_engine_sensor_trend_and_fluctuation(
                engine_sensor_data, window_size
            )
            # We only aggregate trends from series that were not too short
            if trend != "short_series":
                engine_level_trends.append(trend)
                engine_level_fluctuations.append(is_fluctuating)

        overall_sensor_trends[sensor] = _aggregate_engine_trends_to_overall(
            engine_level_trends, engine_level_fluctuations
        )

    return overall_sensor_trends
