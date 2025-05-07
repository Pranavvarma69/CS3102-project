import logging

import pandas as pd


def add_rolling_statistics(
    df: pd.DataFrame, sensor_cols: list[str], window_size: int = 5
) -> pd.DataFrame:
    """
    Add rolling statistics (mean, std, min, max) for sensor measurements.

    Args:
        df: DataFrame with engine data
        sensor_cols: List of sensor column names
        window_size: Size of the rolling window

    Returns:
        DataFrame with added rolling statistics columns
    """
    result_df = df.copy()

    logging.info(f"Adding rolling statistics with window size {window_size}")

    for engine_id in df["engine_id"].unique():
        engine_mask = df["engine_id"] == engine_id

        for col in sensor_cols:
            if col not in df.columns:
                logging.warning(f"Column {col} not found in DataFrame")
                continue

            # Calculate rolling statistics
            rolling_data = df.loc[engine_mask, col].rolling(window=window_size)

            # Add statistics as new columns
            result_df.loc[engine_mask, f"{col}_rolling_mean"] = rolling_data.mean()
            result_df.loc[engine_mask, f"{col}_rolling_std"] = rolling_data.std()
            result_df.loc[engine_mask, f"{col}_rolling_min"] = rolling_data.min()
            result_df.loc[engine_mask, f"{col}_rolling_max"] = rolling_data.max()

    return result_df


def add_lag_features(
    df: pd.DataFrame, sensor_cols: list[str], lag: int = 3
) -> pd.DataFrame:
    """
    Add lagged values for sensor measurements.

    Args:
        df: DataFrame with engine data
        sensor_cols: List of sensor column names
        lag: Number of lagged values to add

    Returns:
        DataFrame with added lag columns
    """
    result_df = df.copy()

    logging.info(f"Adding lag features with lag={lag}")

    for engine_id in df["engine_id"].unique():
        engine_mask = df["engine_id"] == engine_id
        engine_data = df.loc[engine_mask].sort_values("cycle")

        for col in sensor_cols:
            if col not in df.columns:
                logging.warning(f"Column {col} not found in DataFrame")
                continue

            # Add lagged columns
            for i in range(1, lag + 1):
                result_df.loc[engine_mask, f"{col}_lag_{i}"] = (
                    engine_data[col].shift(i).values
                )

    return result_df


def add_rate_of_change(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """
    Add rate of change (percentage change) for sensor measurements.

    Args:
        df: DataFrame with engine data
        sensor_cols: List of sensor column names

    Returns:
        DataFrame with added rate of change columns
    """
    result_df = df.copy()

    logging.info("Adding rate of change features")

    for engine_id in df["engine_id"].unique():
        engine_mask = df["engine_id"] == engine_id
        engine_data = df.loc[engine_mask].sort_values("cycle")

        for col in sensor_cols:
            if col not in df.columns:
                logging.warning(f"Column {col} not found in DataFrame")
                continue

            # Calculate percentage change
            pct_change = engine_data[col].pct_change().values
            result_df.loc[engine_mask, f"{col}_rate_of_change"] = pct_change

            # Calculate absolute change
            abs_change = engine_data[col].diff().values
            result_df.loc[engine_mask, f"{col}_abs_change"] = abs_change

    return result_df


def add_remaining_useful_life(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add Remaining Useful Life (RUL) for each engine.

    Args:
        df: DataFrame with engine data containing 'engine_id' and 'cycle' columns

    Returns:
        DataFrame with added RUL column
    """
    result_df = df.copy()

    logging.info("Adding Remaining Useful Life (RUL) feature")

    # Get the maximum cycle for each engine
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.rename(columns={"cycle": "max_cycle"}, inplace=True)

    # Merge with the original dataframe
    result_df = pd.merge(result_df, max_cycles, on="engine_id", how="left")

    # Calculate RUL as the difference between max cycle and current cycle
    result_df["RUL"] = result_df["max_cycle"] - result_df["cycle"]

    # Drop the temporary max_cycle column
    result_df.drop(columns=["max_cycle"], inplace=True)

    return result_df
