import logging
from typing import cast

import pandas as pd


def check_missing_values(df: pd.DataFrame) -> None:
    """Checks and reports missing values.

    Args:
        df (pd.DataFrame): The DataFrame to check for missing values.

    Raises:
        ValueError: If the DataFrame is empty.

    Returns:
        None
    """
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        logging.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
    else:
        logging.info("No missing values found in the dataframe.")


def identify_constant_features(df: pd.DataFrame) -> list[str]:
    """Identifies features with zero variance (constant values).
    Args:
        df (pd.DataFrame): The DataFrame to check for constant features.

    Returns:
        list[str]: A list of column names that are constant (zero variance).

    """
    logging.info("Identifying constant features...")
    try:
        stds = df.std(numeric_only=True)
        constant_cols = cast(list[str], stds[stds == 0].index.tolist())
        if constant_cols:
            logging.info(f"Found constant features: {constant_cols}")
        else:
            logging.info("No constant features identified.")
        return constant_cols
    except Exception as e:
        logging.error(f"Error calculating standard deviations: {e}")
        return cast(list[str], [])


def drop_features(df: pd.DataFrame, features_to_drop: list[str]) -> pd.DataFrame:
    """Drops specified features from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame from which to drop features.
        features_to_drop (list[str]): A list of feature names to drop.

    Returns:
        pd.DataFrame: The DataFrame after dropping the specified features.

    """
    if not features_to_drop:
        logging.info("No features specified for dropping.")
        return df

    existing_features_to_drop = [col for col in features_to_drop if col in df.columns]
    if not existing_features_to_drop:
        logging.warning(
            f"None of the specified features to drop {features_to_drop} exist in the DataFrame."
        )
        return df

    logging.info(f"Dropping features: {existing_features_to_drop}")
    df_dropped = df.drop(columns=existing_features_to_drop)
    logging.info(f"DataFrame shape after dropping features: {df_dropped.shape}")
    return df_dropped
