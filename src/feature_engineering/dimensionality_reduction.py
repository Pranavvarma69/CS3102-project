import logging
from typing import Optional

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.feature_engineering.window_statistics import (
    add_lag_features,
    add_rate_of_change,
    add_remaining_useful_life,
    add_rolling_statistics,
)


def apply_pca(
    df: pd.DataFrame,
    columns: list[str],
    n_components: int = 2,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Apply PCA dimensionality reduction to the specified columns.

    Args:
        df: DataFrame with engine data
        columns: list of column names to use for PCA
        n_components: Number of PCA components to generate
        standardize: Whether to standardize the data before PCA

    Returns:
        DataFrame with original data and PCA components added
    """
    result_df = df.copy()

    # Ensure all columns exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns]

    if not valid_columns:
        logging.warning("None of the specified columns exist in the DataFrame")
        return result_df

    logging.info(
        f"Applying PCA with {n_components} components to {len(valid_columns)} features"
    )

    # Extract features for PCA
    x = df[valid_columns].values

    # Handle missing values first
    imputer = SimpleImputer(strategy="mean")
    x = imputer.fit_transform(x)

    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(x)

    # Add PCA components to the result DataFrame
    for i in range(n_components):
        result_df[f"pca_{i + 1}"] = pca_result[:, i]

    # Log variance explained
    variance_explained = pca.explained_variance_ratio_
    logging.info(f"Variance explained by PCA components: {variance_explained}")

    return result_df


def apply_tsne(
    df: pd.DataFrame,
    columns: list[str],
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply t-SNE dimensionality reduction to the specified columns.

    Args:
        df: DataFrame with engine data
        columns: list of column names to use for t-SNE
        n_components: Number of t-SNE components to generate
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility

    Returns:
        DataFrame with original data and t-SNE components added
    """
    result_df = df.copy()

    # Ensure all columns exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns]

    if not valid_columns:
        logging.warning("None of the specified columns exist in the DataFrame")
        return result_df

    logging.info(
        f"Applying t-SNE with {n_components} components to {len(valid_columns)} features"
    )

    # Extract features for t-SNE
    x = df[valid_columns].values

    # Handle missing values first
    imputer = SimpleImputer(strategy="mean")
    x = imputer.fit_transform(x)

    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(x) - 1),  # Adjust perplexity if needed
        random_state=random_state,
    )
    tsne_result = tsne.fit_transform(x)

    # Add t-SNE components to the result DataFrame
    for i in range(n_components):
        result_df[f"tsne_{i + 1}"] = tsne_result[:, i]

    return result_df


def generate_engineered_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_size: int = 5,
    lag: int = 3,
    apply_dimensionality_reduction: bool = True,
    n_components: int = 2,
) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline.

    Args:
        df: DataFrame with engine data
        sensor_cols: list of sensor column names
        window_size: Size of the rolling window for statistics
        lag: Number of lagged values to add
        apply_dimensionality_reduction: Whether to apply PCA
        n_components: Number of PCA components if enabled

    Returns:
        DataFrame with all engineered features
    """
    logging.info("Starting feature engineering pipeline")

    # Create a copy of the DataFrame to modify
    result_df = df.copy()

    # Step 1: Add rolling statistics
    result_df = add_rolling_statistics(result_df, sensor_cols, window_size)

    # Step 2: Add lag features
    result_df = add_lag_features(result_df, sensor_cols, lag)

    # Step 3: Add rate of change
    result_df = add_rate_of_change(result_df, sensor_cols)

    # Step 4: Add RUL
    result_df = add_remaining_useful_life(result_df)

    # Step 5: Apply dimensionality reduction if requested
    if apply_dimensionality_reduction:
        # Get all numeric feature columns for PCA except RUL
        feature_cols = [
            col
            for col in result_df.columns
            if col != "RUL"
            and col != "engine_id"
            and col != "cycle"
            and pd.api.types.is_numeric_dtype(result_df[col])
        ]

        # Apply PCA
        result_df = apply_pca(result_df, feature_cols, n_components=n_components)

    # Drop rows with NaN values (caused by lag and rolling window calculations)
    initial_rows = len(result_df)
    result_df = result_df.dropna()
    dropped_rows = initial_rows - len(result_df)
    logging.info(f"Dropped {dropped_rows} rows with NaN values")

    return result_df
