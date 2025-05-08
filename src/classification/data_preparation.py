import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.utils import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def load_and_merge_data(
    dataset_name: str,
    cluster_label_file_suffix: str = "_clustered_all_algos.parquet",
    feature_file_suffix: str = "_features.parquet",
) -> Optional[pd.DataFrame]:
    """
    Loads clustered labels and engineered features, then merges them.

    Args:
        dataset_name: Identifier for the dataset (e.g., "FD001").
        cluster_label_file_suffix: Suffix for the clustered label parquet file.
        feature_file_suffix: Suffix for the features parquet file.

    Returns:
        A merged DataFrame or None if loading fails.
    """
    clustered_labels_dir = os.path.join(PROCESSED_DATA_DIR, "clustered_labels")
    features_dir = os.path.join(PROCESSED_DATA_DIR, "features")

    cluster_file_path = os.path.join(
        clustered_labels_dir, f"{dataset_name}{cluster_label_file_suffix}"
    )
    feature_file_path = os.path.join(
        features_dir, f"{dataset_name}{feature_file_suffix}"
    )

    if not os.path.exists(cluster_file_path):
        logger.error(f"Cluster label file not found: {cluster_file_path}")
        return None
    if not os.path.exists(feature_file_path):
        logger.error(f"Feature file not found: {feature_file_path}")
        return None

    try:
        df_clusters = pd.read_parquet(cluster_file_path)
        logger.info(
            f"Loaded clustered labels from {cluster_file_path}, shape: {df_clusters.shape}"
        )
        df_features = pd.read_parquet(feature_file_path)
        logger.info(
            f"Loaded features from {feature_file_path}, shape: {df_features.shape}"
        )

        # Select essential columns from df_clusters to avoid duplicate feature columns
        # Typically: engine_id, cycle, and the chosen stage label columns
        # Assuming RUL is also in df_features, or you can take it from df_clusters
        cluster_cols_to_keep = ["engine_id", "cycle"] + [
            col for col in df_clusters.columns if "_stage" in col
        ]
        # Ensure no overlap in other columns except keys
        common_cols_data = list(
            set(df_features.columns)
            .intersection(set(df_clusters.columns))
            .difference({"engine_id", "cycle"})  # C405: Use set literal
        )
        if common_cols_data:
            logger.warning(
                f"Common data columns found (excluding keys): {common_cols_data}. "
                f"Features from feature file will be prioritized if names conflict after merge."
            )
            # df_clusters = df_clusters.drop(columns=common_cols_data) # Option 1: Drop from cluster df
            # Option 2: Suffix, or ensure df_features is the source of truth for features

        df_merged = pd.merge(
            df_features,
            df_clusters[cluster_cols_to_keep],
            on=["engine_id", "cycle"],
            how="inner",  # Inner join to keep only rows present in both feature-engineered and clustered data
        )
        logger.info(f"Merged data shape: {df_merged.shape}")
        if df_merged.empty:
            logger.warning("Merged DataFrame is empty. Check keys or data alignment.")
            return None
        return df_merged
    except Exception as e:
        logger.error(f"Error loading or merging data for {dataset_name}: {e}")
        return None


def prepare_features_target(
    df: pd.DataFrame, target_stage_col: str, drop_cols: Optional[list[str]] = None
) -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Prepares features (X) and target (y) for classification.

    Args:
        df: The merged DataFrame containing features and target.
        target_stage_col: The name of the column to be used as the target variable.
        drop_cols: List of additional columns to drop from features (e.g., IDs, RUL, other stage labels).

    Returns:
        A tuple (features_df, target_series) or (None, None) if preparation fails.
    """
    if target_stage_col not in df.columns:
        logger.error(f"Target stage column '{target_stage_col}' not found.")
        return None, None

    df_processed = df.copy()

    # Drop rows where the target stage is NaN (e.g., if clustering was on a subsample)
    initial_rows = len(df_processed)
    df_processed.dropna(subset=[target_stage_col], inplace=True)
    if len(df_processed) < initial_rows:
        logger.info(
            f"Dropped {initial_rows - len(df_processed)} rows with NaN target stage ('{target_stage_col}')."
        )

    if df_processed.empty:
        logger.warning(
            f"DataFrame is empty after dropping NaNs in target column '{target_stage_col}'."
        )
        return None, None

    target_series = df_processed[target_stage_col].astype(
        int
    )  # Ensure target is integer

    default_drop = ["engine_id", "cycle", "RUL"] + [
        col
        for col in df_processed.columns
        if "_stage" in col and col != target_stage_col
    ]
    if drop_cols:
        default_drop.extend(drop_cols)

    cols_to_actually_drop = [col for col in default_drop if col in df_processed.columns]
    features_df = df_processed.drop(
        columns=cols_to_actually_drop
    )  # N806: X -> features_df

    # Ensure all features are numeric and handle any remaining NaNs in features
    # (though feature engineering should ideally handle NaNs for its output)
    features_df = features_df.select_dtypes(include=np.number)  # N806: X -> features_df
    if features_df.isnull().any().any():
        logger.warning("NaNs found in feature set features_df. Imputing with mean.")
        # In a real scenario, more sophisticated imputation might be needed
        # or ensure upstream processes (feature engineering) handle NaNs.
        for col in features_df.columns[features_df.isnull().any()]:
            features_df[col] = features_df[col].fillna(features_df[col].mean())

    if features_df.empty:
        logger.error("Feature set features_df is empty after pre-processing.")
        return None, None

    logger.info(
        f"Features (features_df) shape: {features_df.shape}, Target (target_series) shape: {target_series.shape}"
    )
    return features_df, target_series


def split_data_grouped(
    features_df: pd.DataFrame,  # N803: X -> features_df
    target_series: pd.Series,  # N803: y -> target_series
    engine_ids: pd.Series,
    test_size: float = 0.2,
    n_splits: int = 5,  # For GroupKFold, if used directly for CV
    random_state: Optional[int] = 42,
    use_group_kfold: bool = True,
) -> list[tuple[Any, Any]]:  # Returns list of (train_indices, val_indices)
    """
    Splits data into training and validation sets using GroupKFold based on engine_id.
    If not using GroupKFold directly for CV, it can provide a single train/val split.

    Args:
        features_df: Features DataFrame.
        target_series: Target Series.
        engine_ids: Series containing engine IDs for grouping.
        test_size: Proportion of the dataset to include in the validation split (if not using full CV).
        n_splits: Number of folds for GroupKFold.
        random_state: Random seed. (Note: GroupKFold is deterministic but good practice).
        use_group_kfold: If True, returns folds from GroupKFold. If False, returns a single train/test split
                         trying to respect groups somewhat, though GroupShuffleSplit is better for that.
                         For simplicity, let's return GroupKFold splits.

    Returns:
        List of (train_idx, val_idx) tuples for cross-validation.
    """
    if use_group_kfold:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(features_df, target_series, groups=engine_ids))
        logger.info(
            f"Generated {len(splits)} splits using GroupKFold based on engine_id."
        )
        return splits
    else:
        # For a single train/val split respecting groups, GroupShuffleSplit is better.
        # sklearn.model_selection.GroupShuffleSplit
        # For now, let's stick to GroupKFold and the user can pick one fold or iterate.
        # This function primarily returns the CV splits as per typical ML practice.
        # If a single split is needed:
        logger.warning(
            "Single train/val split not fully group-aware without GroupShuffleSplit. Using GroupKFold's first split logic."
        )
        gkf = GroupKFold(
            n_splits=int(1 / test_size) if 0 < test_size < 1 else 5
        )  # Approximate
        first_split = next(
            gkf.split(features_df, target_series, groups=engine_ids), None
        )
        if first_split:
            return [first_split]
        else:
            logger.error("Could not generate a single split using GroupKFold logic.")
            return []
