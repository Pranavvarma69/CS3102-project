import logging
from typing import Any, Optional  # Using built-in types where ruff fixed

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.classification.data_preparation import (
    load_and_merge_data,
)

logger = logging.getLogger(__name__)


def engineer_time_to_next_stage(
    df: pd.DataFrame, stage_column: str, max_stage_value: int
) -> pd.Series:
    """
    Engineers the target variable: time (in cycles) to the next degradation stage.

    Args:
        df: DataFrame containing 'engine_id', 'cycle', and the specified 'stage_column'.
            It's assumed that for each engine_id, data is sorted by 'cycle'.
        stage_column: The name of the column holding the custom degradation stage labels.
        max_stage_value: The highest possible stage value (e.g., 4 for stages 0-4).

    Returns:
        A pandas Series containing the time_to_next_stage for each data point.
        Points where the next stage is not reached will have pd.NA.
        Points in the max_stage_value will have time_to_next_stage = 0.
    """
    if not all(col in df.columns for col in ["engine_id", "cycle", stage_column]):
        raise ValueError(
            f"DataFrame must contain 'engine_id', 'cycle', and '{stage_column}'."
        )

    ttns_list = []  # To store (index, value) tuples

    for _engine_id, group in df.groupby("engine_id"):  # B007
        group = group.sort_values("cycle")  # Ensure sorted
        current_engine_indices = group.index

        # Create a lookup for the first entry cycle of each stage for this engine
        stage_entry_cycles = {}
        for stage_val in sorted(group[stage_column].unique()):
            if pd.notna(stage_val):
                stage_entry_cycles[stage_val] = group[group[stage_column] == stage_val][
                    "cycle"
                ].min()

        engine_ttns_values = pd.Series(index=current_engine_indices, dtype=float)

        for idx, row in group.iterrows():
            current_cycle = row["cycle"]
            current_stage = row[stage_column]

            if pd.isna(current_stage):
                engine_ttns_values.loc[idx] = pd.NA
                continue

            current_stage = int(current_stage)

            if current_stage == max_stage_value:
                engine_ttns_values.loc[idx] = 0
            else:
                target_next_stage = current_stage + 1

                next_stage_cycle_in_group = group[
                    (group[stage_column] == target_next_stage)
                    & (group["cycle"] > current_cycle)
                ]["cycle"].min()

                if pd.notna(next_stage_cycle_in_group):
                    engine_ttns_values.loc[idx] = (
                        next_stage_cycle_in_group - current_cycle
                    )
                else:
                    # Next stage was not reached for this engine after the current cycle
                    engine_ttns_values.loc[idx] = pd.NA

        for idx, val in engine_ttns_values.items():
            ttns_list.append((idx, val))

    # Create the final series preserving original DataFrame's index
    final_ttns_series = pd.Series(dict(ttns_list), name="time_to_next_stage")
    return final_ttns_series.reindex(df.index)


def prepare_regression_data(
    dataset_name: str,
    target_stage_col: str,  # From clustering, e.g., kmeans_stage
    max_cluster_stage_value: int,  # e.g. 4 if stages are 0,1,2,3,4
    cluster_label_file_suffix: str = "_clustered_all_algos.parquet",
    feature_file_suffix: str = "_features.parquet",
) -> tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:  # UP006
    """
    Loads data, engineers TTNS target, prepares features (x_features) and target (y_target) for regression.

    Returns:
        A tuple (x_features, y_target, engine_ids_for_split) or (None, None, None) if preparation fails.
    """
    logger.info(
        f"Preparing regression data for {dataset_name} using target stage {target_stage_col}"
    )
    df_merged = load_and_merge_data(
        dataset_name, cluster_label_file_suffix, feature_file_suffix
    )
    if df_merged is None:
        logger.error(f"Failed to load or merge data for {dataset_name} for regression.")
        return None, None, None

    if target_stage_col not in df_merged.columns:
        logger.error(
            f"Target stage column '{target_stage_col}' not found in merged data."
        )
        return None, None, None
    if "engine_id" not in df_merged.columns or "cycle" not in df_merged.columns:
        logger.error("'engine_id' or 'cycle' not found in merged data.")
        return None, None, None

    # Engineer the 'time_to_next_stage' target variable
    df_merged["time_to_next_stage"] = engineer_time_to_next_stage(
        df_merged, target_stage_col, max_cluster_stage_value
    )

    # Drop rows where TTNS is NA (e.g., next stage not reached or other issues)
    initial_rows = len(df_merged)
    df_merged.dropna(subset=["time_to_next_stage"], inplace=True)
    if len(df_merged) < initial_rows:
        logger.info(
            f"Dropped {initial_rows - len(df_merged)} rows with NA 'time_to_next_stage'."
        )

    if df_merged.empty:
        logger.warning("DataFrame is empty after TTNS engineering and NA drop.")
        return None, None, None

    y_target = df_merged["time_to_next_stage"]

    # Define features: exclude IDs, original RUL, other stage labels, and the new TTNS target
    # Also exclude the clustering stage column used for TTNS generation if it's not an intended feature.
    # For regression, current stage can be a feature.

    # Columns to exclude from features
    cols_to_drop_for_x = ["engine_id", "cycle", "RUL", "time_to_next_stage"]  # N806
    # Add any other stage columns from clustering output that are not the primary target_stage_col
    # (unless they are intended as features)
    other_stage_cols = [
        col for col in df_merged.columns if "_stage" in col and col != target_stage_col
    ]
    cols_to_drop_for_x.extend(other_stage_cols)

    # If target_stage_col itself should not be a feature (it's used to define "next"), drop it too.
    # However, current stage IS a very important feature for predicting TTNS.
    # So, target_stage_col (e.g. 'kmeans_stage') should be kept as a feature.

    x_features = df_merged.drop(  # N806
        columns=[col for col in cols_to_drop_for_x if col in df_merged.columns]
    )

    # Ensure all features are numeric and handle NaNs (feature engineering should ideally handle this)
    x_features = x_features.select_dtypes(include=np.number)  # N806
    if x_features.isnull().any().any():
        logger.warning(
            "NaNs found in feature set x_features for regression. Imputing with mean."
        )
        for col in x_features.columns[x_features.isnull().any()]:
            x_features[col] = x_features[col].fillna(
                x_features[col].mean()
            )  # Or more sophisticated imputation

    if x_features.empty:
        logger.error(
            "Feature set x_features is empty after pre-processing for regression."
        )
        return None, None, None

    engine_ids_for_split = df_merged["engine_id"]  # For GroupKFold

    logger.info(
        f"Regression features (x_features) shape: {x_features.shape}, Target (y_target) shape: {y_target.shape}"
    )
    return x_features, y_target, engine_ids_for_split


def split_data_grouped_regression(
    x_data: pd.DataFrame,  # N803
    y_data: pd.Series,
    engine_ids: pd.Series,
    n_splits: int = 5,
) -> list[tuple[Any, Any]]:  # UP006
    """
    Splits data for regression using GroupKFold based on engine_id.
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(x_data, y_data, groups=engine_ids))
    logger.info(
        f"Generated {len(splits)} regression splits using GroupKFold based on engine_id."
    )
    return splits
