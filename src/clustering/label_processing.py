# src/clustering/label_processing.py
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def map_clusters_to_stages_by_rul(
    df_with_labels: pd.DataFrame,
    raw_cluster_col: str,
    rul_col: str = "RUL",
    num_stages: int = 5,
) -> pd.Series:
    """
    Maps raw cluster labels to ordered degradation stages (0 to num_stages-1)
    based on the average RUL within each cluster.

    Stage 0: Normal (highest average RUL)
    Stage num_stages-1: Failure (lowest average RUL)

    Args:
        df_with_labels: DataFrame containing the raw cluster labels and RUL column.
        raw_cluster_col: Name of the column with raw cluster labels.
        rul_col: Name of the RUL column.
        num_stages: The number of degradation stages (typically 5).

    Returns:
        A pandas Series with ordered stage labels.
    """
    if raw_cluster_col not in df_with_labels.columns:
        logger.error(f"Raw cluster column '{raw_cluster_col}' not found in DataFrame.")
        raise ValueError(
            f"Raw cluster column '{raw_cluster_col}' not found in DataFrame."
        )
    if rul_col not in df_with_labels.columns:
        logger.error(f"RUL column '{rul_col}' not found in DataFrame.")
        raise ValueError(f"RUL column '{rul_col}' not found in DataFrame.")

    logger.info(
        f"Mapping clusters from '{raw_cluster_col}' to {num_stages} stages using RUL from '{rul_col}'."
    )

    # Calculate average RUL for each raw cluster
    cluster_avg_rul = (
        df_with_labels.groupby(raw_cluster_col)[rul_col]
        .mean()
        .sort_values(ascending=False)
    )  # Highest RUL first

    if len(cluster_avg_rul) < num_stages:
        logger.warning(
            f"Number of unique clusters ({len(cluster_avg_rul)}) is less than "
            f"desired stages ({num_stages}). Stages will be assigned based on available clusters."
        )
    # Ensure num_stages doesn't exceed actual number of clusters
    # This creates a mapping from raw_cluster_id -> ordered_stage_label
    stage_mapping = {
        cluster_id: stage for stage, cluster_id in enumerate(cluster_avg_rul.index)
    }

    # Handle cases where n_clusters < num_stages by assigning available stages
    # or if n_clusters > num_stages, some higher stage numbers might not be assigned
    # if we strictly map. This ensures all points get a stage based on their cluster's rank.

    # Map raw cluster labels to ordered stage labels
    ordered_stage_labels = df_with_labels[raw_cluster_col].map(stage_mapping)

    # Ensure all stages are within 0 to num_stages-1, mostly for safety.
    # If k-means produces less than num_stages clusters, some stages might be missing.
    # The mapping above handles this by assigning stages 0, 1, ... up to k-1.

    logger.info(f"Cluster to Stage mapping based on RUL ({raw_cluster_col}):")
    for cluster_id, stage in stage_mapping.items():
        logger.info(
            f"  Raw Cluster {cluster_id} (Avg RUL: {cluster_avg_rul[cluster_id]:.2f}) -> Stage {stage}"
        )

    return pd.Series(
        ordered_stage_labels,
        index=df_with_labels.index,
        name=f"{raw_cluster_col.replace('_raw_labels', '')}_stage",
    )
