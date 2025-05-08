import logging

import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)


def calculate_clustering_metrics(
    features_scaled: pd.DataFrame, labels: pd.Series
) -> dict[str, float]:  # Changed from '-> dict' to '-> dict[str, float]'
    """
    Calculates Silhouette Score and Davies-Bouldin Index for clustering results.

    Args:
        features_scaled: DataFrame of scaled features used for clustering.
        labels: Series of cluster labels (raw or mapped stages).

    Returns:
        A dictionary with 'silhouette_score' and 'davies_bouldin_score'.
        Returns NaN for scores if computation fails (e.g., < 2 clusters).
    """
    metrics: dict[str, float] = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
    }  # Added type hint for metrics
    if labels.nunique() < 2:
        logger.warning(
            "Cannot calculate clustering metrics with less than 2 unique labels."
        )
        return metrics
    if len(features_scaled) != len(labels):
        logger.error("Feature and label counts do not match. Cannot calculate metrics.")
        return metrics
    if features_scaled.empty or labels.empty:
        logger.warning("Empty features or labels. Cannot calculate metrics.")
        return metrics

    try:
        metrics["silhouette"] = silhouette_score(features_scaled, labels)
        logger.info(f"Calculated Silhouette Score: {metrics['silhouette']:.4f}")
    except ValueError as e:
        logger.warning(f"Could not calculate Silhouette Score: {e}")
    except Exception as e:
        logger.error(f"Unexpected error calculating Silhouette Score: {e}")

    try:
        metrics["davies_bouldin"] = davies_bouldin_score(features_scaled, labels)
        logger.info(f"Calculated Davies-Bouldin Score: {metrics['davies_bouldin']:.4f}")
    except ValueError as e:
        logger.warning(f"Could not calculate Davies-Bouldin Score: {e}")
    except Exception as e:
        logger.error(f"Unexpected error calculating Davies-Bouldin Score: {e}")

    return metrics


def get_stage_statistics(
    df_with_stages: pd.DataFrame, stage_col: str, rul_col: str = "RUL"
) -> pd.DataFrame:
    """
    Computes statistics for each degradation stage.

    Args:
        df_with_stages: DataFrame containing the stage labels and RUL.
        stage_col: Name of the column with stage labels.
        rul_col: Name of the RUL column.

    Returns:
        A DataFrame with statistics (count, percentage, RUL mean/median/std) for each stage.
    """
    if stage_col not in df_with_stages.columns:
        logger.error(f"Stage column '{stage_col}' not found.")
        return pd.DataFrame()
    if rul_col not in df_with_stages.columns:
        logger.warning(f"RUL column '{rul_col}' not found. RUL stats will be NaN.")
        # Add a dummy RUL column with NaNs if not present to avoid breaking groupby
        df_with_stages = df_with_stages.copy()  # Avoid SettingWithCopyWarning
        df_with_stages[rul_col] = float("nan")

    total_count = len(df_with_stages)
    stage_stats = (
        df_with_stages.groupby(stage_col)
        .agg(
            count=(rul_col, "size"),
            mean_RUL=(rul_col, "mean"),
            median_RUL=(rul_col, "median"),
            std_RUL=(rul_col, "std"),
        )
        .reset_index()
    )  # Ensure stage_col is a column

    stage_stats.rename(columns={stage_col: "stage"}, inplace=True)
    stage_stats["percentage"] = (stage_stats["count"] / total_count) * 100
    stage_stats = stage_stats.sort_values(by="stage")

    logger.info(f"Calculated statistics for stages in column '{stage_col}':")
    for _, row in stage_stats.iterrows():
        logger.info(
            f"  Stage {row['stage']}: Count={row['count']} ({row['percentage']:.2f}%), "
            f"Mean RUL={row['mean_RUL']:.2f}, Median RUL={row['median_RUL']:.2f}"
        )
    return stage_stats
