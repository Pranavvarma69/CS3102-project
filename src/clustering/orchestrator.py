import logging
import os
import random  # For setting seed for subsampling
from typing import Any, Callable, Optional, TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.clustering.algorithms import (
    apply_agglomerative_clustering,
    apply_kmeans_clustering,
)
from src.clustering.evaluation import (
    calculate_clustering_metrics,
    get_stage_statistics,
)
from src.clustering.label_processing import map_clusters_to_stages_by_rul
from src.clustering.visualization import (
    plot_cluster_distributions_pca,
    plot_sensor_trends_with_stages,
)
from src.feature_engineering.window_statistics import add_remaining_useful_life
from src.utils import (
    IMAGE_DIR,
    SENSOR_MEASUREMENTS,
)

logger = logging.getLogger(__name__)

MAX_SAMPLES_AGGLOMERATIVE = 30000  # Threshold for subsampling
SUBSAMPLE_RANDOM_SEED = 42  # For reproducibility of subsampling


# Define TypedDicts for better type checking
class AlgorithmConfig(TypedDict):
    cluster_func: Callable[[pd.DataFrame, int], pd.Series]
    raw_label_col: str
    stage_col: str


class AlgoRunStats(TypedDict, total=False):
    plot_paths_pca: Optional[str]
    plot_paths_sensor_trends_samples: list[str]
    notes: str
    silhouette: float
    davies_bouldin: float
    stage_distribution: list[dict[str, Any]]


class DatasetStats(TypedDict):
    dataset_name: str
    num_stages_configured: int
    algorithms: dict[str, AlgoRunStats]


def run_clustering_for_dataset(
    df_train_raw: pd.DataFrame,
    dataset_name: str,
    num_stages: int = 5,
    n_sample_engines_for_plot: int = 3,
) -> tuple[pd.DataFrame, DatasetStats]:
    logger.info(f"--- Starting Clustering Pipeline for Dataset: {dataset_name} ---")

    all_dataset_stats: DatasetStats = {
        "dataset_name": dataset_name,
        "num_stages_configured": num_stages,
        "algorithms": {},
    }

    active_sensor_cols = [
        col for col in SENSOR_MEASUREMENTS if col in df_train_raw.columns
    ]
    if not active_sensor_cols:
        logger.error(f"No sensor columns for {dataset_name}. Skipping.")
        return pd.DataFrame(), all_dataset_stats

    df_features_for_clustering = df_train_raw[active_sensor_cols].copy()
    if df_features_for_clustering.isnull().any().any():
        df_features_for_clustering = df_features_for_clustering.fillna(
            df_features_for_clustering.mean()
        )

    scaler = StandardScaler()
    df_features_scaled = pd.DataFrame(
        scaler.fit_transform(df_features_for_clustering),
        columns=active_sensor_cols,
        index=df_features_for_clustering.index,
    )

    df_full_with_meta = df_train_raw[["engine_id", "cycle"] + active_sensor_cols].copy()
    df_full_with_meta = add_remaining_useful_life(df_full_with_meta)

    algorithms_to_run: dict[str, AlgorithmConfig] = {
        "kmeans": {
            "cluster_func": apply_kmeans_clustering,
            "raw_label_col": "kmeans_raw_labels",
            "stage_col": "kmeans_stage",
        },
        "agglomerative": {
            "cluster_func": apply_agglomerative_clustering,
            "raw_label_col": "agglomerative_raw_labels",
            "stage_col": "agglomerative_stage",
        },
    }

    for algo_name, config in algorithms_to_run.items():
        logger.info(f"--- Running Clustering with {algo_name} for {dataset_name} ---")
        algo_stats: AlgoRunStats = {
            "plot_paths_pca": None,
            "plot_paths_sensor_trends_samples": [],
        }
        is_subsampled_run = False
        subsample_size_info = None

        current_features_for_algo = df_features_scaled
        meta_for_current_algo_processing = df_full_with_meta.copy()

        if (
            algo_name == "agglomerative"
            and df_features_scaled.shape[0] > MAX_SAMPLES_AGGLOMERATIVE
        ):
            is_subsampled_run = True
            subsample_size_info = MAX_SAMPLES_AGGLOMERATIVE
            logger.warning(
                f"Dataset {dataset_name} ({df_features_scaled.shape[0]} samples) exceeds "
                f"{MAX_SAMPLES_AGGLOMERATIVE} for Agglomerative. Using subsample of {subsample_size_info}."
            )
            algo_stats["notes"] = (
                f"Metrics based on a subsample of {subsample_size_info} points."
            )

            random.seed(SUBSAMPLE_RANDOM_SEED)
            np.random.seed(SUBSAMPLE_RANDOM_SEED)
            sample_indices = np.random.choice(
                df_features_scaled.index, size=subsample_size_info, replace=False
            )
            current_features_for_algo = df_features_scaled.loc[sample_indices]
            meta_for_current_algo_processing = df_full_with_meta.loc[
                sample_indices
            ].copy()

        # Apply clustering (on full or subsampled features)
        # Changed: Passing num_stages as a positional argument (line 158 from original error report)
        raw_labels = config["cluster_func"](current_features_for_algo, num_stages)
        meta_for_current_algo_processing[config["raw_label_col"]] = raw_labels

        meta_for_current_algo_processing[config["stage_col"]] = (
            map_clusters_to_stages_by_rul(
                meta_for_current_algo_processing,
                raw_cluster_col=config["raw_label_col"],
                rul_col="RUL",
                num_stages=num_stages,
            )
        )

        # Calculate metrics (on full or subsampled features & labels)
        quality_metrics = calculate_clustering_metrics(
            current_features_for_algo,
            meta_for_current_algo_processing[config["raw_label_col"]],
        )
        # Changed: Individual assignment instead of algo_stats.update() (line 180 from original error report)
        algo_stats["silhouette"] = quality_metrics["silhouette"]
        algo_stats["davies_bouldin"] = quality_metrics["davies_bouldin"]

        stage_distribution_stats_df = get_stage_statistics(
            meta_for_current_algo_processing,
            stage_col=config["stage_col"],
            rul_col="RUL",
        )
        algo_stats["stage_distribution"] = stage_distribution_stats_df.to_dict(
            orient="records"
        )

        if is_subsampled_run:
            df_full_with_meta[config["stage_col"]] = pd.NA
        else:
            df_full_with_meta[config["stage_col"]] = meta_for_current_algo_processing[
                config["stage_col"]
            ]

        pca_plot_filename = f"{dataset_name}_{algo_name}_pca_clusters.png"
        plot_cluster_distributions_pca(
            current_features_for_algo,
            meta_for_current_algo_processing[config["stage_col"]],
            dataset_name,
            algo_name,
            IMAGE_DIR,
            is_subsample=is_subsampled_run,
            subsample_size=subsample_size_info,
        )
        algo_stats["plot_paths_pca"] = os.path.join(
            IMAGE_DIR, f"clustering/{dataset_name}", pca_plot_filename
        )

        unique_engines = df_full_with_meta["engine_id"].unique()
        sample_engine_ids_for_plot = np.random.choice(
            unique_engines,
            min(n_sample_engines_for_plot, len(unique_engines)),
            replace=False,
        )
        current_algo_sensor_plots: list[str] = []
        for engine_id in sample_engine_ids_for_plot:
            sensor_plot_filename = (
                f"{dataset_name}_engine_{engine_id}_{algo_name}_sensor_stage_trends.png"
            )
            plot_sensor_trends_with_stages(
                df_full_with_meta,
                engine_id,
                active_sensor_cols[:6],
                config["stage_col"],
                dataset_name,
                algo_name,
                IMAGE_DIR,
                is_subsample_algo=is_subsampled_run,
            )
            current_algo_sensor_plots.append(
                os.path.join(
                    IMAGE_DIR, f"clustering/{dataset_name}", sensor_plot_filename
                )
            )
        algo_stats["plot_paths_sensor_trends_samples"] = current_algo_sensor_plots
        all_dataset_stats["algorithms"][algo_name] = algo_stats
        logger.info(f"Finished processing {algo_name} for {dataset_name}.")

    output_cols = (
        ["engine_id", "cycle"]
        + active_sensor_cols
        + [config["stage_col"] for config in algorithms_to_run.values()]
        + ["RUL"]
    )
    final_output_cols_exist = [
        col for col in output_cols if col in df_full_with_meta.columns
    ]
    df_final_output = df_full_with_meta[final_output_cols_exist].copy()

    logger.info(f"--- Clustering Pipeline for Dataset: {dataset_name} Completed ---")
    return df_final_output, all_dataset_stats
