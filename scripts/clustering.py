import argparse
import logging
import os
import sys
from datetime import datetime  # For timestamp in report filename
from typing import Any

import numpy as np  # Make sure numpy is imported
import pandas as pd

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.clustering.orchestrator import run_clustering_for_dataset  # noqa: E402
from src.data_loading import load_dataset  # noqa: E402
from src.utils import (  # noqa: E402
    DATA_DIR,
    IMAGE_DIR,
    PROCESSED_DATA_DIR,
    ensure_dir,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATASET_NAMES = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_NUM_STAGES = 5


def format_stats_for_report(stats: dict[str, Any]) -> str:  # Changed type hint
    """Formats the statistics dictionary for human-readable report."""
    report_lines = []
    report_lines.append(f"  Dataset Name: {stats.get('dataset_name', 'N/A')}")
    report_lines.append(
        f"  Num Stages Configured: {stats.get('num_stages_configured', 'N/A')}"
    )

    output_parquet = stats.get("output_parquet_path", "N/A")
    report_lines.append(f"  Output Parquet: {output_parquet}")

    # Check for status and error message (optional, as they might not always be set if no error)
    status = stats.get("status")
    if (
        status and status != "SUCCESS"
    ):  # Only show status if it's not a plain success, or always show
        report_lines.append(f"  Status: {status}")
        error_msg = stats.get("error_message")
        if error_msg:
            report_lines.append(f"    Error Details: {error_msg}")

    report_lines.append("  Algorithm Details:")
    for algo_name, algo_data in stats.get("algorithms", {}).items():
        report_lines.append(f"    Algorithm: {algo_name.upper()}")

        if "notes" in algo_data:
            report_lines.append(f"      Note: {algo_data['notes']}")

        silhouette_val = algo_data.get("silhouette")
        davies_bouldin_val = algo_data.get("davies_bouldin")

        formatted_silhouette = (
            f"{silhouette_val:.4f}"
            if silhouette_val is not None and not np.isnan(silhouette_val)
            else "N/A"
        )
        formatted_davies_bouldin = (
            f"{davies_bouldin_val:.4f}"
            if davies_bouldin_val is not None and not np.isnan(davies_bouldin_val)
            else "N/A"
        )

        report_lines.append(f"      Silhouette Score: {formatted_silhouette}")
        report_lines.append(f"      Davies-Bouldin Score: {formatted_davies_bouldin}")

        report_lines.append(f"      PCA Plot: {algo_data.get('plot_paths_pca', 'N/A')}")
        report_lines.append(
            f"      Sample Sensor Trend Plots: {len(algo_data.get('plot_paths_sensor_trends_samples', []))} generated"
        )

        report_lines.append("      Stage Distribution & RUL Stats:")
        stage_dist = algo_data.get("stage_distribution", [])
        if stage_dist:
            try:
                df_dist = pd.DataFrame(stage_dist)
                cols_to_print = [
                    "stage",
                    "count",
                    "percentage",
                    "mean_RUL",
                    "median_RUL",
                    "std_RUL",
                ]
                existing_cols = [col for col in cols_to_print if col in df_dist.columns]

                df_print = df_dist[existing_cols].copy()
                float_format_map = {
                    "percentage": "%.2f%%",  # Add %% for literal %
                    "mean_RUL": "%.2f",
                    "median_RUL": "%.2f",
                    "std_RUL": "%.2f",
                }

                for col, fmt_str in float_format_map.items():
                    if col in df_print.columns:
                        df_print[col] = df_print[col].apply(
                            lambda x: (
                                fmt_str % x if pd.notna(x) else "N/A"  # noqa: B023
                            )
                        )

                dist_str = df_print.to_string(index=False, header=True)

                indented_dist_str = "\n".join(
                    ["        " + line for line in dist_str.split("\n")]
                )
                report_lines.append(indented_dist_str)
            except Exception as e:
                logger.error(
                    f"Error formatting stage distribution for report (try block): {e}"
                )
                # Fallback if DataFrame formatting fails
                for stage_info in stage_dist:
                    stage_s = stage_info.get("stage", "N/A")
                    count_s = stage_info.get("count", "N/A")

                    perc_raw = stage_info.get("percentage")
                    # Ensure '%' is literal if perc_raw is a number, handle NA
                    perc_s = f"{perc_raw:.2f}%" if pd.notna(perc_raw) else "N/A"

                    mean_rul_raw = stage_info.get("mean_RUL")
                    mean_rul_s = (
                        f"{mean_rul_raw:.2f}" if pd.notna(mean_rul_raw) else "N/A"
                    )

                    median_rul_raw = stage_info.get("median_RUL")
                    median_rul_s = (
                        f"{median_rul_raw:.2f}" if pd.notna(median_rul_raw) else "N/A"
                    )

                    std_rul_raw = stage_info.get("std_RUL")
                    std_rul_s = f"{std_rul_raw:.2f}" if pd.notna(std_rul_raw) else "N/A"

                    report_lines.append(
                        f"        Stage {stage_s}: Count={count_s} ({perc_s}), "
                        f"Mean RUL={mean_rul_s}, Median RUL={median_rul_s}, Std RUL={std_rul_s}"
                    )
        else:
            report_lines.append(
                "        No stage distribution data available or all stages were NA."
            )
        report_lines.append("    " + "-" * 20)
    return "\n".join(report_lines)


def main(
    dataset_names_to_process: list[str], num_stages: int
) -> None:  # Changed type hint
    """
    Main script to run the clustering pipeline for specified CMAPSS datasets.
    Generates custom degradation stage labels using clustering and a summary report.
    """
    run_timestamp = datetime.now()
    logger.info(
        f"--- Starting Multi-Stage Failure Labeling (Clustering Phase) at {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')} ---"
    )

    clustered_labels_dir = os.path.join(PROCESSED_DATA_DIR, "clustered_labels")
    ensure_dir(clustered_labels_dir)
    clustering_image_main_dir = os.path.join(IMAGE_DIR, "clustering")
    ensure_dir(clustering_image_main_dir)

    reports_dir = os.path.join(BASE_DIR, "reports", "clustering")
    ensure_dir(reports_dir)
    summary_report_filename = (
        f"clustering_run_summary_{run_timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    summary_report_path = os.path.join(reports_dir, summary_report_filename)

    overall_report_content = [
        f"Clustering Pipeline Run Summary - {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Datasets Processed: {', '.join(dataset_names_to_process)}",
        f"Number of Stages Target: {num_stages}\n{'=' * 40}\n",
    ]

    for dataset_name in dataset_names_to_process:
        logger.info(f"Processing dataset: {dataset_name}")
        ensure_dir(os.path.join(clustering_image_main_dir, dataset_name))

        train_filename = f"train_{dataset_name}.txt"
        try:
            df_train_raw = load_dataset(train_filename, data_dir=DATA_DIR)
        except Exception as e:
            logger.error(
                f"Failed to load training data for {dataset_name}: {e}. Skipping."
            )
            overall_report_content.append(
                f"Dataset: {dataset_name}\n  Status: FAILED (Load Error)\n  Error: {e}\n{'-' * 40}\n"
            )
            continue

        # This dictionary will hold the original stats plus any new keys we add
        reportable_stats: dict[str, Any]

        try:
            df_clustered_output, dataset_stats_original = run_clustering_for_dataset(
                df_train_raw.copy(),
                dataset_name,
                num_stages=num_stages,
                n_sample_engines_for_plot=3,
            )
            # Initialize reportable_stats with a copy of the original stats
            reportable_stats = {**dataset_stats_original}

        except Exception as e:
            logger.error(
                f"Clustering pipeline failed critically for {dataset_name}: {e}. Skipping."
            )
            # If pipeline fails here, dataset_stats_original might not exist.
            # We create a minimal reportable_stats for failure reporting.
            reportable_stats = {
                "dataset_name": dataset_name,
                "num_stages_configured": num_stages,
                "status": "FAILED (Pipeline Error)",
                "error_message": str(e),
                "algorithms": {},  # Ensure algorithms key exists for format_stats_for_report
            }
            overall_report_content.append(format_stats_for_report(reportable_stats))
            overall_report_content.append(f"{'-' * 40}\n")
            continue

        if df_clustered_output.empty:
            logger.warning(
                f"Clustering pipeline returned empty DataFrame for {dataset_name}. No output saved."
            )
            reportable_stats["status"] = "SKIPPED (Empty Output)"
            # No output_parquet_path to set here, format_stats_for_report will handle its absence
            overall_report_content.append(format_stats_for_report(reportable_stats))
            overall_report_content.append(f"{'-' * 40}\n")
            continue

        output_filename = f"{dataset_name}_clustered_all_algos.parquet"
        output_path = os.path.join(clustered_labels_dir, output_filename)
        reportable_stats["output_parquet_path"] = output_path  # Add to our working copy

        try:
            df_clustered_output.to_parquet(output_path, index=False)
            logger.info(
                f"Successfully saved clustered data for {dataset_name} to {output_path}"
            )
            logger.info(
                f"Output Data Head for {dataset_name} (first 3 rows):\n{df_clustered_output.head(3).to_string()}"
            )
            reportable_stats["status"] = "SUCCESS"

        except Exception as e:
            logger.error(
                f"Failed to save clustered data for {dataset_name} to {output_path}: {e}"
            )
            reportable_stats["status"] = "FAILED (Save Error)"
            reportable_stats["error_message"] = str(e)

        overall_report_content.append(format_stats_for_report(reportable_stats))
        overall_report_content.append(f"{'-' * 40}\n")

    try:
        with open(summary_report_path, "w") as f:
            f.write("\n".join(overall_report_content))
        logger.info(f"Clustering run summary saved to: {summary_report_path}")
    except OSError as e:
        logger.error(f"Failed to write summary report: {e}")

    logger.info(
        f"--- Multi-Stage Failure Labeling (Clustering Phase) Complete at {datetime.now().strftime('%Y-%m-%d %H:%M%S')} ---"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Clustering for Multi-Stage Failure Labeling on CMAPSS dataset."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DATASET_NAMES,
        choices=DATASET_NAMES,
        help=(
            "List of dataset identifiers (e.g., FD001 FD003). "
            f"Defaults to all: {DATASET_NAMES}."
        ),
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=DEFAULT_NUM_STAGES,
        help=f"Number of degradation stages to derive. Defaults to {DEFAULT_NUM_STAGES}.",
    )
    args = parser.parse_args()

    main(dataset_names_to_process=args.datasets, num_stages=args.num_stages)
