import logging
import os
from typing import Optional  # Removed List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import IMAGE_DIR, ensure_dir

logger = logging.getLogger(__name__)


def plot_risk_score_trends(
    df_with_scores: pd.DataFrame,
    risk_score_col: str,
    dataset_name: str,
    config_identifier: str,  # To uniquely identify this run/config
    report_image_subfolder: str,  # e.g., "risk_assessment/FD001/risk_run_XYZ" (relative to IMAGE_DIR)
    engine_ids_to_plot: Optional[list[int]] = None,
    num_sample_engines: int = 3,
    figsize: tuple[int, int] = (15, 7),
) -> list[str]:
    """
    Plots the risk score over cycles for selected or sample engines.

    Args:
        df_with_scores: DataFrame with 'engine_id', 'cycle', and risk_score_col.
        risk_score_col: Name of the risk score column to plot.
        dataset_name: Name of the dataset (e.g., "FD001"). (Used for IMAGE_DIR structure, but report_image_subfolder is now more direct)
        config_identifier: Unique string for this configuration run.
        report_image_subfolder: Specific subfolder path relative to IMAGE_DIR where plots will be saved.
                                Example: "risk_assessment/FD001/risk_run_XYZ_timestamp".
        engine_ids_to_plot: Specific list of engine IDs to plot. If None, samples.
        num_sample_engines: Number of engines to sample if engine_ids_to_plot is None.
        figsize: Figure size.

    Returns:
        List of paths to saved plots.
    """
    saved_plot_paths: list[str] = []
    if risk_score_col not in df_with_scores.columns:
        logger.error(f"Risk score column '{risk_score_col}' not found for plotting.")
        return saved_plot_paths
    if (
        "engine_id" not in df_with_scores.columns
        or "cycle" not in df_with_scores.columns
    ):
        logger.error("'engine_id' or 'cycle' column not found for plotting.")
        return saved_plot_paths

    ids_to_plot_final: list[int]  # Declare type for the variable that will hold the IDs

    if engine_ids_to_plot is None:
        unique_engines = df_with_scores["engine_id"].unique()
        if len(unique_engines) == 0:
            logger.warning(
                f"No engines found in data for {dataset_name} to plot risk trends."
            )
            return saved_plot_paths
        sample_size = min(num_sample_engines, len(unique_engines))
        # Ensure ids_to_plot_final is a list[int]
        ids_to_plot_final = np.random.choice(
            unique_engines, sample_size, replace=False
        ).tolist()
        logger.info(
            f"Sampling {sample_size} engines for risk score trend plots: {ids_to_plot_final}"
        )
    else:
        ids_to_plot_final = engine_ids_to_plot
        logger.info(
            f"Plotting risk score trends for specified engines: {ids_to_plot_final}"
        )

    base_plot_dir = os.path.join(IMAGE_DIR, report_image_subfolder)
    ensure_dir(base_plot_dir)

    for engine_id in ids_to_plot_final:  # Iterate over the correctly typed list
        engine_data = df_with_scores[
            df_with_scores["engine_id"] == engine_id
        ].sort_values("cycle")
        if engine_data.empty:
            logger.warning(f"No data for engine {engine_id} to plot risk score trend.")
            continue

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            engine_data["cycle"],
            engine_data[risk_score_col],
            marker=".",
            linestyle="-",
            label=risk_score_col,
        )

        ax.set_title(
            f"Risk Score Trend for Engine {engine_id} ({dataset_name}) - {config_identifier}"
        )
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Risk Score")
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.tight_layout()

        plot_filename = f"risk_trend_engine_{engine_id}_{config_identifier}.png"
        full_save_path = os.path.join(base_plot_dir, plot_filename)

        try:
            fig.savefig(full_save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Risk score trend plot saved to {full_save_path}")
            saved_plot_paths.append(full_save_path)
        except Exception as e:
            logger.error(f"Failed to save risk score trend plot: {e}")
        finally:
            plt.close(fig)

    return saved_plot_paths
