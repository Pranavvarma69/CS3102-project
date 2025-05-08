import logging
from typing import Optional  # Changed from 'from typing import List, Optional'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# from sklearn.preprocessing import StandardScaler # Not used directly here
from src.utils import save_plot

logger = logging.getLogger(__name__)


def plot_cluster_distributions_pca(
    df_features_scaled: pd.DataFrame,
    stage_labels: pd.Series,  # This might be from a subsample
    dataset_name: str,
    algorithm_name: str,
    image_dir_base: str,  # e.g., IMAGE_DIR from utils
    n_components: int = 2,
    figsize: tuple[int, int] = (10, 8),
    is_subsample: bool = False,  # Flag to indicate if data is from a subsample
    subsample_size: Optional[int] = None,
) -> None:
    logger.info(
        f"Plotting PCA of clusters for {dataset_name} using {algorithm_name}."
        f"{' (on subsample)' if is_subsample else ''}"
    )
    if df_features_scaled.empty or stage_labels.empty:
        logger.warning("Data for PCA plot is empty. Skipping plot.")
        return
    if len(df_features_scaled) != len(
        stage_labels
    ):  # Should match if subsampled correctly
        logger.error(
            "Mismatch in length between features and labels. Skipping PCA plot."
        )
        return

    if stage_labels.isna().all():
        logger.warning(
            f"All stage labels are NA for {algorithm_name} on {dataset_name}. Skipping PCA plot."
        )
        # Optionally, could plot without hue if desired:
        # sns.scatterplot(x="PC1", y="PC2", data=pca_df, alpha=0.7, ax=ax, legend=False)
        return

    pca = PCA(n_components=n_components)
    try:
        principal_components = pca.fit_transform(df_features_scaled)
    except Exception as e:
        logger.error(f"PCA failed: {e}. Skipping PCA plot.")
        return

    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f"PC{i + 1}" for i in range(n_components)],
        index=df_features_scaled.index,  # Use index from df_features_scaled
    )
    # Align stage_labels with pca_df (important if stage_labels came from a differently indexed subsample,
    # but orchestrator should ensure they align if subsampling is done for both features and meta)
    pca_df["stage"] = (
        stage_labels.loc[pca_df.index]
        if not stage_labels.index.equals(pca_df.index)
        else stage_labels
    )

    fig, ax = plt.subplots(figsize=figsize)
    title_suffix = (
        f"{' (Subsample N=' + str(subsample_size) + ')' if is_subsample else ''}"
    )
    plot_title = (
        f"Clusters ({algorithm_name}) by PCA - {dataset_name}{title_suffix}\n"
        f"Explained Variance: {pca.explained_variance_ratio_[:n_components].sum():.2%}"
    )

    # Filter out NA stages for plotting if any exist but not all are NA
    plot_data_filtered = pca_df.dropna(subset=["stage"])
    if plot_data_filtered.empty:
        logger.warning(
            f"No non-NA stage labels to plot for PCA ({algorithm_name}, {dataset_name}). Skipping."
        )
        plt.close(fig)
        return

    unique_stages_to_plot = plot_data_filtered["stage"].nunique()
    palette = sns.color_palette("viridis", n_colors=max(1, unique_stages_to_plot))

    sns.scatterplot(
        x="PC1",
        y="PC2" if n_components >= 2 else "PC1",
        hue="stage",
        palette=palette,
        data=plot_data_filtered,  # Use filtered data
        legend="full",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title(plot_title)
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})")
    if n_components >= 2:
        ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.grid(True, alpha=0.3)

    if unique_stages_to_plot > 10:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:10], labels[:10], title="Stage (Sample)")
    elif unique_stages_to_plot > 0:  # only add legend if there's something to show
        ax.legend(title="Stage")
    else:  # no legend if no unique stages
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plot_filename = f"{dataset_name}_{algorithm_name}_pca_clusters.png"
    save_plot(
        fig, f"clustering/{dataset_name}", plot_filename
    )  # save_plot is from src.utils
    logger.info(
        f"Saved PCA cluster plot: {plot_filename} to images/clustering/{dataset_name}/"
    )
    plt.close(fig)


def plot_sensor_trends_with_stages(
    df_full: pd.DataFrame,
    engine_id_to_plot: int,
    sensor_cols: list[str],  # Changed from List[str] to list[str]
    stage_col: str,
    dataset_name: str,
    algorithm_name: str,
    image_dir_base: str,  # e.g., IMAGE_DIR from utils
    figsize: tuple[int, int] = (18, 12),
    is_subsample_algo: bool = False,  # Indicates if this algorithm's stages might be NA due to subsampling
) -> None:
    engine_data = df_full[df_full["engine_id"] == engine_id_to_plot].sort_values(
        "cycle"
    )
    if engine_data.empty:
        logger.warning(
            f"No data for engine {engine_id_to_plot} in {dataset_name} to plot sensor trends."
        )
        return

    logger.info(
        f"Plotting sensor trends for engine {engine_id_to_plot} ({dataset_name}, {algorithm_name} stages)."
        f"{' (Algorithm stages might be NA due to subsampling)' if is_subsample_algo and engine_data[stage_col].isna().any() else ''}"
    )

    n_sensors = len(sensor_cols)
    n_cols_plot = min(3, n_sensors)
    n_rows_plot = (n_sensors + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=figsize, sharex=True)
    axes = np.array(axes).flatten()

    # Handle stages for hue
    valid_stages_data = engine_data.dropna(subset=[stage_col])
    hue_actual = stage_col if not valid_stages_data.empty else None

    num_unique_stages = 0
    palette = None
    if hue_actual:
        num_unique_stages = valid_stages_data[stage_col].nunique()
        if num_unique_stages > 0:
            palette = sns.color_palette("viridis", n_colors=max(1, num_unique_stages))
        else:  # All are NA or only one value after NA drop
            hue_actual = None

    for i, sensor in enumerate(sensor_cols):
        if sensor not in engine_data.columns:
            logger.warning(
                f"Sensor {sensor} not found for plotting engine {engine_id_to_plot}."
            )
            if i < len(axes):
                axes[i].set_visible(False)
            continue

        ax = axes[i]
        sns.scatterplot(
            x="cycle",
            y=sensor,
            hue=hue_actual,  # Use stage_col only if there are valid stages to color by
            palette=palette,
            data=engine_data,  # Plot all data; hue will only apply to non-NA stages
            ax=ax,
            legend=(
                i == 0 and hue_actual is not None and num_unique_stages > 0
            ),  # Legend only on first plot if hue is active
        )
        ax.set_title(f"{sensor} (Engine {engine_id_to_plot})")
        ax.set_ylabel(sensor)
        ax.grid(True, alpha=0.3)

    if (
        n_sensors > 0
        and hue_actual
        and num_unique_stages > 0
        and axes[0].get_legend() is not None
    ):
        axes[0].legend(
            title=f"Stage ({algorithm_name})",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
    elif n_sensors > 0 and axes[0].get_legend() is not None:  # Remove legend if no hue
        axes[0].get_legend().remove()

    for j in range(n_sensors, len(axes)):
        axes[j].set_visible(False)

    title_suffix = (
        f"{' (Stages based on subsample for this algo)' if is_subsample_algo else ''}"
    )
    fig.suptitle(
        f"Sensor Trends by Stage ({algorithm_name}) - Eng {engine_id_to_plot}, {dataset_name}{title_suffix}",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout(
        rect=(
            0,
            0,
            0.9 if hue_actual and num_unique_stages > 0 else 1.0,
            1,
        )  # Changed list to tuple for rect
    )  # Adjust for legend

    plot_filename = f"{dataset_name}_engine_{engine_id_to_plot}_{algorithm_name}_sensor_stage_trends.png"
    save_plot(
        fig, f"clustering/{dataset_name}", plot_filename
    )  # save_plot is from src.utils
    logger.info(
        f"Saved sensor trends with stages plot: {plot_filename} to images/clustering/{dataset_name}/"
    )
    plt.close(fig)
