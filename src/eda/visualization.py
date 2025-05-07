import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_sensor_trends(
    df: pd.DataFrame,
    engine_id: int,
    sensor_cols: list[str],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot the trend of multiple sensors for a specific engine over time.

    Args:
        df: DataFrame containing engine data
        engine_id: ID of the engine to plot
        sensor_cols: List of sensor columns to plot
        save_path: Path to save the figure (optional)
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib Figure object
    """
    # Filter data for the specific engine
    engine_data = df[df["engine_id"] == engine_id].sort_values("cycle")

    if len(engine_data) == 0:
        logging.warning(f"No data found for engine_id {engine_id}")
        return plt.figure()

    # Determine number of rows and columns for subplots
    n_sensors = len(sensor_cols)
    n_cols = min(3, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = np.array([axes])  # Make sure axes is always iterable
    axes = axes.flatten()

    for i, sensor in enumerate(sensor_cols):
        if sensor not in engine_data.columns:
            continue

        ax = axes[i]
        ax.plot(engine_data["cycle"], engine_data[sensor], marker="o", linestyle="-")
        ax.set_title(f"{sensor} Trend for Engine {engine_id}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(sensor)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved sensor trends plot to {save_path}")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Create a heatmap of correlations between the specified columns.

    Args:
        df: DataFrame containing engine data
        columns: List of columns to include in correlation analysis
        save_path: Path to save the figure (optional)
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib Figure object
    """
    # Ensure all columns exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns]

    if not valid_columns:
        logging.warning("None of the specified columns exist in the DataFrame")
        return plt.figure()

    # Calculate correlation matrix
    corr_matrix = df[valid_columns].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(
        np.ones_like(corr_matrix, dtype=bool)
    )  # Create mask for upper triangle

    # Use seaborn heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    plt.title("Sensor Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved correlation heatmap to {save_path}")

    return fig


def plot_sensor_distributions(
    df: pd.DataFrame,
    sensor_cols: list[str],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot distributions of sensor values across all engines.

    Args:
        df: DataFrame containing engine data
        sensor_cols: List of sensor columns to plot
        save_path: Path to save the figure (optional)
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib Figure object
    """
    valid_sensors = [col for col in sensor_cols if col in df.columns]

    if not valid_sensors:
        logging.warning("None of the specified sensors exist in the DataFrame")
        return plt.figure()

    # Determine grid layout
    n_sensors = len(valid_sensors)
    n_cols = min(3, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, sensor in enumerate(valid_sensors):
        ax = axes[i]
        sns.histplot(df[sensor], kde=True, ax=ax)
        ax.set_title(f"Distribution of {sensor}")
        ax.set_xlabel(sensor)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved sensor distributions plot to {save_path}")

    return fig


def plot_sensor_degradation_patterns(
    df: pd.DataFrame,
    sensor_cols: list[str],
    normalize: bool = True,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot sensor values over normalized cycle time to visualize degradation patterns.

    Args:
        df: DataFrame containing engine data
        sensor_cols: List of sensor columns to plot
        normalize: Whether to normalize cycle to [0,1] range
        n_samples: Number of random engines to sample for plotting
        save_path: Path to save the figure (optional)
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib Figure object
    """
    valid_sensors = [col for col in sensor_cols if col in df.columns]

    if not valid_sensors:
        logging.warning("None of the specified sensors exist in the DataFrame")
        return plt.figure()

    # Sample a few engines for visualization
    unique_engines = df["engine_id"].unique()
    if len(unique_engines) > n_samples:
        sample_engines = np.random.choice(unique_engines, n_samples, replace=False)
    else:
        sample_engines = unique_engines

    # Create subplots
    n_sensors = len(valid_sensors)
    n_cols = min(3, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, sensor in enumerate(valid_sensors):
        ax = axes[i]

        for engine_id in sample_engines:
            engine_data = df[df["engine_id"] == engine_id].copy()

            if normalize:
                # Normalize cycle to [0,1] range for each engine
                max_cycle = engine_data["cycle"].max()
                engine_data["norm_cycle"] = engine_data["cycle"] / max_cycle
                x_col = "norm_cycle"
                x_label = "Normalized Cycle"
            else:
                x_col = "cycle"
                x_label = "Cycle"

            ax.plot(
                engine_data[x_col],
                engine_data[sensor],
                alpha=0.7,
                label=f"Engine {engine_id}",
            )

        ax.set_title(f"{sensor} Degradation Pattern")
        ax.set_xlabel(x_label)
        ax.set_ylabel(sensor)
        ax.grid(True, alpha=0.3)

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved degradation patterns plot to {save_path}")

    return fig
