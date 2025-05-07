import logging
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _prepare_pca_plot_data(
    df: pd.DataFrame,
    pca_col_x: str,
    pca_col_y: str,
    hue_col: Optional[str],
    sample_frac: float,
    random_state: Optional[int],
) -> tuple[
    Optional[pd.DataFrame], Optional[str]
]:  # Use lowercase tuple for Python 3.9+
    """
    Validates PCA columns, samples data if needed, and determines the active hue column.
    Returns a tuple of (DataFrame_for_plotting, active_hue_column_name).
    Returns (None, None) if essential PCA columns are missing.
    """
    if not (pca_col_x in df.columns and pca_col_y in df.columns):
        logger.error(
            f"PCA columns '{pca_col_x}' or '{pca_col_y}' not found in DataFrame."
        )
        return None, None

    plot_df = df
    # Apply sampling only if sample_frac is meaningful and df is large enough
    if (
        0.0 < sample_frac < 1.0 and len(df) > 1000
    ):  # Reduced threshold for sampling demo
        plot_df = df.sample(frac=sample_frac, random_state=random_state)
        logger.info(
            f"Plotting a sample of {len(plot_df)} points ({sample_frac * 100:.1f}%) for PCA visualization."
        )

    active_hue_col: Optional[str] = None
    if hue_col:
        if hue_col in plot_df.columns:
            if plot_df[hue_col].nunique() > 1:
                active_hue_col = hue_col
            else:
                logger.info(
                    f"Hue column '{hue_col}' has only one unique value. Plotting without hue."
                )
        else:
            logger.warning(
                f"Hue column '{hue_col}' not found in DataFrame. Plotting without hue."
            )
    return plot_df, active_hue_col


def _style_pca_plot_legend(
    ax: plt.Axes, plot_df: pd.DataFrame, active_hue_col: Optional[str]
) -> bool:
    """
    Styles the legend of the PCA plot (title, visibility, position).
    Returns True if the legend is visible and positioned outside the plot, False otherwise.
    """
    legend_is_outside_and_visible = False
    legend_obj = ax.get_legend()

    if legend_obj and active_hue_col:  # Legend exists and hue was used
        num_unique_hue = plot_df[active_hue_col].nunique()

        if num_unique_hue > 20 and num_unique_hue < 150:
            legend_obj.set_title(f"{active_hue_col} (sample of {num_unique_hue})")
        elif num_unique_hue >= 150:
            legend_obj.set_visible(False)
            logger.info(
                f"Legend for '{active_hue_col}' hidden due to {num_unique_hue} unique values."
            )
        else:  # Fewer than ~20 unique values
            legend_obj.set_title(active_hue_col)

        if legend_obj.get_visible():
            plt.setp(
                legend_obj.get_texts(), fontsize="x-small"
            )  # Make legend text smaller
            legend_obj.set_bbox_to_anchor((1.02, 1.0))  # Position outside
            legend_obj.set_loc("upper left")
            legend_is_outside_and_visible = True
            logger.info(
                f"Styled legend for '{active_hue_col}'. Moved outside plot area."
            )

    return legend_is_outside_and_visible


def plot_pca_components(
    df: pd.DataFrame,
    pca_col_x: str = "pca_1",
    pca_col_y: str = "pca_2",
    hue_col: Optional[str] = None,
    sample_frac: float = 0.1,
    random_state: Optional[int] = 42,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Creates a scatter plot of PCA components, orchestrating data prep and styling.

    Args:
        df: DataFrame containing PCA components.
        pca_col_x: Name of the column for the first PCA component.
        pca_col_y: Name of the column for the second PCA component.
        hue_col: Optional column name to use for coloring points.
        sample_frac: Fraction of data to sample for plotting if df is large.
        random_state: Random state for sampling.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_df, active_hue_col = _prepare_pca_plot_data(
        df, pca_col_x, pca_col_y, hue_col, sample_frac, random_state
    )

    if plot_df is None:  # Indicates PCA columns were not found
        ax.text(
            0.5,
            0.5,
            f"PCA columns not found:\n{pca_col_x}, {pca_col_y}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            color="red",
        )
        return fig  # Return figure with error message

    # Perform the scatter plot
    sns.scatterplot(
        x=pca_col_x,
        y=pca_col_y,
        hue=plot_df[active_hue_col] if active_hue_col else None,
        data=plot_df,
        ax=ax,
        alpha=0.7,
        linewidth=0.5,
        edgecolor="w",
        legend="auto" if active_hue_col else False,
    )

    # Set basic plot aesthetics
    plot_title = f"2D PCA Components ({pca_col_x} vs {pca_col_y})"
    if active_hue_col:
        plot_title += f" (Hued by {active_hue_col})"
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel(pca_col_x, fontsize=12)
    ax.set_ylabel(pca_col_y, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Style the legend
    legend_moved_outside = _style_pca_plot_legend(ax, plot_df, active_hue_col)

    # Adjust layout, especially if legend was moved outside
    rect_val: Optional[tuple[float, float, float, float]] = None
    if legend_moved_outside:
        rect_val = (0.0, 0.0, 0.83, 1.0)  # Adjust right boundary for external legend

    plt.tight_layout(rect=rect_val)

    return fig
