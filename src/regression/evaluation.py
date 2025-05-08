import logging
import os
from typing import Any, Optional  # Using built-in types where ruff fixed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import IMAGE_DIR, ensure_dir

logger = logging.getLogger(__name__)


def get_regression_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> dict[str, float]:  # UP006
    """
    Calculates common regression metrics.
    """
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        logger.warning(
            "Invalid input for get_regression_metrics. Returning empty dict."
        )
        return {}

    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
    }
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"R2 Score: {metrics['r2_score']:.4f}")
    return metrics


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    dataset_name: str,
    model_name: str,
    config_identifier: str,  # For unique filenames
    save_dir_suffix: str = "",
    figsize: tuple[int, int] = (8, 8),
) -> Optional[str]:
    """
    Generates and saves a scatter plot of actual vs. predicted values.
    Returns the path to the saved plot or None.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Cannot plot actual vs predicted with empty data.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", s=50)

    # Line for perfect predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    ax.set_xlabel("Actual Values (Time to Next Stage)")
    ax.set_ylabel("Predicted Values (Time to Next Stage)")
    ax.set_title(
        f"Actual vs. Predicted - {model_name} on {dataset_name}\nConfig: {config_identifier}"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plot_subdir = os.path.join(
        IMAGE_DIR, "regression", dataset_name, model_name, save_dir_suffix
    )
    ensure_dir(plot_subdir)
    filename = f"actual_vs_pred_{config_identifier}.png"
    full_save_path = os.path.join(plot_subdir, filename)

    try:
        fig.savefig(full_save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Actual vs. Predicted plot saved to {full_save_path}")
        plt.close(fig)
        return full_save_path
    except Exception as e:
        logger.error(f"Failed to save Actual vs. Predicted plot: {e}")
        plt.close(fig)
        return None


def plot_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    dataset_name: str,
    model_name: str,
    config_identifier: str,
    save_dir_suffix: str = "",
    figsize: tuple[int, int] = (10, 6),
) -> Optional[str]:
    """
    Generates and saves a residuals plot (residuals vs. predicted values).
    Returns the path to the saved plot or None.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Cannot plot residuals with empty data.")
        return None

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", s=50)
    ax.axhline(0, color="r", linestyle="--", lw=2, label="Zero Residual")
    ax.set_xlabel("Predicted Values (Time to Next Stage)")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title(
        f"Residuals Plot - {model_name} on {dataset_name}\nConfig: {config_identifier}"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plot_subdir = os.path.join(
        IMAGE_DIR, "regression", dataset_name, model_name, save_dir_suffix
    )
    ensure_dir(plot_subdir)
    filename = f"residuals_{config_identifier}.png"
    full_save_path = os.path.join(plot_subdir, filename)

    try:
        fig.savefig(full_save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Residuals plot saved to {full_save_path}")
        plt.close(fig)
        return full_save_path
    except Exception as e:
        logger.error(f"Failed to save residuals plot: {e}")
        plt.close(fig)
        return None


# Feature importance for regression models (e.g., from Random Forest or Linear models)
def get_regression_feature_importance(
    model: Any,
    feature_names: list[str],  # UP006
) -> Optional[pd.DataFrame]:
    """
    Extracts feature importance from a trained regression model.
    Handles .feature_importances_ (tree-based) and .coef_ (linear).
    """
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For linear models, coef_ might be 1D or 2D (multi-output regression, less common here)
        if model.coef_.ndim == 1:
            importances = np.abs(model.coef_)  # Absolute importance for linear models
        elif (
            model.coef_.ndim > 1 and model.coef_.shape[0] == 1
        ):  # Common case for single target regression
            importances = np.abs(model.coef_[0])
        else:  # More complex multi-output, take mean over outputs
            importances = np.abs(model.coef_).mean(axis=0)
    else:
        logger.warning(
            f"Model {model.__class__.__name__} does not have standard feature_importances_ or coef_ attribute for regression."
        )
        return None

    if importances is not None:
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        return importance_df.sort_values(by="importance", ascending=False)
    return None


def plot_regression_feature_importance(
    importance_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    config_identifier: str,
    top_n: int = 20,
    save_dir_suffix: str = "",
    figsize: tuple[int, int] = (10, 8),
) -> Optional[str]:
    """
    Plots and saves feature importances for regression.
    """
    if importance_df is None or importance_df.empty:
        logger.info(
            f"No feature importance data to plot for regression model {model_name} on {dataset_name}."
        )
        return None

    top_features = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        x="importance", y="feature", data=top_features, palette="viridis", ax=ax
    )
    ax.set_title(
        f"Top {top_n} Feature Importances (Regression) - {model_name} on {dataset_name}\nConfig: {config_identifier}"
    )

    plot_subdir = os.path.join(
        IMAGE_DIR, "regression", dataset_name, model_name, save_dir_suffix
    )
    ensure_dir(plot_subdir)
    filename = f"feature_importance_regr_{config_identifier}.png"
    full_save_path = os.path.join(plot_subdir, filename)

    try:
        fig.savefig(full_save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Regression feature importance plot saved to {full_save_path}")
        plt.close(fig)
        return full_save_path
    except Exception as e:
        logger.error(f"Failed to save regression feature importance plot: {e}")
        plt.close(fig)
        return None
