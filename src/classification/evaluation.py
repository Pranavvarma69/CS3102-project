import logging
import os
from typing import (
    Any,
    Optional,
)

# Removed 'Union' if it was added before and not needed here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import IMAGE_DIR, ensure_dir

logger = logging.getLogger(__name__)


def get_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_pred_proba: Optional[np.ndarray] = None,
    labels: Optional[list[int]] = None,
) -> dict[str, Any]:
    """
    Calculates various classification metrics.
    Args:
        labels: Optional list of unique sorted labels for classification report.
    """
    if labels is None:
        # Ruff C414 fix: Remove unnecessary list() call
        unique_elements = y_true.unique()
        labels = sorted(unique_elements)  # type: ignore # sorted can handle numpy arrays / pandas unique outputs
        # Mypy might complain depending on precise type of unique_elements
        # If y_true.unique() is guaranteed to be list-like of ints, this is fine.
        # Or cast explicitly: labels = sorted([int(x) for x in unique_elements])
        # For now, assuming sorted() handles it, can refine if mypy still flags unique_elements type.
        # A safer approach for mypy would be:
        # labels = sorted(list(map(int, y_true.unique())))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score_macro": f1_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "f1_score_weighted": f1_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, labels=labels, zero_division=0
        ),
    }
    # Per-class metrics (can be extracted from classification_report dict as well)
    # f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    # precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    # recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    # for i, label in enumerate(labels):
    #     metrics[f"f1_class_{label}"] = f1_per_class[i]
    #     metrics[f"precision_class_{label}"] = precision_per_class[i]
    #     metrics[f"recall_class_{label}"] = recall_per_class[i]

    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (Macro): {metrics['f1_score_macro']:.4f}")
    # logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, labels=labels, zero_division=0)}") # Log full report
    return metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[int],  # Unique sorted labels
    dataset_name: str,
    model_name: str,
    file_identifier: str,  # To make filename unique (e.g., params_hash or timestamp)
    save_dir_suffix: str = "",  # e.g. "baseline" or "tuned"
    figsize: tuple[int, int] = (8, 6),
) -> str:
    """
    Generates and saves a confusion matrix plot.
    Returns the path to the saved plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(
        f"Confusion Matrix - {model_name} on {dataset_name}\n({file_identifier})"
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Define path structure: images/classification/DATASET_NAME/MODEL_NAME/save_dir_suffix/filename.png
    plot_subdir = os.path.join(
        IMAGE_DIR, "classification", dataset_name, model_name, save_dir_suffix
    )
    ensure_dir(plot_subdir)  # ensure_dir from src.utils

    filename = f"cm_{file_identifier}.png"
    full_save_path = os.path.join(plot_subdir, filename)

    try:
        fig.savefig(full_save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Confusion matrix saved to {full_save_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix to {full_save_path}: {e}")
        plt.close(fig)  # Close figure even if save fails
        return ""  # Return empty string on failure
    plt.close(fig)
    return full_save_path


def get_feature_importance(
    model: BaseEstimator, feature_names: list[str]
) -> Optional[pd.DataFrame]:
    """
    Extracts feature importance from a trained model.
    """
    importances: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For linear models, coef_ can be multi-dimensional for multi-class
        # Taking absolute mean across classes for simplicity, or sum if preferred
        if model.coef_.ndim > 1:
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            # For binary or single output tasks, coef_ might be (1, n_features) or (n_features,)
            coef_array = (
                model.coef_[0]
                if model.coef_.ndim == 2 and model.coef_.shape[0] == 1
                else model.coef_
            )
            importances = np.abs(coef_array)

    else:
        logger.warning(
            f"Model {model.__class__.__name__} does not have standard feature_importances_ or coef_ attribute."
        )
        return None

    if (
        importances is None
    ):  # Should not happen if logic above is correct, but as a safeguard
        logger.warning("Could not extract importances.")
        return None

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    file_identifier: str,
    top_n: int = 20,
    save_dir_suffix: str = "",
    figsize: tuple[int, int] = (10, 8),
) -> Optional[str]:
    """
    Plots and saves feature importances.
    Returns path to saved plot or None.
    """
    if importance_df is None or importance_df.empty:
        logger.info(
            f"No feature importance data to plot for {model_name} on {dataset_name}."
        )
        return None

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        x="importance", y="feature", data=top_features, palette="viridis", ax=ax
    )
    ax.set_title(
        f"Top {top_n} Feature Importances - {model_name} on {dataset_name}\n({file_identifier})"
    )

    plot_subdir = os.path.join(
        IMAGE_DIR, "classification", dataset_name, model_name, save_dir_suffix
    )
    ensure_dir(plot_subdir)
    filename = f"feature_importance_{file_identifier}.png"
    full_save_path = os.path.join(plot_subdir, filename)

    try:
        fig.savefig(full_save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Feature importance plot saved to {full_save_path}")
    except Exception as e:
        logger.error(f"Failed to save feature importance plot: {e}")
        plt.close(fig)
        return None
    plt.close(fig)
    return full_save_path
