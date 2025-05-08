import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Conditional imports for XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None  # type: ignore

logger = logging.getLogger(__name__)


def get_classifier(  # noqa: C901
    model_name: str,
    model_params: Optional[dict[str, Any]] = None,
    random_state: int = 42,
) -> ClassifierMixin:
    """
    Returns a classifier instance based on the model name and parameters.
    """
    params = model_params if model_params else {}

    # Ensure random_state is passed if the model supports it
    common_params: dict[str, Any] = {}
    if any(
        p in ["random_state", "seed", "random_seed"] for p in params
    ):  # If user provided it
        pass
    else:  # Add default if not provided and model supports it
        # This logic can be refined per model
        if model_name not in [
            "GaussianNB",
            "KNeighborsClassifier",
        ]:  # These don't always take random_state directly at init
            common_params["random_state"] = random_state

    if model_name == "LogisticRegression":
        # solver 'liblinear' is good for small datasets, 'saga' for larger, supports L1/L2.
        # 'lbfgs' is default but might warn for L1.
        merged_params = {
            "solver": "liblinear",
            "max_iter": 1000,
            **common_params,
            **params,
        }
        return LogisticRegression(**merged_params)
    elif model_name == "RandomForestClassifier":
        merged_params = {**common_params, **params}
        return RandomForestClassifier(**merged_params)
    elif model_name == "SVC":
        # SVC can be slow; consider LinearSVC for faster linear SVM.
        # probability=True is needed for predict_proba but makes it slower.
        merged_params = {"probability": True, **common_params, **params}
        return SVC(**merged_params)
    elif model_name == "KNeighborsClassifier":
        return KNeighborsClassifier(**params)  # n_neighbors is a key param
    elif model_name == "GaussianNB":
        return GaussianNB(**params)
    elif model_name == "MLPClassifier":
        merged_params = {
            "max_iter": 500,
            "early_stopping": True,
            **common_params,
            **params,
        }
        return MLPClassifier(**merged_params)
    elif model_name == "XGBClassifier" and XGBClassifier:
        merged_params = {
            **common_params,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            **params,
        }
        # 'use_label_encoder': False to suppress warning with newer XGBoost versions
        return XGBClassifier(**merged_params)
    elif model_name == "LGBMClassifier" and LGBMClassifier:
        merged_params = {**common_params, **params}
        return LGBMClassifier(**merged_params)
    else:
        logger.error(f"Model {model_name} not supported or library not installed.")
        raise ValueError(f"Model {model_name} not supported or library not installed.")


def train_model(
    model: ClassifierMixin,
    x_train_df: pd.DataFrame,  # N803: X_train -> x_train_df
    y_train_series: pd.Series,  # N803: y_train -> y_train_series
    x_val_df: Optional[pd.DataFrame] = None,  # N803: X_val -> x_val_df
    y_val_series: Optional[pd.Series] = None,  # N803: y_val -> y_val_series
    use_smote: bool = False,
) -> ClassifierMixin:
    """
    Trains the given model. Handles SMOTE if specified.
    x_val_df, y_val_series are for models like XGBoost/LightGBM that can use early stopping.
    """
    x_train_resampled_df: pd.DataFrame
    y_train_resampled_series: pd.Series

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            logger.info("Applying SMOTE to training data.")
            # Ensure y_train_series is 1D for SMOTE
            y_train_squeezed = (
                y_train_series.squeeze()
                if isinstance(y_train_series, pd.DataFrame)
                else y_train_series
            )

            # Check class distribution before SMOTE
            unique_classes, counts = np.unique(y_train_squeezed, return_counts=True)
            min_samples_for_smote = 6  # k_neighbors default is 5, so need at least k+1 samples for minority class

            smote_compatible = True
            for cls, count in zip(unique_classes, counts):
                if count < min_samples_for_smote:
                    logger.warning(
                        f"Class {cls} has {count} samples, less than required {min_samples_for_smote} for SMOTE. Skipping SMOTE."
                    )
                    smote_compatible = False
                    break

            if smote_compatible:
                smote = SMOTE(random_state=42)  # k_neighbors default is 5
                x_train_resampled_df, y_train_resampled_series = smote.fit_resample(
                    x_train_df, y_train_squeezed
                )  # N806
                logger.info(
                    f"Training data shape after SMOTE: {x_train_resampled_df.shape}"
                )
            else:  # SMOTE cannot be applied
                x_train_resampled_df, y_train_resampled_series = (
                    x_train_df,
                    y_train_squeezed,
                )  # N806

        except ImportError:
            logger.warning(
                "imblearn is not installed. Cannot use SMOTE. Training without it."
            )
            x_train_resampled_df, y_train_resampled_series = (
                x_train_df,
                y_train_series.squeeze(),
            )  # N806
        except (
            ValueError
        ) as e:  # Catches SMOTE errors like not enough samples in a class
            logger.warning(f"SMOTE failed: {e}. Training without it.")
            x_train_resampled_df, y_train_resampled_series = (
                x_train_df,
                y_train_series.squeeze(),
            )  # N806
    else:
        x_train_resampled_df, y_train_resampled_series = (
            x_train_df,
            y_train_series.squeeze(),
        )  # N806

    logger.info(f"Training model: {model.__class__.__name__}")

    # Handle models that support early stopping
    fit_params: dict[str, Any] = {}
    model_name = model.__class__.__name__
    if (
        model_name in ["XGBClassifier", "LGBMClassifier"]
        and x_val_df is not None
        and y_val_series is not None
    ):
        fit_params["early_stopping_rounds"] = 10
        fit_params["eval_set"] = [(x_val_df, y_val_series.squeeze())]
        fit_params["verbose"] = (
            False  # Suppress verbose output during fitting for these
        )
        logger.info(f"Using early stopping for {model_name} with eval_set.")

    model.fit(x_train_resampled_df, y_train_resampled_series, **fit_params)
    logger.info("Model training complete.")
    return model
