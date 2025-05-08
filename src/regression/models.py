import logging
from typing import Any, Optional  # Using built-in types where ruff fixed

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore

logger = logging.getLogger(__name__)


def get_regressor(
    model_name: str,
    model_params: Optional[dict[str, Any]] = None,
    random_state: int = 42,  # mypy, UP006
) -> Any:  # mypy
    """
    Returns a regressor instance based on the model name and parameters.
    """
    params = model_params.copy() if model_params else {}

    # Common params application (some models might not use random_state directly at init)
    if "random_state" not in params and model_name not in [
        "SVR",
        "Ridge",
        "LinearRegression",
    ]:  # SVR uses it internally but not at init for main algorithm always
        params["random_state"] = random_state

    if model_name == "LinearRegression":
        return LinearRegression(**params)
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    elif model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "SVR":
        # SVR can be slow, especially with large C or certain kernels
        # Default kernel='rbf'
        return SVR(**params)
    elif model_name == "XGBRegressor" and XGBRegressor:
        # Ensure 'objective' is suitable for regression, e.g., 'reg:squarederror'
        if "objective" not in params:
            params["objective"] = "reg:squarederror"
        if "eval_metric" not in params:  # Good for early stopping if used
            params["eval_metric"] = "rmse"
        # 'use_label_encoder' is for classifiers, not relevant here
        return XGBRegressor(**params)
    else:
        logger.error(f"Regressor {model_name} not supported or library not installed.")
        raise ValueError(
            f"Regressor {model_name} not supported or library not installed."
        )


def train_regressor(
    model: Any,  # Regressor instance
    x_train: pd.DataFrame,  # N803
    y_train: pd.Series,
    x_val: Optional[pd.DataFrame] = None,  # N803
    y_val: Optional[pd.Series] = None,
) -> Any:  # mypy
    """
    Trains the given regressor.
    x_val, y_val are for models like XGBoost that can use early stopping.
    """
    logger.info(f"Training regressor: {model.__class__.__name__}")

    # Ensure y_train is 1D
    y_train_squeezed = (
        y_train.squeeze() if isinstance(y_train, pd.DataFrame) else y_train
    )

    fit_params: dict[str, Any] = {}  # UP006, mypy
    model_name = model.__class__.__name__

    if model_name == "XGBRegressor" and x_val is not None and y_val is not None:
        fit_params["early_stopping_rounds"] = 10  # Example value
        fit_params["eval_set"] = [(x_val, y_val.squeeze())]
        fit_params["verbose"] = False  # Suppress verbose output during fitting
        logger.info(f"Using early stopping for {model_name} with eval_set.")

    try:
        model.fit(x_train, y_train_squeezed, **fit_params)
        logger.info("Regressor training complete.")
    except Exception as e:
        logger.error(f"Error during training {model_name}: {e}", exc_info=True)
        raise
    return model
