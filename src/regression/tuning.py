import logging
from typing import Any  # Using built-in types where ruff fixed

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__name__)


# Define RMSE scorer for hyperparameter tuning
def rmse_scorer(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates Root Mean Squared Error."""
    # Ensure the return type is explicitly float for mypy
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Use greater_is_better=False for RMSE since lower is better
SCORER_RMSE = make_scorer(rmse_scorer, greater_is_better=False)


DEFAULT_REGRESSION_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {  # UP006
    "LinearRegression": {
        "fit_intercept": [True, False],
        # 'normalize' is deprecated in newer sklearn, use StandardScaler beforehand
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 150],  # Reduced for speed
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
    },
    "SVR": {  # Can be very slow
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],  # poly can be very slow
        "gamma": ["scale", "auto"],
        "epsilon": [0.1, 0.2],
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 150],  # Reduced for speed
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
}


def perform_regression_hyperparameter_tuning(
    model: Any,  # Regressor instance
    x_train: pd.DataFrame,  # N803
    y_train: pd.Series,
    param_grid: dict[str, Any],  # UP006
    cv_splits: list[tuple[Any, Any]],  # UP006: Pre-computed GroupKFold splits
    scoring: Any = SCORER_RMSE,  # Default to RMSE, lower is better
    search_method: str = "grid",
    n_iter_random: int = 10,
) -> tuple[Any, dict[str, Any]]:  # UP006
    """
    Performs hyperparameter tuning for regression models.
    """
    logger.info(
        f"Starting regression hyperparameter tuning for {model.__class__.__name__} using {search_method.upper()}Search."
    )
    logger.info(f"Parameter grid: {param_grid}")

    y_train_squeezed = (
        y_train.squeeze() if isinstance(y_train, pd.DataFrame) else y_train
    )

    if search_method == "grid":
        search_cv = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_splits,  # Pass pre-computed GroupKFold splits
            verbose=1,
            n_jobs=-1,
        )
    elif search_method == "random":
        search_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter_random,
            scoring=scoring,
            cv=cv_splits,
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError("search_method must be 'grid' or 'random'")

    try:
        search_cv.fit(x_train, y_train_squeezed)
        logger.info(f"Best parameters found: {search_cv.best_params_}")
        # Ensure score is treated as float if it might be Any type from search_cv.best_score_
        best_score_val: float = search_cv.best_score_
        logger.info(
            f"Best score ({scoring}): {best_score_val:.4f}"
        )  # For RMSE, higher score is worse if greater_is_better=False
        return search_cv.best_estimator_, search_cv.best_params_
    except Exception as e:
        logger.error(f"Regression hyperparameter tuning failed: {e}", exc_info=True)
        logger.warning("Falling back to default model parameters for regression.")
        # Fit with default params as fallback
        model.fit(x_train, y_train_squeezed)
        return model, model.get_params()
