import logging
from typing import Any, Union  # Mypy fix: Import Union for Python < 3.10 compatibility

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# from sklearn.metrics import make_scorer, f1_score # Not directly used for string 'f1_macro'

logger = logging.getLogger(__name__)

# Define hyperparameter grids for different models
# These are examples and should be expanded/tuned
DEFAULT_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],  # 'liblinear' supports l1 and l2
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced", "balanced_subsample"],
    },
    "SVC": {  # Can be very slow
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],  # 'poly' can also be slow
        "gamma": ["scale", "auto"],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean"],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(50,), (100,), (50, 25)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    },
    "XGBClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "LGBMClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [20, 31, 40],  # Default 31
        "max_depth": [-1, 5, 10],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
}


def perform_hyperparameter_tuning(
    model: BaseEstimator,  # Untrained model instance
    x_train_df: pd.DataFrame,  # N803: X_train -> x_train_df
    y_train_series: pd.Series,  # N803: y_train -> y_train_series
    param_grid: dict[str, Any],
    cv_splits: list[tuple[Any, Any]],  # Pre-computed GroupKFold splits
    scoring: str = "f1_macro",  # e.g., 'f1_macro', 'accuracy'
    search_method: str = "grid",  # 'grid' or 'random'
    n_iter_random: int = 10,  # For RandomizedSearchCV
) -> tuple[BaseEstimator, dict[str, Any]]:
    """
    Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Args:
        model: The classifier instance.
        x_train_df: Training features.
        y_train_series: Training target.
        param_grid: Dictionary of parameters to search.
        cv_splits: List of (train_idx, test_idx) for cross-validation (e.g., from GroupKFold).
        scoring: Scoring metric to optimize.
        search_method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
        n_iter_random: Number of parameter settings that are sampled for RandomizedSearchCV.

    Returns:
        A tuple (best_estimator, best_params).
    """
    logger.info(
        f"Starting hyperparameter tuning for {model.__class__.__name__} using {search_method.upper()}Search."
    )
    logger.info(f"Parameter grid: {param_grid}")

    # Scorer for multi-class F1 macro
    # Ensure labels are passed if your target has missing classes in some folds
    # For simplicity, assuming labels are consistent or sklearn handles it.
    # If issues arise, create scorer: make_scorer(f1_score, average='macro', labels=sorted(y_train_series.unique()))

    # Mypy fix: Use Union for Python < 3.10 compatibility
    search_cv: Union[GridSearchCV, RandomizedSearchCV]
    if search_method == "grid":
        search_cv = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_splits,  # type: ignore # cv_splits is a list of tuples, compatible with CV splitter
            verbose=1,
            n_jobs=-1,
        )
    elif search_method == "random":
        search_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter_random,
            scoring=scoring,
            cv=cv_splits,  # type: ignore
            verbose=1,
            n_jobs=-1,
            random_state=42,  # For reproducibility of random search
        )
    else:
        raise ValueError("search_method must be 'grid' or 'random'")

    try:
        search_cv.fit(x_train_df, y_train_series.squeeze())
        logger.info(f"Best parameters found: {search_cv.best_params_}")
        logger.info(f"Best {scoring} score: {search_cv.best_score_:.4f}")
        return search_cv.best_estimator_, search_cv.best_params_
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}", exc_info=True)
        # Fallback to default model if tuning fails catastrophically
        # This is a simple fallback; more robust error handling might be needed
        # For instance, some parameter combinations might be invalid.
        logger.warning("Falling back to default model parameters.")
        # Ensure model is a ClassifierMixin for fit method, though BaseEstimator is more general
        if hasattr(model, "fit"):
            model.fit(x_train_df, y_train_series.squeeze())  # type: ignore[union-attr]
        return model, model.get_params()
