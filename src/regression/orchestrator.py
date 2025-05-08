import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from src.regression.data_preparation import (
    prepare_regression_data,
    split_data_grouped_regression,
)
from src.regression.evaluation import (
    get_regression_feature_importance,
    get_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
)
from src.regression.models import get_regressor, train_regressor
from src.regression.tuning import (
    DEFAULT_REGRESSION_PARAM_GRIDS,
    perform_regression_hyperparameter_tuning,
)
from src.utils import (
    REPORTS_DIR,
    ensure_dir,
)

logger = logging.getLogger(__name__)


def generate_config_identifier(params: dict[str, Any], prefix: str = "") -> str:
    """Creates a short, unique identifier from parameters and a prefix."""
    param_string = json.dumps(params, sort_keys=True, default=str)
    return f"{prefix}_{hashlib.sha256(param_string.encode('utf-8')).hexdigest()[:8]}"


def run_regression_pipeline_for_fold(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    dataset_name: str,
    fold_num: int,
    model_config: dict[str, Any],
    config_name_for_run: str,
) -> dict[str, Any]:
    """
    Runs the regression pipeline for a single fold: trains, predicts, evaluates, and plots.
    """
    fold_results: dict[str, Any] = {
        "fold": fold_num,
        "model_name": model_name,
        "config_name": config_name_for_run,
    }

    current_model_params = model_config.get("params_to_use", {})
    fold_config_identifier = generate_config_identifier(
        current_model_params, prefix=f"f{fold_num}_{config_name_for_run}"
    )

    logger.info(
        f"[Fold {fold_num}] Training {model_name} (Config: {config_name_for_run}) with params: {current_model_params}"
    )

    try:
        regressor_instance = get_regressor(model_name, current_model_params)
        trained_regressor = train_regressor(
            regressor_instance, x_train, y_train, x_val, y_val
        )

        y_pred_val = trained_regressor.predict(x_val)
        y_pred_series = pd.Series(y_pred_val, index=y_val.index)

        val_metrics = get_regression_metrics(y_val, y_pred_series)
        fold_results["params_used"] = current_model_params
        fold_results["validation_metrics"] = val_metrics

        plot_save_dir_suffix = f"fold_validations/{config_name_for_run}"
        plot_actual_vs_predicted(
            y_val,
            y_pred_series,
            dataset_name,
            model_name,
            fold_config_identifier,
            save_dir_suffix=plot_save_dir_suffix,
        )
        plot_residuals(
            y_val,
            y_pred_series,
            dataset_name,
            model_name,
            fold_config_identifier,
            save_dir_suffix=plot_save_dir_suffix,
        )

        if hasattr(x_train, "columns") and x_train.columns.tolist():
            feat_importance_df = get_regression_feature_importance(
                trained_regressor, x_train.columns.tolist()
            )
            if feat_importance_df is not None:
                fold_results["feature_importances_sample_fold"] = (
                    feat_importance_df.head(10).to_dict(orient="records")
                )

    except Exception as e:
        logger.error(
            f"[Fold {fold_num}] Error during model training/evaluation for {model_name} ({config_name_for_run}): {e}",
            exc_info=True,
        )
        fold_results["error"] = str(e)
    return fold_results


def _setup_run_environment(
    dataset_name: str, target_stage_col: str
) -> tuple[str, str, str]:
    """Sets up timestamps and base directories for reports."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_detailed_report_dir = os.path.join(
        REPORTS_DIR, "regression", dataset_name, target_stage_col
    )
    ensure_dir(base_detailed_report_dir)

    dataset_run_summary_filename = (
        f"summary_regr_{dataset_name}_{target_stage_col}_{run_timestamp}.json"
    )
    dataset_run_summary_path = os.path.join(
        REPORTS_DIR, "regression", dataset_run_summary_filename
    )
    ensure_dir(os.path.dirname(dataset_run_summary_path))
    return run_timestamp, base_detailed_report_dir, dataset_run_summary_path


def _load_and_prepare_data_splits(
    dataset_name: str,
    target_stage_col: str,
    max_cluster_stage_value: int,
    n_cv_splits: int,
) -> Optional[tuple[pd.DataFrame, pd.Series, pd.Series, list[tuple[Any, Any]]]]:
    """Loads data, prepares it for regression, and creates CV splits."""
    x_all, y_all, engine_ids_all = prepare_regression_data(
        dataset_name, target_stage_col, max_cluster_stage_value
    )
    if x_all is None or y_all is None or engine_ids_all is None:
        logger.error(f"Failed to prepare regression data for {dataset_name}. Aborting.")
        return None

    cv_splits: list[tuple[Any, Any]] = split_data_grouped_regression(
        x_all, y_all, engine_ids_all, n_splits=n_cv_splits
    )
    if not cv_splits:
        logger.error(f"Failed to generate CV splits for {dataset_name}. Aborting.")
        return None
    return x_all, y_all, engine_ids_all, cv_splits


def _determine_model_parameters(
    model_name: str,
    config_run_details: dict[str, Any],
    x_all: pd.DataFrame,
    y_all: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    config_run_name: str,
) -> dict[str, Any]:
    """Determines model parameters, performing tuning if specified."""
    params_for_cv_folds: dict[str, Any] = config_run_details.get(
        "initial_params", {}
    ).copy()
    if config_run_details.get("perform_tuning", False):
        logger.info(
            f"Performing Hyperparameter Tuning for {model_name} (Config: {config_run_name})"
        )
        param_grid_tuning = config_run_details.get(
            "param_grid", DEFAULT_REGRESSION_PARAM_GRIDS.get(model_name)
        )
        if not param_grid_tuning:
            logger.warning(
                f"No param_grid for {model_name} tuning. Using initial_params for {config_run_name}."
            )
        else:
            base_regressor_for_tuning = get_regressor(model_name, random_state=42)
            typed_cv_splits: list[tuple[Any, Any]] = cv_splits
            _best_estimator, best_params_from_tuning = (
                perform_regression_hyperparameter_tuning(
                    base_regressor_for_tuning,
                    x_all,
                    y_all,
                    param_grid_tuning,
                    cv_splits=typed_cv_splits,
                    scoring=config_run_details.get(
                        "tuning_scoring", "neg_root_mean_squared_error"
                    ),
                    search_method=config_run_details.get(
                        "tuning_search_method", "grid"
                    ),
                    n_iter_random=config_run_details.get("tuning_n_iter", 10),
                )
            )
            # best_params_from_tuning is dict[str, Any] as per perform_regression_hyperparameter_tuning
            params_for_cv_folds = best_params_from_tuning
            logger.info(
                f"Tuned best params for {model_name} (Config: {config_run_name}): {params_for_cv_folds}"
            )
    return params_for_cv_folds  # This should be dict[str, Any]


def _run_cv_evaluation_for_config(
    model_name: str,
    params_to_use: dict[str, Any],
    x_all: pd.DataFrame,
    y_all: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    dataset_name: str,
    config_run_name: str,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Runs cross-validation for a given model configuration and collects results."""
    fold_results_list: list[dict[str, Any]] = []
    fold_metrics_accumulator: dict[str, list[float]] = {}

    for i, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(
            f"--- Fold {i + 1}/{len(cv_splits)} for {model_name}, Config: {config_run_name} ---"
        )
        x_train_fold, x_val_fold = x_all.iloc[train_idx], x_all.iloc[val_idx]
        y_train_fold, y_val_fold = y_all.iloc[train_idx], y_all.iloc[val_idx]

        fold_run_result = run_regression_pipeline_for_fold(
            model_name=model_name,
            x_train=x_train_fold,
            y_train=y_train_fold,
            x_val=x_val_fold,
            y_val=y_val_fold,
            dataset_name=dataset_name,
            fold_num=i + 1,
            model_config={"params_to_use": params_to_use},
            config_name_for_run=config_run_name,
        )
        fold_results_list.append(fold_run_result)
        if (
            "validation_metrics" in fold_run_result
            and fold_run_result["validation_metrics"]
        ):
            for metric_key, metric_val in fold_run_result["validation_metrics"].items():
                fold_metrics_accumulator.setdefault(metric_key, []).append(
                    float(metric_val)
                )
    return fold_results_list, fold_metrics_accumulator


def _save_config_run_report(
    summary_data: dict[str, Any],
    model_base_report_dir: str,
    config_run_name: str,
    params_used: dict[str, Any],
) -> None:
    """Saves the detailed report for a specific model configuration run."""
    config_run_report_ident = generate_config_identifier(
        params_used, prefix=config_run_name
    )
    config_run_report_filename = f"report_{config_run_report_ident}.json"
    config_run_report_path = os.path.join(
        model_base_report_dir, config_run_report_filename
    )
    try:
        with open(config_run_report_path, "w") as f:
            json.dump(summary_data, f, indent=4, default=lambda o: "<not serializable>")
        logger.info(
            f"Saved detailed report for {summary_data.get('model_name')} ({config_run_name}) to {config_run_report_path}"
        )
    except TypeError as e:
        logger.error(f"JSON serialization error for {config_run_report_path}: {e}")


# This is the function that had the error at line 216 (return statement)
def _process_single_model_config(
    model_name: str,
    config_run_details: dict[str, Any],
    x_all: pd.DataFrame,
    y_all: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    dataset_name: str,
    target_stage_col: str,
    model_base_report_dir: str,
) -> dict[str, Any]:  # The function is declared to return dict[str, Any]
    """Processes a single model configuration: tuning, CV, reporting."""
    config_run_name = config_run_details.get(
        "name", f"unnamed_config_{generate_config_identifier(config_run_details)}"
    )
    logger.info(
        f"--- Processing Model: {model_name}, Config Run: {config_run_name} ---"
    )

    # Explicitly annotate the type of the variable receiving the result
    # from _determine_model_parameters.
    determined_params: dict[str, Any] = _determine_model_parameters(
        model_name, config_run_details, x_all, y_all, cv_splits, config_run_name
    )

    current_config_run_summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "target_stage_col_source": target_stage_col,
        "model_name": model_name,
        "config_run_name": config_run_name,
        "params_determined_for_folds": determined_params,  # Use the explicitly typed variable
        "original_config_settings": config_run_details,
        "fold_results": [],
        "average_validation_metrics": {},
    }

    fold_results_list, fold_metrics_accumulator = _run_cv_evaluation_for_config(
        model_name,
        determined_params,
        x_all,
        y_all,
        cv_splits,
        dataset_name,
        config_run_name,
    )
    current_config_run_summary["fold_results"] = fold_results_list

    avg_val_metrics: dict[str, float] = {
        f"avg_{key}": float(np.mean(values))
        for key, values in fold_metrics_accumulator.items()
        if values
    }
    current_config_run_summary["average_validation_metrics"] = avg_val_metrics

    _save_config_run_report(
        current_config_run_summary,
        model_base_report_dir,
        config_run_name,
        determined_params,
    )
    # The variable current_config_run_summary is annotated as dict[str, Any].
    # The cast here is to reinforce this to mypy if it's losing track.
    return cast(dict[str, Any], current_config_run_summary)


def run_regression_for_dataset(
    dataset_name: str,
    target_stage_col: str,
    max_cluster_stage_value: int,
    model_configurations: dict[str, list[dict[str, Any]]],
    n_cv_splits: int = 5,
) -> list[dict[str, Any]]:
    """
    Main orchestrator for regression on a single dataset.
    """
    _run_timestamp, base_detailed_report_dir, dataset_run_summary_path = (
        _setup_run_environment(dataset_name, target_stage_col)
    )
    logger.info(
        f"--- Starting Regression Pipeline for Dataset: {dataset_name}, Target Stage Col: {target_stage_col} ---"
    )

    data_prep_result = _load_and_prepare_data_splits(
        dataset_name, target_stage_col, max_cluster_stage_value, n_cv_splits
    )
    if data_prep_result is None:
        return [
            {"dataset": dataset_name, "error": "Data preparation or splitting failed"}
        ]

    x_all: pd.DataFrame = data_prep_result[0]
    y_all: pd.Series = data_prep_result[1]
    # _engine_ids_all: pd.Series = data_prep_result[2] # Not used here
    cv_splits_list: list[tuple[Any, Any]] = data_prep_result[3]

    overall_results_list: list[dict[str, Any]] = []
    for model_name, configs_for_model in model_configurations.items():
        model_base_report_dir = os.path.join(base_detailed_report_dir, model_name)
        ensure_dir(model_base_report_dir)

        for config_run_details in configs_for_model:
            config_summary = _process_single_model_config(
                model_name,
                config_run_details,
                x_all,
                y_all,
                cv_splits_list,
                dataset_name,
                target_stage_col,
                model_base_report_dir,
            )
            overall_results_list.append(config_summary)

    logger.info(f"--- Regression Pipeline for Dataset: {dataset_name} Completed ---")
    try:
        with open(dataset_run_summary_path, "w") as f:
            json.dump(
                overall_results_list,
                f,
                indent=4,
                default=lambda o: "<not serializable>",
            )
        logger.info(
            f"Full regression run summary for {dataset_name} saved to {dataset_run_summary_path}"
        )
    except TypeError as e:
        logger.error(
            f"JSON serialization error for overall summary {dataset_run_summary_path}: {e}"
        )
    return overall_results_list
