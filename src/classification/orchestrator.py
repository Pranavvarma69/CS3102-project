import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Optional, cast  # Ensure cast is imported

import joblib  # <--- Add this import
import numpy as np
import pandas as pd

from src.classification.data_preparation import (
    load_and_merge_data,
    prepare_features_target,
    split_data_grouped,
)
from src.classification.evaluation import (
    get_classification_metrics,
    get_feature_importance,
    plot_confusion_matrix,
)

# plot_feature_importance # If you want to plot for the final model
from src.classification.models import get_classifier, train_model
from src.classification.tuning import DEFAULT_PARAM_GRIDS, perform_hyperparameter_tuning
from src.utils import (
    REPORTS_DIR,
    ensure_dir,
)

logger = logging.getLogger(__name__)


def generate_params_identifier(params: dict[str, Any]) -> str:
    """Creates a short, unique identifier from a dictionary of parameters."""
    param_string = json.dumps(params, sort_keys=True, default=str)
    # Changed to sha256 for better security, kept [:8] for brevity of identifier
    return hashlib.sha256(param_string.encode("utf-8")).hexdigest()[:8]


# run_classification_pipeline_for_fold remains largely the same.
# Ensure its base_report_dir argument points to the model_specific_report_dir
# and its confusion matrix plotting uses a clear path structure.
# For example, plot_confusion_matrix could take model_specific_image_dir.


def run_classification_pipeline_for_fold(
    model_name: str,
    x_train_df: pd.DataFrame,
    y_train_series: pd.Series,
    x_val_df: pd.DataFrame,
    y_val_series: pd.Series,
    dataset_name: str,
    target_stage_col: str,
    fold_num: int,
    model_config: dict[str, Any],
    unique_labels: list[int],
) -> dict[str, Any]:
    fold_results: dict[str, Any] = {"fold": fold_num, "model_name": model_name}

    initial_params = model_config.get("initial_params", {})
    use_smote_fold = model_config.get("use_smote", False)
    class_weight_fold = model_config.get("class_weight", None)

    if class_weight_fold and "class_weight" not in initial_params:
        initial_params["class_weight"] = class_weight_fold

    current_model_params = initial_params.copy()
    config_run_name_for_id = model_config.get("name", "unnamed_config")
    model_identifier_str = (
        f"fold{fold_num}_{model_name}_{config_run_name_for_id}_smote{use_smote_fold}"
    )
    if current_model_params:
        model_identifier_str += f"_{generate_params_identifier(current_model_params)}"

    try:
        model_instance = get_classifier(model_name, current_model_params)
        trained_model = train_model(
            model_instance,
            x_train_df,
            y_train_series,
            x_val_df,
            y_val_series,
            use_smote=use_smote_fold,
        )

        y_pred_val = trained_model.predict(x_val_df)
        y_pred_proba_val = None
        if hasattr(trained_model, "predict_proba"):
            y_pred_proba_val = trained_model.predict_proba(x_val_df)

        val_metrics = get_classification_metrics(
            y_val_series, y_pred_val, y_pred_proba_val, labels=unique_labels
        )
        fold_results["initial_params_used_for_fold"] = current_model_params
        fold_results["smote_applied_in_fold"] = use_smote_fold
        fold_results["validation_metrics"] = val_metrics

        cm_path = plot_confusion_matrix(
            y_true=y_val_series,
            y_pred=y_pred_val,
            labels=unique_labels,
            dataset_name=dataset_name,
            model_name=model_name,
            file_identifier=model_identifier_str,
            save_dir_suffix=os.path.join(
                target_stage_col,
                model_config.get("name", "default_config"),
                "fold_validations",
            ),
        )
        fold_results["confusion_matrix_val_path"] = cm_path

        if x_train_df.columns.tolist():
            feat_importance_df = get_feature_importance(
                trained_model,
                x_train_df.columns.tolist(),
            )
            if feat_importance_df is not None:
                fold_results["feature_importances_sample"] = feat_importance_df.head(
                    10
                ).to_dict(orient="records")
    except Exception as e:
        logger.error(
            f"[Fold {fold_num}] Error during model training/evaluation for {model_name} ({config_run_name_for_id}): {e}",
            exc_info=True,
        )
        fold_results["error"] = str(e)

    return fold_results


def _prepare_initial_data_and_splits(
    dataset_name: str, target_stage_col: str, n_cv_splits: int
) -> tuple[
    Optional[pd.DataFrame], Optional[pd.Series], Optional[list], Optional[list[int]]
]:
    """Loads data, prepares features/target, and generates CV splits."""
    df_merged = load_and_merge_data(dataset_name)
    if df_merged is None:
        logger.error(f"Failed to load or merge data for {dataset_name}. Aborting.")
        return None, None, None, None

    x_all_df, y_all_series = prepare_features_target(df_merged, target_stage_col)
    if x_all_df is None or y_all_series is None:
        logger.error(f"Failed to prepare features/target for {dataset_name}. Aborting.")
        return None, None, None, None

    if "engine_id" not in df_merged.columns:
        logger.error(
            "'engine_id' column not found in df_merged. Required for GroupKFold splitting. Aborting."
        )
        return None, None, None, None

    engine_ids_all = df_merged.loc[x_all_df.index, "engine_id"]
    unique_labels = sorted(cast(list, y_all_series.unique().tolist()))
    logger.info(f"Unique target labels found: {unique_labels}")

    cv_splits_list = split_data_grouped(
        x_all_df,
        y_all_series,
        engine_ids_all,
        n_splits=n_cv_splits,
        use_group_kfold=True,
    )
    if not cv_splits_list:
        logger.error("Failed to generate CV splits. Aborting.")
        return (
            x_all_df,
            y_all_series,
            None,
            unique_labels,
        )  # Return data even if splits fail for some reason

    return x_all_df, y_all_series, cv_splits_list, unique_labels


def _process_model_configuration(  # noqa: C901
    model_name: str,
    model_config_original: dict[str, Any],
    config_idx: int,
    x_all_df: pd.DataFrame,
    y_all_series: pd.Series,
    cv_splits_list: list,  # Expect list of tuples
    n_cv_splits: int,
    dataset_name: str,
    target_stage_col: str,
    unique_labels: list[int],
    model_specific_artifacts_dir: str,
    perform_final_fit_on_full_train: bool,
) -> dict[str, Any]:
    """Processes a single model configuration: tuning, CV, final fit, and reporting."""
    model_config = model_config_original.copy()
    config_name = model_config.get("name", f"config_{config_idx + 1}")
    logger.info(f"--- Processing Model: {model_name}, Config: {config_name} ---")

    all_fold_metrics_for_this_config = []
    params_to_use_for_folds_and_final_model = model_config.get(
        "initial_params", {}
    ).copy()
    tuned_model_candidate: Optional[Any] = None

    if model_config.get("perform_tuning", False):
        logger.info(
            f"Performing Hyperparameter Tuning for {model_name} (Config: {config_name})"
        )
        param_grid = model_config.get("param_grid", DEFAULT_PARAM_GRIDS.get(model_name))
        if not param_grid:
            logger.warning(
                f"No param_grid for {model_name} tuning. Using initial_params for CV."
            )
        else:
            base_model_for_tuning = get_classifier(model_name, random_state=42)
            try:
                tuned_model_candidate, best_params_from_tuning = (
                    perform_hyperparameter_tuning(
                        base_model_for_tuning,
                        x_all_df,
                        y_all_series,
                        param_grid,
                        cv_splits=cv_splits_list,
                        scoring=model_config.get("tuning_scoring", "f1_macro"),
                        search_method=model_config.get("tuning_search_method", "grid"),
                        n_iter_random=model_config.get("tuning_n_iter", 10),
                    )
                )
                params_to_use_for_folds_and_final_model = best_params_from_tuning
                logger.info(
                    f"Tuned best params for {model_name} (Config: {config_name}): {params_to_use_for_folds_and_final_model}"
                )
            except Exception as e:
                logger.error(
                    f"Hyperparameter tuning failed for {model_name} (Config: {config_name}): {e}. Using initial_params.",
                    exc_info=True,
                )

    fold_runner_model_config = model_config.copy()
    fold_runner_model_config["initial_params"] = params_to_use_for_folds_and_final_model
    fold_runner_model_config["name"] = config_name

    for i, (train_idx, val_idx) in enumerate(cv_splits_list):
        logger.info(
            f"--- Fold {i + 1}/{n_cv_splits} for {model_name}, Config: {config_name} using params: {params_to_use_for_folds_and_final_model} ---"
        )
        x_train_fold, x_val_fold = (
            x_all_df.iloc[train_idx],
            x_all_df.iloc[val_idx],
        )
        y_train_fold, y_val_fold = (
            y_all_series.iloc[train_idx],
            y_all_series.iloc[val_idx],
        )
        fold_run_summary = run_classification_pipeline_for_fold(
            model_name=model_name,
            x_train_df=x_train_fold,
            y_train_series=y_train_fold,
            x_val_df=x_val_fold,
            y_val_series=y_val_fold,
            dataset_name=dataset_name,
            target_stage_col=target_stage_col,
            fold_num=i + 1,
            model_config=fold_runner_model_config,
            unique_labels=unique_labels,
        )
        all_fold_metrics_for_this_config.append(fold_run_summary)

    avg_val_metrics: dict[str, Any] = {}
    valid_fold_results = [
        f_res
        for f_res in all_fold_metrics_for_this_config
        if "validation_metrics" in f_res and f_res["validation_metrics"]
    ]
    if valid_fold_results:
        metric_keys_from_fold = valid_fold_results[0]["validation_metrics"].keys()
        for key in metric_keys_from_fold:
            metric_values_for_avg = [
                res["validation_metrics"][key]
                for res in valid_fold_results
                if isinstance(res["validation_metrics"].get(key), (int, float))
            ]
            if metric_values_for_avg:
                avg_val_metrics[f"avg_{key}"] = np.mean(metric_values_for_avg)
                avg_val_metrics[f"std_{key}"] = np.std(metric_values_for_avg)

    final_trained_model_object = None
    path_to_saved_final_model: Optional[str] = None
    status = "COMPLETED_NO_FINAL_MODEL_TRAINED"

    if perform_final_fit_on_full_train:
        logger.info(
            f"Performing final model fit on full training data for {model_name} (Config: {config_name})"
        )
        if tuned_model_candidate is not None and model_config.get(
            "perform_tuning", False
        ):
            final_trained_model_object = tuned_model_candidate
            logger.info(
                f"Using best estimator from tuning as final model for {model_name} (Config: {config_name})."
            )
        else:
            logger.info(
                f"Training new final model for {model_name} (Config: {config_name}) with params: {params_to_use_for_folds_and_final_model}"
            )
            final_model_instance_to_train = get_classifier(
                model_name, params_to_use_for_folds_and_final_model
            )
            try:
                use_smote_for_final_model = model_config_original.get(
                    "use_smote", False
                )
                class_weight_for_final_model = model_config_original.get(
                    "class_weight", None
                )
                if (
                    class_weight_for_final_model
                    and "class_weight" not in params_to_use_for_folds_and_final_model
                ):
                    final_model_instance_to_train.set_params(
                        class_weight=class_weight_for_final_model
                    )

                final_trained_model_object = train_model(
                    final_model_instance_to_train,
                    x_all_df,
                    y_all_series,
                    use_smote=use_smote_for_final_model,
                )
                logger.info(
                    f"Final model for {model_name} (Config: {config_name}) trained successfully."
                )
            except Exception as e_final_train:
                logger.error(
                    f"Error training final model for {model_name} (Config: {config_name}): {e_final_train}",
                    exc_info=True,
                )
                status = "COMPLETED_WITH_FINAL_TRAIN_FAILURE"

        if final_trained_model_object:
            model_filename_for_saving = f"model_{model_name}_{config_name}_{generate_params_identifier(params_to_use_for_folds_and_final_model)}.joblib"
            path_to_saved_final_model = os.path.join(
                model_specific_artifacts_dir, model_filename_for_saving
            )
            try:
                joblib.dump(final_trained_model_object, path_to_saved_final_model)
                logger.info(f"Saved final model to: {path_to_saved_final_model}")
                status = "COMPLETED"
            except Exception as e_save_model:
                logger.error(
                    f"Failed to save final model {path_to_saved_final_model}: {e_save_model}",
                    exc_info=True,
                )
                path_to_saved_final_model = None
                status = "COMPLETED_WITH_MODEL_SAVE_FAILURE"
        elif (
            status != "COMPLETED_WITH_FINAL_TRAIN_FAILURE"
        ):  # if not already set to train failure
            status = "COMPLETED_FINAL_TRAIN_NOT_ATTEMPTED_OR_FAILED_SILENTLY"

    current_config_run_summary = {
        "dataset_name": dataset_name,
        "target_stage_col": target_stage_col,
        "model_name": model_name,
        "config_name": config_name,
        "original_config_settings": model_config_original,
        "params_determined_and_used": params_to_use_for_folds_and_final_model,
        "saved_model_path": path_to_saved_final_model,
        "average_validation_metrics_cv": avg_val_metrics,
        "fold_results_cv": all_fold_metrics_for_this_config,
        "status": status,
    }

    config_run_report_filename = f"report_{model_name}_{config_name}_{generate_params_identifier(params_to_use_for_folds_and_final_model)}.json"
    path_to_config_run_report = os.path.join(
        model_specific_artifacts_dir, config_run_report_filename
    )
    try:
        with open(path_to_config_run_report, "w") as f:
            json.dump(current_config_run_summary, f, indent=4, default=str)
        logger.info(
            f"Saved detailed report for {model_name} ({config_name}) to {path_to_config_run_report}"
        )
    except Exception as e_json_report:
        logger.error(
            f"Failed to save JSON report for {model_name} ({config_name}): {e_json_report}",
            exc_info=True,
        )
    return current_config_run_summary


def run_classification_for_dataset(
    dataset_name: str,
    target_stage_col: str,
    model_configurations: dict[str, list[dict[str, Any]]],
    n_cv_splits: int = 5,
    perform_final_fit_on_full_train: bool = True,
) -> list[dict[str, Any]]:
    logger.info(
        f"--- Starting Classification Pipeline for Dataset: {dataset_name}, Target: {target_stage_col} ---"
    )
    run_timestamp_for_paths = datetime.now().strftime("%Y%m%d_%H%M%S")

    overall_run_summary_filename = (
        f"summary_run_{dataset_name}_{target_stage_col}_{run_timestamp_for_paths}.json"
    )
    overall_run_summary_path = os.path.join(
        REPORTS_DIR, "classification", overall_run_summary_filename
    )
    ensure_dir(os.path.dirname(overall_run_summary_path))

    dataset_target_report_dir = os.path.join(
        REPORTS_DIR, "classification", dataset_name, target_stage_col
    )
    ensure_dir(dataset_target_report_dir)

    x_all_df, y_all_series, cv_splits_list, unique_labels = (
        _prepare_initial_data_and_splits(dataset_name, target_stage_col, n_cv_splits)
    )

    if (
        x_all_df is None or y_all_series is None or unique_labels is None
    ):  # unique_labels implies data loading was okay
        # Error messages are already logged by the helper function
        return [
            {
                "dataset": dataset_name,
                "target_stage_col": target_stage_col,
                "status": "ERROR",
                "error_message": "Initial data preparation or CV split generation failed.",
            }
        ]
    if cv_splits_list is None:  # Specifically if splits failed but data was loaded
        return [
            {
                "dataset": dataset_name,
                "target_stage_col": target_stage_col,
                "status": "ERROR",
                "error_message": "CV split generation failed after data loading.",
            }
        ]

    overall_results_for_run_summary_file = []

    for model_name, configs_for_model in model_configurations.items():
        model_specific_artifacts_dir = os.path.join(
            dataset_target_report_dir, model_name
        )
        ensure_dir(model_specific_artifacts_dir)

        for config_idx, model_config_original in enumerate(configs_for_model):
            config_summary = _process_model_configuration(
                model_name=model_name,
                model_config_original=model_config_original,
                config_idx=config_idx,
                x_all_df=x_all_df,
                y_all_series=y_all_series,
                cv_splits_list=cv_splits_list,
                n_cv_splits=n_cv_splits,
                dataset_name=dataset_name,
                target_stage_col=target_stage_col,
                unique_labels=unique_labels,
                model_specific_artifacts_dir=model_specific_artifacts_dir,
                perform_final_fit_on_full_train=perform_final_fit_on_full_train,
            )
            overall_results_for_run_summary_file.append(config_summary)

    logger.info(
        f"--- Classification Pipeline for Dataset: {dataset_name} Completed ---"
    )
    try:
        with open(overall_run_summary_path, "w") as f:
            json.dump(overall_results_for_run_summary_file, f, indent=4, default=str)
        logger.info(
            f"Full run summary for {dataset_name} (target: {target_stage_col}) saved to {overall_run_summary_path}"
        )
    except Exception as e_overall_summary:
        logger.error(
            f"Failed to save overall run summary JSON: {e_overall_summary}",
            exc_info=True,
        )

    return overall_results_for_run_summary_file
