import hashlib  # For creating unique identifiers from params
import json  # For saving params
import logging
import os
from datetime import datetime
from typing import Any, Optional

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
from src.classification.models import get_classifier, train_model
from src.classification.tuning import DEFAULT_PARAM_GRIDS, perform_hyperparameter_tuning
from src.utils import REPORTS_DIR, ensure_dir

logger = logging.getLogger(__name__)


def generate_params_identifier(params: dict[str, Any]) -> str:
    """Creates a short, unique identifier from a dictionary of parameters."""
    # Sorting ensures that the same parameters in a different order produce the same hash
    param_string = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_string.encode("utf-8")).hexdigest()[:8]  # noqa: S324


def run_classification_pipeline_for_fold(
    model_name: str,
    x_train_df: pd.DataFrame,  # N803
    y_train_series: pd.Series,  # N803
    x_val_df: pd.DataFrame,  # N803
    y_val_series: pd.Series,  # N803
    dataset_name: str,
    target_stage_col: str,
    fold_num: int,
    model_config: dict[str, Any],  # Includes initial_params, use_smote, perform_tuning
    base_report_dir: str,
    unique_labels: list[int],
) -> dict[str, Any]:
    """
    Runs the classification pipeline for a single fold of cross-validation.
    """
    fold_results: dict[str, Any] = {"fold": fold_num, "model_name": model_name}

    initial_params = model_config.get("initial_params", {})
    use_smote_fold = model_config.get("use_smote", False)
    class_weight_fold = model_config.get(
        "class_weight", None
    )  # For models that support it

    if class_weight_fold and "class_weight" not in initial_params:
        initial_params["class_weight"] = class_weight_fold

    # --- Baseline Model Training (or Tuned if params provided) ---
    logger.info(
        f"[Fold {fold_num}] Training {model_name} with params: {initial_params}, SMOTE: {use_smote_fold}"
    )

    current_model_params = initial_params.copy()
    model_identifier_str = f"fold{fold_num}_{model_name}_smote{use_smote_fold}"
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
        fold_results["initial_params"] = current_model_params
        fold_results["smote_applied"] = use_smote_fold
        fold_results["validation_metrics"] = val_metrics

        # Save confusion matrix for this fold's validation
        cm_path = plot_confusion_matrix(
            y_val_series,
            y_pred_val,
            labels=unique_labels,
            dataset_name=dataset_name,
            model_name=model_name,
            file_identifier=model_identifier_str,
            save_dir_suffix="fold_validations",
        )
        fold_results["confusion_matrix_val_path"] = cm_path

        # Feature importance (optional, can be averaged later)
        if x_train_df.columns.tolist():  # type: ignore[union-attr]
            feat_importance_df = get_feature_importance(
                trained_model,
                x_train_df.columns.tolist(),  # type: ignore[union-attr]
            )
            if feat_importance_df is not None:
                fold_results["feature_importances_sample"] = feat_importance_df.head(
                    10
                ).to_dict(orient="records")
                # Optionally plot per-fold importances, or average them later
                # plot_feature_importance(feat_importance_df, dataset_name, model_name, model_identifier_str, save_dir_suffix="fold_validations")

    except Exception as e:
        logger.error(
            f"[Fold {fold_num}] Error during model training/evaluation for {model_name}: {e}",
            exc_info=True,
        )
        fold_results["error"] = str(e)

    return fold_results


def run_classification_for_dataset(  # noqa: C901
    dataset_name: str,
    target_stage_col: str,  # e.g., "kmeans_stage"
    model_configurations: dict[
        str, list[dict[str, Any]]
    ],  # model_name -> list of configs
    n_cv_splits: int = 5,
    perform_final_fit_on_full_train: bool = True,  # This parameter is not used in the current logic
) -> list[dict[str, Any]]:  # List of results per model configuration
    """
    Main orchestrator for running classification on a single dataset.
    A model configuration is a dict: {"name": "config1", "initial_params": {}, "use_smote": False, "perform_tuning": True, "class_weight": None}
    """
    logger.info(
        f"--- Starting Classification Pipeline for Dataset: {dataset_name}, Target: {target_stage_col} ---"
    )

    run_summary_path = os.path.join(
        REPORTS_DIR,
        "classification",
        f"summary_run_{dataset_name}_{target_stage_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    dataset_report_dir = os.path.join(
        REPORTS_DIR, "classification", dataset_name, target_stage_col
    )
    ensure_dir(dataset_report_dir)

    df_merged = load_and_merge_data(dataset_name)
    if df_merged is None:
        logger.error(f"Failed to load or merge data for {dataset_name}. Aborting.")
        return [{"dataset": dataset_name, "error": "Data loading failed"}]

    x_all_df: Optional[pd.DataFrame]
    y_all_series: Optional[pd.Series]
    x_all_df, y_all_series = prepare_features_target(
        df_merged, target_stage_col
    )  # N806

    if x_all_df is None or y_all_series is None:
        logger.error(f"Failed to prepare features/target for {dataset_name}. Aborting.")
        return [{"dataset": dataset_name, "error": "Feature/target preparation failed"}]

    if "engine_id" not in df_merged.columns:
        logger.error("'engine_id' column not found for GroupKFold splitting. Aborting.")
        return [{"dataset": dataset_name, "error": "'engine_id' missing"}]

    # Align engine_ids with x_all_df, y_all_series after potential row drops in prepare_features_target
    engine_ids_all = df_merged.loc[x_all_df.index, "engine_id"]

    unique_labels = sorted(y_all_series.unique().tolist())
    logger.info(f"Unique target labels found: {unique_labels}")

    cv_splits = split_data_grouped(
        x_all_df,
        y_all_series,
        engine_ids_all,
        n_splits=n_cv_splits,
        use_group_kfold=True,
    )
    if not cv_splits:
        logger.error("Failed to generate CV splits. Aborting.")
        return [{"dataset": dataset_name, "error": "CV split generation failed"}]

    overall_results = []

    for model_name, configs in model_configurations.items():
        for config_idx, model_config in enumerate(configs):
            config_name = model_config.get("name", f"config_{config_idx + 1}")
            logger.info(
                f"--- Processing Model: {model_name}, Config: {config_name} ---"
            )

            # Store results for each fold for this model configuration
            all_fold_metrics_for_config = []

            perform_tuning_config = model_config.get("perform_tuning", False)
            best_params_from_tuning = model_config.get(
                "initial_params", {}
            ).copy()  # Start with initial, override if tuned

            if perform_tuning_config:
                logger.info(
                    f"Performing Hyperparameter Tuning for {model_name} using first CV split for selection."
                )
                # For tuning, we typically use the CV splits within GridSearchCV/RandomizedSearchCV
                # The `cv_splits` here are GroupKFold, suitable for this.
                param_grid = model_config.get(
                    "param_grid", DEFAULT_PARAM_GRIDS.get(model_name)
                )
                if not param_grid:
                    logger.warning(
                        f"No param_grid defined for {model_name} in config or defaults. Skipping tuning."
                    )
                else:
                    # Get a base model instance
                    base_model_for_tuning = get_classifier(model_name, random_state=42)

                    # Use full x_all_df, y_all_series for tuning with GroupKFold CV
                    tuned_model_instance, best_params_cv = (
                        perform_hyperparameter_tuning(
                            base_model_for_tuning,
                            x_all_df,  # type: ignore[arg-type] # x_all_df is checked not None
                            y_all_series,  # type: ignore[arg-type] # y_all_series is checked not None
                            param_grid,
                            cv_splits=cv_splits,  # Pass GroupKFold splits
                            scoring=model_config.get("tuning_scoring", "f1_macro"),
                        )
                    )
                    best_params_from_tuning = (
                        best_params_cv  # Use these for subsequent fold training
                    )
                    logger.info(
                        f"Tuned best params for {model_name}: {best_params_from_tuning}"
                    )

            # Cross-validation loop
            for i, (train_idx, val_idx) in enumerate(cv_splits):
                logger.info(
                    f"--- Fold {i + 1}/{n_cv_splits} for {model_name}, Config: {config_name} ---"
                )
                x_train_fold_df, x_val_fold_df = (
                    x_all_df.iloc[train_idx],
                    x_all_df.iloc[val_idx],
                )  # N806
                y_train_fold_series, y_val_fold_series = (
                    y_all_series.iloc[train_idx],
                    y_all_series.iloc[val_idx],
                )

                # Pass the potentially tuned parameters to the fold runner
                current_fold_model_config = model_config.copy()
                current_fold_model_config["initial_params"] = (
                    best_params_from_tuning  # Use tuned params
                )

                fold_run_result = run_classification_pipeline_for_fold(
                    model_name=model_name,
                    x_train_df=x_train_fold_df,
                    y_train_series=y_train_fold_series,
                    x_val_df=x_val_fold_df,
                    y_val_series=y_val_fold_series,
                    dataset_name=dataset_name,
                    target_stage_col=target_stage_col,
                    fold_num=i + 1,
                    model_config=current_fold_model_config,  # Pass updated config with tuned params
                    base_report_dir=dataset_report_dir,
                    unique_labels=unique_labels,
                )
                all_fold_metrics_for_config.append(fold_run_result)

            # Aggregate results for this model configuration
            avg_val_metrics: dict[str, Any] = {}
            if all_fold_metrics_for_config:
                # Example: average f1_macro across folds
                # More sophisticated aggregation can be done
                valid_fold_results = [
                    f_res
                    for f_res in all_fold_metrics_for_config
                    if "validation_metrics" in f_res
                ]
                if valid_fold_results:
                    # Iterate over keys present in the first valid result's validation_metrics
                    # This assumes all valid folds have the same metric keys.
                    for key in valid_fold_results[0]["validation_metrics"].keys():
                        # Ensure the metric is numeric before trying to calculate mean
                        metric_values = [
                            res["validation_metrics"][key]
                            for res in valid_fold_results
                            if isinstance(
                                res["validation_metrics"].get(key), (int, float)
                            )
                        ]
                        if (
                            metric_values
                        ):  # Only average if there are numeric values for this key
                            avg_val_metrics[f"avg_{key}"] = np.mean(metric_values)

            config_summary = {
                "dataset_name": dataset_name,
                "target_stage_col": target_stage_col,
                "model_name": model_name,
                "config_name": config_name,
                "model_initial_config": model_config,  # Original config for this run
                "tuned_params_used": (
                    best_params_from_tuning
                    if perform_tuning_config
                    else model_config.get("initial_params")
                ),
                "average_validation_metrics": avg_val_metrics,
                "fold_results": all_fold_metrics_for_config,
            }
            overall_results.append(config_summary)

            # Save detailed report for this configuration
            config_report_filename = f"{model_name}_{config_name}_{generate_params_identifier(best_params_from_tuning)}.json"
            config_report_path = os.path.join(
                dataset_report_dir, model_name, config_report_filename
            )
            ensure_dir(os.path.dirname(config_report_path))
            with open(config_report_path, "w") as f:
                json.dump(
                    config_summary,
                    f,
                    indent=4,
                    default=lambda o: str(o),  # More robust default
                )
            logger.info(
                f"Saved detailed report for {model_name} ({config_name}) to {config_report_path}"
            )

    logger.info(
        f"--- Classification Pipeline for Dataset: {dataset_name} Completed ---"
    )
    # Save overall summary for the dataset run
    with open(run_summary_path, "w") as f:
        json.dump(
            overall_results, f, indent=4, default=lambda o: str(o)
        )  # More robust default
    logger.info(f"Full run summary saved to {run_summary_path}")
    return overall_results
