import hashlib
import json
import logging
import os
from typing import (
    Any,
    Optional,
)

# Removed Dict and added Optional for a helper function
import numpy as np
import pandas as pd

from src.risk_assessment.calculation import (
    calculate_risk_scores,
    normalize_risk_scores,
)
from src.risk_assessment.data_preparation import (
    load_model_and_data_for_risk_assessment,
    prepare_features_for_prediction,
)
from src.risk_assessment.decision import (
    document_threshold_tuning_strategy,
    generate_alerts,
)
from src.risk_assessment.visualization import plot_risk_score_trends
from src.utils import REPORTS_DIR, ensure_dir

logger = logging.getLogger(__name__)


def _save_operation_summary(summary: dict[str, Any], file_path: str) -> None:
    """Helper function to save operation summary to a JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(summary, f, indent=4, default=str)
        logger.info(f"Operation summary saved to {file_path}")
    except Exception as write_e:
        logger.error(f"Failed to write operation summary to {file_path}: {write_e}")


def _load_and_prepare_data(
    dataset_name: str,
    trained_model_path: str,
    operation_summary: dict[str, Any],
    report_dir_dataset: str,
    report_identifier_str: str,
) -> tuple[Optional[Any], Optional[pd.DataFrame], Optional[list[str]]]:
    """Loads model and data, and infers feature names."""
    logger.info("Step 1: Loading model and assessment data...")
    model, df_assessment_full = load_model_and_data_for_risk_assessment(
        dataset_name, trained_model_path
    )
    if model is None or df_assessment_full is None:
        logger.error("Failed to load model or data.")
        operation_summary["status"] = "FAILED_DATA_LOAD"
        operation_summary["error_message"] = "Model or assessment data loading failed."
        summary_path = os.path.join(
            report_dir_dataset, f"summary_{report_identifier_str}.json"
        )
        _save_operation_summary(operation_summary, summary_path)
        return None, None, None

    try:
        if hasattr(model, "feature_names_in_"):
            model_feature_names = model.feature_names_in_.tolist()
            logger.info(f"Inferred model feature names: {model_feature_names[:5]}...")
        elif hasattr(model, "steps"):  # Check pipeline
            final_estimator = model.steps[-1][1]
            if hasattr(final_estimator, "feature_names_in_"):
                model_feature_names = final_estimator.feature_names_in_.tolist()
                logger.info(
                    f"Inferred model feature names from pipeline's final step: {model_feature_names[:5]}..."
                )
            elif len(model.steps) > 1 and hasattr(
                model.steps[-2][1], "get_feature_names_out"
            ):
                # This case remains complex and might need more specific handling depending on the preprocessor
                logger.warning(
                    "Attempting to infer feature names from a preprocessor step. This might be unreliable."
                )
                # Assuming the step before the final one is a transformer with get_feature_names_out
                preprocessor = model.steps[-2][1]
                # This part is tricky as get_feature_names_out might need input_features if they changed names
                # For simplicity, we'll try without, but this is a common point of failure
                model_feature_names = preprocessor.get_feature_names_out().tolist()
                logger.info(
                    f"Inferred feature names from preprocessor: {model_feature_names[:5]}..."
                )
            else:
                raise AttributeError(
                    "Final estimator in pipeline does not have 'feature_names_in_' and previous steps lack standard output names."
                )
        else:
            raise AttributeError(
                f"Model {model.__class__.__name__} does not have 'feature_names_in_'. "
                "Feature names must be known to prepare data."
            )
        return model, df_assessment_full, model_feature_names
    except AttributeError as e:
        logger.error(f"Could not determine feature names from the loaded model: {e}")
        operation_summary["status"] = "FAILED_FEATURE_NAME_INFERENCE"
        operation_summary["error_message"] = str(e)
        summary_path = os.path.join(
            report_dir_dataset, f"summary_{report_identifier_str}.json"
        )
        _save_operation_summary(operation_summary, summary_path)
        return None, None, None


def _predict_stages_and_probabilities(
    model: Any,
    x_predict: pd.DataFrame,
    df_assessment_full: pd.DataFrame,
    max_possible_stage_val: int,
    operation_summary: dict[str, Any],
    report_dir_dataset: str,
    report_identifier_str: str,
) -> Optional[pd.DataFrame]:
    """Gets stage predictions and probabilities from the model."""
    logger.info("Step 3: Getting stage predictions and probabilities...")
    try:
        predicted_stages_raw = model.predict(x_predict)
        if not hasattr(model, "predict_proba"):
            logger.warning(
                f"Model {model.__class__.__name__} does not support predict_proba. Cannot calculate probabilities."
            )
            operation_summary["status"] = "FAILED_PREDICTION_PROBA"
            operation_summary["error_message"] = "Model lacks predict_proba method."
            summary_path = os.path.join(
                report_dir_dataset, f"summary_{report_identifier_str}.json"
            )
            _save_operation_summary(operation_summary, summary_path)
            return None
        predicted_probas_raw = model.predict_proba(x_predict)
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        operation_summary["status"] = "FAILED_PREDICTION"
        operation_summary["error_message"] = str(e)
        summary_path = os.path.join(
            report_dir_dataset, f"summary_{report_identifier_str}.json"
        )
        _save_operation_summary(operation_summary, summary_path)
        return None

    df_assessment_full["predicted_stage"] = pd.Series(
        predicted_stages_raw, index=x_predict.index
    )

    num_classes = predicted_probas_raw.shape[1]
    if max_possible_stage_val != num_classes - 1:
        logger.warning(
            f"max_possible_stage_val ({max_possible_stage_val}) from args does not match "
            f"model's number of classes - 1 ({num_classes - 1}). Using model's class count for probability indexing."
        )

    proba_col_stage4_idx = num_classes - 1
    for i in range(num_classes):
        df_assessment_full[f"proba_stage_{i}"] = pd.Series(
            predicted_probas_raw[:, i], index=x_predict.index
        )

    df_assessment_full["predicted_stage"] = (
        df_assessment_full["predicted_stage"].fillna(-1).astype(int)
    )
    for i in range(num_classes):
        df_assessment_full[f"proba_stage_{i}"] = df_assessment_full[
            f"proba_stage_{i}"
        ].fillna(0.0)

    operation_summary["internal_num_classes_from_model"] = (
        num_classes  # Store for later use
    )
    logger.info(
        f"Predicted stages and probabilities added. P(Stage {proba_col_stage4_idx}) sample: {df_assessment_full[f'proba_stage_{proba_col_stage4_idx}'].head().tolist()}"
    )
    return df_assessment_full


def _calculate_and_normalize_risk(
    df_assessment_full: pd.DataFrame,
    max_possible_stage_val: int,
    risk_score_formula: str,
    normalization_method: str,
    proba_col_stage4_idx: int,
) -> pd.DataFrame:
    """Calculates raw and normalized risk scores."""
    logger.info("Step 4: Calculating raw risk scores...")
    proba_col_name_for_calc = f"proba_stage_{proba_col_stage4_idx}"
    df_assessment_full["raw_risk_score"] = calculate_risk_scores(
        df_assessment_full,
        proba_col_stage4=proba_col_name_for_calc,
        predicted_stage_col="predicted_stage",
        max_possible_stage=max_possible_stage_val,
        formula_type=risk_score_formula,
    )
    df_assessment_full["raw_risk_score"].replace(
        [np.inf, -np.inf], np.nan, inplace=True
    )
    if df_assessment_full["raw_risk_score"].isnull().any():
        max_non_nan_score = df_assessment_full["raw_risk_score"].max()
        logger.warning(
            f"Risk scores contained Inf or NaN, replaced with max finite score: {max_non_nan_score if pd.notna(max_non_nan_score) else 'NaN'}"
        )
        df_assessment_full["raw_risk_score"].fillna(
            max_non_nan_score if pd.notna(max_non_nan_score) else 0, inplace=True
        )
    logger.info(
        f"Raw risk scores calculated. Sample: {df_assessment_full['raw_risk_score'].head().tolist()}"
    )

    logger.info("Step 5: Normalizing risk scores...")
    df_assessment_full["normalized_risk_score"] = normalize_risk_scores(
        df_assessment_full["raw_risk_score"], method=normalization_method
    )
    logger.info(
        f"Normalized risk scores calculated. Sample: {df_assessment_full['normalized_risk_score'].head().tolist()}"
    )
    return df_assessment_full


def _save_assessment_results(
    df_assessment: pd.DataFrame,
    target_stage_col: str,
    num_classes: int,
    report_dir: str,
    report_id_str: str,
    operation_summary: dict[str, Any],
) -> None:
    """Saves the final assessment results to a parquet file."""
    logger.info("Step 8: Saving assessment results...")
    results_filename = f"risk_assessment_results_{report_id_str}.parquet"
    results_path = os.path.join(report_dir, results_filename)
    try:
        cols_to_save = [
            "engine_id",
            "cycle",
            target_stage_col,  # No longer target_stage_col_name_for_classification
            "predicted_stage",
            f"proba_stage_{num_classes - 1}",  # Use num_classes from model
            "raw_risk_score",
            "normalized_risk_score",
            "maintenance_alert",
        ]
        for i in range(num_classes):  # Use num_classes from model
            cols_to_save.append(f"proba_stage_{i}")
        if "RUL" in df_assessment.columns:
            cols_to_save.append("RUL")

        final_cols_to_save = [c for c in cols_to_save if c in df_assessment.columns]
        missing_cols = set(cols_to_save) - set(final_cols_to_save)
        if missing_cols:
            logger.warning(
                f"Columns intended for saving but not found in final DataFrame: {missing_cols}"
            )

        df_assessment[final_cols_to_save].to_parquet(results_path, index=False)
        logger.info(f"Risk assessment results saved to: {results_path}")
        operation_summary["artifacts"]["results_parquet"] = results_path
    except Exception as e:
        logger.error(
            f"Failed to save risk assessment results to {results_path}: {e}",
            exc_info=True,
        )
        operation_summary["status"] = "FAILED_SAVE_RESULTS"
        operation_summary["error_message"] = str(e)
        # Summary will be saved by the main function in its finally block or at the end


# --- THIS IS THE FUNCTION SIGNATURE TO CHECK CAREFULLY ---
def run_risk_assessment_for_dataset(
    dataset_name: str,
    trained_model_path: str,
    target_stage_col_name_for_classification: str,
    max_possible_stage_val: int,
    run_timestamp: str,  # THIS ARGUMENT HAS NO DEFAULT
    # --- ARGUMENTS BELOW HAVE DEFAULTS ---
    risk_score_formula: str = "urgency_inverted",
    normalization_method: str = "min_max",
    alert_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Orchestrates the risk assessment pipeline for a given dataset and trained model.

    Args:
        dataset_name (str): e.g., "FD001"
        trained_model_path (str): Path to the saved trained classifier model.
        target_stage_col_name_for_classification (str): The stage column used when training the classifier.
                                                         Used to find true stages for context if available.
        max_possible_stage_val (int): Max value for a stage (e.g., 4 if stages 0-4).
        run_timestamp (str): Timestamp for this run, for report naming.
        risk_score_formula (str): Formula to use for risk calculation.
        normalization_method (str): Method for normalizing risk scores.
        alert_threshold (float): Threshold for generating alerts.


    Returns:
        A dictionary containing the summary of the risk assessment run.
    """
    logger.info(
        f"--- Starting Risk Assessment Pipeline for Dataset: {dataset_name} ---"
    )
    operation_summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "trained_model_path": trained_model_path,
        "risk_score_formula": risk_score_formula,
        "normalization_method": normalization_method,
        "alert_threshold": alert_threshold,
        "status": "INITIATED",
        "artifacts": {},
    }

    # Create unique identifier for this specific risk assessment run config
    config_str = f"{dataset_name}_{os.path.basename(trained_model_path)}_{risk_score_formula}_{normalization_method}"
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[
        :10
    ]  # Changed to sha256 and 10 chars
    report_identifier_str = f"risk_run_{config_hash}_{run_timestamp}"

    report_dir_dataset = os.path.join(
        REPORTS_DIR, "risk_assessment", dataset_name, report_identifier_str
    )
    image_dir_dataset_run_rel = (
        os.path.join(  # Define image subfolder relative path for visualization function
            "risk_assessment", dataset_name, report_identifier_str
        )
    )
    ensure_dir(report_dir_dataset)
    summary_path = os.path.join(
        report_dir_dataset, f"summary_{report_identifier_str}.json"
    )

    try:
        model, df_assessment_full, model_feature_names = _load_and_prepare_data(
            dataset_name,
            trained_model_path,
            operation_summary,
            report_dir_dataset,
            report_identifier_str,
        )
        if model is None or df_assessment_full is None or model_feature_names is None:
            # Error already logged and summary updated by helper
            return operation_summary

        # 2. Prepare Features for Prediction
        logger.info("Step 2: Preparing features for prediction...")
        x_predict = prepare_features_for_prediction(
            df_assessment_full, model_feature_names
        )
        if x_predict is None:
            logger.error("Failed to prepare features for prediction. Aborting.")
            operation_summary["status"] = "FAILED_FEATURE_PREP"
            operation_summary["error_message"] = (
                "Feature preparation for prediction failed."
            )
            _save_operation_summary(operation_summary, summary_path)
            return operation_summary

        # 3. Get Stage Predictions and Probabilities
        df_assessment_full = _predict_stages_and_probabilities(
            model,
            x_predict,
            df_assessment_full,
            max_possible_stage_val,
            operation_summary,
            report_dir_dataset,
            report_identifier_str,
        )
        if df_assessment_full is None:
            # Error handled and summary saved in helper
            return operation_summary

        num_classes_from_model = operation_summary.get(
            "internal_num_classes_from_model", max_possible_stage_val + 1
        )

        # 4. Calculate Raw Risk Score & 5. Normalize Risk Score
        df_assessment_full = _calculate_and_normalize_risk(
            df_assessment_full,
            max_possible_stage_val,
            risk_score_formula,
            normalization_method,
            num_classes_from_model - 1,
        )

        # 6. Generate Alerts
        logger.info("Step 6: Generating alerts...")
        df_assessment_full["maintenance_alert"] = generate_alerts(
            df_assessment_full,
            normalized_risk_score_col="normalized_risk_score",
            threshold=alert_threshold,
        )
        operation_summary["alerts_generated"] = int(
            df_assessment_full["maintenance_alert"].sum()
        )
        operation_summary["total_data_points_assessed"] = len(df_assessment_full)

        # 7. Visualization
        logger.info("Step 7: Generating risk score trend plots...")
        try:
            plot_paths_risk_norm = plot_risk_score_trends(
                df_assessment_full,
                risk_score_col="normalized_risk_score",
                dataset_name=dataset_name,
                config_identifier=f"norm_{report_identifier_str}",
                report_image_subfolder=image_dir_dataset_run_rel,  # Pass the relative path here
                num_sample_engines=5,
            )
            operation_summary["artifacts"]["normalized_risk_score_plots"] = (
                plot_paths_risk_norm
            )
        except Exception as e:
            logger.error(
                f"Error plotting normalized risk score trends: {e}", exc_info=True
            )
            # Continue, plotting is not critical for core assessment

        # 8. Save Results
        _save_assessment_results(
            df_assessment_full,
            target_stage_col_name_for_classification,
            num_classes_from_model,  # Pass num_classes from model
            report_dir_dataset,
            report_identifier_str,
            operation_summary,
        )
        if operation_summary["status"].startswith("FAILED_SAVE_RESULTS"):
            _save_operation_summary(
                operation_summary, summary_path
            )  # Save summary if saving results failed
            return operation_summary

        # 9. Document Threshold Tuning
        operation_summary["threshold_tuning_notes"] = (
            document_threshold_tuning_strategy()
        )
        operation_summary["status"] = "COMPLETED_SUCCESSFULLY"
        logger.info(
            f"--- Risk Assessment Pipeline for Dataset: {dataset_name} Completed Successfully ---"
        )

    except Exception as e:  # Catch any other unexpected error
        logger.error(
            f"An unexpected error occurred in the risk assessment pipeline: {e}",
            exc_info=True,
        )
        operation_summary["status"] = "FAILED_UNEXPECTED"
        operation_summary["error_message"] = str(e)
    finally:
        # Always save the final (or partial on error) summary
        _save_operation_summary(operation_summary, summary_path)

    return operation_summary
