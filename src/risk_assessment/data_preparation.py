import logging
import os
from typing import Any, Optional

import joblib  # For loading scikit-learn models
import pandas as pd

from src.utils import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def load_model_and_data_for_risk_assessment(
    dataset_name: str,
    model_path: str,
    feature_file_suffix: str = "_features.parquet",
    clustered_label_file_suffix: str = "_clustered_all_algos.parquet",
    # This line uses Any, so the import is needed
) -> tuple[Optional[Any], Optional[pd.DataFrame]]:
    """
    Loads a trained classifier model and the required data for risk assessment.
    The data loaded is the merged features and cluster labels, which the model
    would expect for making predictions.

    Args:
        dataset_name: Identifier for the dataset (e.g., "FD001").
        model_path: Full path to the saved trained classifier model.
        feature_file_suffix: Suffix for the features parquet file.
        clustered_label_file_suffix: Suffix for the clustered label parquet file.

    Returns:
        A tuple (trained_model, df_assessment_data).
        df_assessment_data contains features, engine_id, cycle, and original stage labels.
        Returns (None, None) if loading fails.
    """
    logger.info(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None
    try:
        trained_model = joblib.load(model_path)
        logger.info(f"Successfully loaded model: {trained_model.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None, None

    # --- Load data (similar to classification data prep) ---
    clustered_labels_dir = os.path.join(PROCESSED_DATA_DIR, "clustered_labels")
    features_dir = os.path.join(PROCESSED_DATA_DIR, "features")

    cluster_file_path = os.path.join(
        clustered_labels_dir, f"{dataset_name}{clustered_label_file_suffix}"
    )
    feature_file_path = os.path.join(
        features_dir, f"{dataset_name}{feature_file_suffix}"
    )

    if not os.path.exists(cluster_file_path):
        logger.error(f"Cluster label file not found: {cluster_file_path}")
        # Return the model even if data loading fails partially, might be useful? Or return None? Let's return None for data consistency.
        # Returning trained_model, None might be misleading if data is required.
        return None, None
    if not os.path.exists(feature_file_path):
        logger.error(f"Feature file not found: {feature_file_path}")
        return None, None

    try:
        df_clusters = pd.read_parquet(cluster_file_path)
        df_features = pd.read_parquet(feature_file_path)

        # Keep essential columns for identification and potential feature use
        # The features used for prediction are in df_features.
        # df_clusters mainly provides engine_id, cycle, and true stages for context.
        cluster_cols_to_keep = ["engine_id", "cycle"] + [
            col for col in df_clusters.columns if "_stage" in col or "RUL" in col
        ]
        # Ensure we only take specified columns if they exist
        cluster_cols_to_keep = [
            c for c in cluster_cols_to_keep if c in df_clusters.columns
        ]

        df_assessment_data = pd.merge(
            df_features,
            df_clusters[cluster_cols_to_keep],
            on=["engine_id", "cycle"],
            how="inner",
        )
        logger.info(
            f"Loaded and merged data for assessment. Shape: {df_assessment_data.shape}"
        )
        if df_assessment_data.empty:
            logger.warning(
                "Merged data for assessment is empty. Check keys or data alignment."
            )
            return trained_model, None  # Return model but indicate data issue
        # If merge succeeded and is not empty, return both
        return trained_model, df_assessment_data
    except Exception as e:
        logger.error(f"Error loading or merging data for {dataset_name}: {e}")
        # Return None for data part on error
        return trained_model, None


def prepare_features_for_prediction(
    df_assessment: pd.DataFrame, model_feature_names: list[str]
) -> Optional[pd.DataFrame]:
    """
    Prepares the feature set from the assessment data that the loaded model expects.
    Handles potential NaNs in features by mean imputation.

    Args:
        df_assessment: DataFrame containing all loaded data.
        model_feature_names: List of feature names the model was trained on.

    Returns:
        DataFrame (x_predict) ready for model.predict_proba() or None if error.
    """
    missing_model_features = [
        f for f in model_feature_names if f not in df_assessment.columns
    ]
    if missing_model_features:
        logger.error(
            f"Assessment data is missing features the model was trained on: {missing_model_features}"
        )
        return None

    x_predict = df_assessment[model_feature_names].copy()

    # Handle NaNs in features (e.g., from rolling windows at the start of series)
    if x_predict.isnull().any().any():
        logger.warning(
            "NaNs found in features for prediction. Imputing with column means."
        )
        # Impute NaNs using the mean of each respective column
        for col in x_predict.columns[x_predict.isnull().any()]:
            mean_val = x_predict[col].mean()
            x_predict[col] = x_predict[col].fillna(mean_val)
            logger.debug(f"Imputed NaNs in column '{col}' with mean {mean_val:.4f}")

    if x_predict.empty:
        logger.error("Feature set x_predict is empty after pre-processing.")
        return None

    # Final check after imputation
    if x_predict.isnull().any().any():
        logger.error("NaNs still present after imputation. Check imputation logic.")
        # Log columns with NaNs
        nan_cols_after_impute = x_predict.columns[x_predict.isnull().any()].tolist()
        logger.error(f"Columns with NaNs after imputation: {nan_cols_after_impute}")
        # Log a sample of rows with NaNs
        rows_with_nan = x_predict[x_predict.isnull().any(axis=1)]
        logger.error(f"Sample rows with NaNs after imputation:\n{rows_with_nan.head()}")
        return None

    return x_predict
