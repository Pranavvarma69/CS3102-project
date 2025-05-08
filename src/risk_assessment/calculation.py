import logging

import pandas as pd

logger = logging.getLogger(__name__)

EPSILON = 1e-6  # To prevent division by zero


def calculate_risk_scores(
    df_with_predictions: pd.DataFrame,
    proba_col_stage4: str,  # e.g., "proba_stage_4"
    predicted_stage_col: str,  # e.g., "predicted_stage"
    max_possible_stage: int,  # e.g., 4 for stages 0-4
    formula_type: str = "urgency_inverted",
) -> pd.Series:
    """
    Calculates risk scores based on predicted stage probabilities and current predicted stage.

    Args:
        df_with_predictions: DataFrame with probability of stage 4 and predicted stage.
        proba_col_stage4: Column name for P(Stage 4).
        predicted_stage_col: Column name for the classifier's predicted stage.
        max_possible_stage: The highest numerical value a stage can take (e.g., 4).
        formula_type: Type of risk score formula:
            - "urgency_inverted": P(Stage 4) / ((MaxStage - PredictedStage) + epsilon)
            - "severity_product": P(Stage 4) * (PredictedStage + 1)

    Returns:
        A pandas Series with calculated raw risk scores.
    """
    logger.info(f"Calculating risk scores using formula: {formula_type}")
    if proba_col_stage4 not in df_with_predictions.columns:
        logger.error(f"Probability column '{proba_col_stage4}' not found.")
        raise ValueError(f"Probability column '{proba_col_stage4}' not found.")
    if predicted_stage_col not in df_with_predictions.columns:
        logger.error(f"Predicted stage column '{predicted_stage_col}' not found.")
        raise ValueError(f"Predicted stage column '{predicted_stage_col}' not found.")

    failure_prob = df_with_predictions[proba_col_stage4]
    current_stage = df_with_predictions[predicted_stage_col]

    if formula_type == "urgency_inverted":
        # Higher stage means smaller denominator (more urgent)
        # Time proxy: (max_possible_stage - current_stage).
        # If current_stage is max_possible_stage, time_proxy is 0.
        time_proxy = max_possible_stage - current_stage
        risk_score = failure_prob / (time_proxy + EPSILON)
        # Cap risk score if time_proxy is 0 and failure_prob is high (e.g. P(S4)/epsilon)
        # This can lead to very large numbers. Consider an alternative for Stage 4 itself.
        # For Stage 4, risk could be directly P(Stage 4) * some_high_multiplier or just P(Stage 4)
        # For now, using the direct formula.
    elif formula_type == "severity_product":
        # Higher stage means higher multiplier (more severe)
        severity_factor = current_stage + 1  # Stage 0 -> 1, Stage 4 -> 5
        risk_score = failure_prob * severity_factor
    else:
        logger.error(f"Unknown risk score formula_type: {formula_type}")
        raise ValueError(f"Unknown risk score formula_type: {formula_type}")

    return pd.Series(risk_score, name="raw_risk_score")


def normalize_risk_scores(risk_scores: pd.Series, method: str = "min_max") -> pd.Series:
    """
    Normalizes risk scores.

    Args:
        risk_scores: Series of raw risk scores.
        method: Normalization method ("min_max").

    Returns:
        A pandas Series with normalized risk scores.
    """
    logger.info(f"Normalizing risk scores using method: {method}")
    if risk_scores.empty:
        logger.warning("Risk scores series is empty, cannot normalize.")
        return risk_scores

    if method == "min_max":
        min_score = risk_scores.min()
        max_score = risk_scores.max()
        if max_score == min_score:  # Avoid division by zero if all scores are the same
            # If all scores are same, either 0, 0.5, or 1 depending on preference for constant input.
            # Let's assume 0 if min=max=0, else 0.5 or 1.
            # If range is zero, normalized scores are 0 if min_score is also 0, else 1 (or 0.5).
            # A common practice is to return 0.0 or 0.5 for all if range is zero.
            return pd.Series(
                0.5 if min_score != 0 else 0.0,
                index=risk_scores.index,
                name="normalized_risk_score",
            )
        normalized = (risk_scores - min_score) / (max_score - min_score)
    else:
        logger.error(f"Unknown normalization method: {method}")
        raise ValueError(f"Unknown normalization method: {method}")

    return pd.Series(normalized, name="normalized_risk_score")
