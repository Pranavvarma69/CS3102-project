from typing import Any

import numpy as np
import pandas as pd


# (1) Get stage probabilities from classifier (dummy example)
def get_stage_probabilities(classifier: Any, x_features: np.ndarray) -> np.ndarray:
    """
    Replace this with classifier.predict_proba(x_features)
    Here, returns dummy probabilities assuming 5 classes.
    """
    np.random.seed(0)
    dummy_probs = np.random.dirichlet(np.ones(5), size=len(x_features))
    return dummy_probs


# (2) Predict time to next stage from regressor (dummy example)
def predict_time_to_stage(regressor: Any, x_features: np.ndarray) -> np.ndarray:
    """
    Replace this with regressor.predict(x_features)
    Here, returns dummy predictions between 0 and 100.
    """
    np.random.seed(1)
    return np.random.uniform(low=0, high=100, size=len(x_features))


# (3) Calculate risk score
def calculate_risk_score(
    prob_stage_4: np.ndarray,
    time_left: np.ndarray,
    max_time: float = 300,
    method: str = "default",
) -> np.ndarray:
    """Calculate risk score using numpy."""
    if method == "default":
        epsilon = 1e-6
        risk = prob_stage_4 / (time_left + epsilon)
    elif method == "alternative":
        risk = prob_stage_4 * (max_time - time_left)
    else:
        print("Invalid method. Use 'default' or 'alternative'.")
        risk = np.zeros_like(prob_stage_4)
    return risk


# (4) Normalize risk scores using min-max scaling (manual with numpy)
def normalize_risk_score(risk_scores: np.ndarray) -> np.ndarray:
    min_val = np.min(risk_scores)
    max_val = np.max(risk_scores)
    if max_val - min_val == 0:
        return np.zeros_like(risk_scores)
    return (risk_scores - min_val) / (max_val - min_val)


# (5) Find best threshold based on F1 score (no sklearn)
def find_best_threshold(y_true: np.ndarray, risk_scores: np.ndarray) -> float:
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_thresh = 0

    for thresh in thresholds:
        y_pred = (risk_scores >= thresh).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh


# (6) Print table of risk scores over cycles
def print_risk_trend_table(
    engine_cycles: np.ndarray, risk_scores: np.ndarray, engine_id: np.ndarray
) -> None:
    df = pd.DataFrame(
        {
            "Engine_ID": engine_id,
            "Cycle": engine_cycles,
            "Normalized_Risk_Score": risk_scores,
        }
    )
    print(df)


# (7) Test functions
def test_functions() -> None:
    probs = np.array([0.1, 0.3, 0.6, 0.9])
    times = np.array([100, 50, 25, 10])
    risk = calculate_risk_score(probs, times)
    normalized_risk = normalize_risk_score(risk)
    assert normalized_risk.min() >= 0 and normalized_risk.max() <= 1
    print("✅ All tests passed!")
