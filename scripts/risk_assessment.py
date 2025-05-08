import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.risk_assessment.orchestrator import (  # noqa: E402
    run_risk_assessment_for_dataset,
)
from src.utils import REPORTS_DIR, ensure_dir  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATASET_NAMES = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_TARGET_STAGE_COL_PREFIX = "kmeans_stage"  # Example, from clustering
DEFAULT_MAX_STAGE_VALUE = 4  # Assuming stages 0, 1, 2, 3, 4 (total 5 stages)


def main(args) -> None:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"--- Starting Risk Assessment Script --- Timestamp: {run_timestamp}")

    # Ensure base directories exist
    ensure_dir(os.path.join(REPORTS_DIR, "risk_assessment"))
    # Image directories are handled by the orchestrator/visualization modules

    if not os.path.exists(args.model_path):
        logger.error(f"Trained model path not found: {args.model_path}. Exiting.")
        return

    overall_run_summary = {}

    for dataset_name in args.datasets_to_process:
        logger.info(
            f"===== Processing risk assessment for dataset: {dataset_name} ====="
        )

        dataset_summary_report_dir = os.path.join(
            REPORTS_DIR, "risk_assessment", dataset_name
        )
        ensure_dir(dataset_summary_report_dir)

        # Unique identifier for this run's config, combining key parameters.
        # The orchestrator will further refine this for its own report file.
        config_str = f"{dataset_name}_{os.path.basename(args.model_path)}_{args.risk_formula}_{args.norm_method}"
        config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:8]
        run_identifier = f"risk_assessment_run_{config_hash}_{run_timestamp}"

        assessment_summary = run_risk_assessment_for_dataset(
            dataset_name=dataset_name,
            trained_model_path=args.model_path,
            target_stage_col_name_for_classification=args.target_stage_column_name,
            max_possible_stage_val=args.max_cluster_stage_value,
            risk_score_formula=args.risk_formula,
            normalization_method=args.norm_method,
            alert_threshold=args.alert_threshold,
            run_timestamp=run_timestamp,  # Pass timestamp for unique sub-folder naming in orchestrator
        )
        overall_run_summary[f"{dataset_name}_{run_identifier}"] = assessment_summary
        logger.info(f"===== Finished risk assessment for dataset: {dataset_name} =====")

    # Save overall summary for all datasets processed in this script execution
    script_summary_filename = f"risk_assessment_script_summary_{run_timestamp}.json"
    script_summary_path = os.path.join(
        REPORTS_DIR, "risk_assessment", script_summary_filename
    )
    try:
        with open(script_summary_path, "w") as f:
            json.dump(
                overall_run_summary, f, indent=4, default=str
            )  # default=str for path objects etc.
        logger.info(
            f"Overall risk assessment script summary saved to: {script_summary_path}"
        )
    except Exception as e:
        logger.error(f"Failed to save overall script summary: {e}")

    logger.info(
        f"--- Risk Assessment Script Completed --- Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Risk Assessment pipeline for CMAPSS dataset."
    )
    parser.add_argument(
        "--datasets_to_process",
        type=str,
        nargs="+",
        default=DATASET_NAMES,
        choices=DATASET_NAMES,
        help="List of dataset identifiers (e.g., FD001 FD003).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Full path to the trained classifier model file (e.g., .pkl or .joblib).",
    )
    # We need to know which stage column the classifier was trained on for context,
    # even if the model itself doesn't directly use it as a feature after one-hot encoding etc.
    # This helps in interpreting results and potentially in future logic.
    parser.add_argument(
        "--target_stage_column_name",
        type=str,
        default=DEFAULT_TARGET_STAGE_COL_PREFIX,
        help="Name of the stage column from clustering that the classifier was TRAINED on (e.g., kmeans_stage).",
    )
    parser.add_argument(
        "--max_cluster_stage_value",
        type=int,
        default=DEFAULT_MAX_STAGE_VALUE,  # Max value is 4 if stages are 0,1,2,3,4
        help="The maximum numerical value for a stage label (e.g., 4 if stages are 0-4).",
    )
    parser.add_argument(
        "--risk_formula",
        type=str,
        default="urgency_inverted",
        choices=["urgency_inverted", "severity_product"],
        help="Risk score formula to use.",
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default="min_max",
        choices=["min_max"],  # Add more if implemented
        help="Normalization method for risk scores.",
    )
    parser.add_argument(
        "--alert_threshold",
        type=float,
        default=0.7,
        help="Threshold for normalized risk score to issue an alert.",
    )
    # Potentially add: feature_names_path if model doesn't store feature_names_in_

    parsed_args = parser.parse_args()
    main(parsed_args)
