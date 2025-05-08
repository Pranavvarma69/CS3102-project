import argparse
import logging
import os
import sys
from datetime import datetime

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.classification.orchestrator import run_classification_for_dataset  # noqa: E402
from src.classification.tuning import DEFAULT_PARAM_GRIDS  # noqa: E402
from src.utils import (  # noqa: E402
    IMAGE_DIR,
    REPORTS_DIR,
    ensure_dir,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATASET_NAMES = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_TARGET_STAGE_COL_PREFIX = "kmeans_stage"  # Example, can be changed via args


def main(args) -> None:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"--- Starting Classification Script --- Timestamp: {run_timestamp}")

    # Ensure base directories exist
    ensure_dir(os.path.join(REPORTS_DIR, "classification"))
    ensure_dir(os.path.join(IMAGE_DIR, "classification"))

    # Define model configurations to run
    # RandomForestClassifier and MLPClassifier have been removed.
    model_configurations = {
        "LogisticRegression": [
            {
                "name": "baseline_balanced",
                "initial_params": {"C": 1.0, "solver": "liblinear"},
                "use_smote": False,
                "perform_tuning": False,
                "class_weight": "balanced",
            },
            {
                "name": "tuned_smote_random",
                "initial_params": {},
                "use_smote": True,
                "perform_tuning": True,
                "class_weight": None,
                "param_grid": DEFAULT_PARAM_GRIDS.get("LogisticRegression"),
                "tuning_search_method": "random",  # Using random search
                "tuning_n_iter": 10,  # Number of iterations for random search
                "tuning_scoring": "f1_macro",  # Added default scoring
            },
        ],
        "KNeighborsClassifier": [
            {
                "name": "baseline_k5",
                "initial_params": {"n_neighbors": 5},
                "use_smote": False,
                "perform_tuning": False,
            },
            {
                "name": "tuned_smote_random",
                "initial_params": {},
                "use_smote": True,
                "perform_tuning": True,
                "param_grid": DEFAULT_PARAM_GRIDS.get("KNeighborsClassifier"),
                "tuning_search_method": "random",
                "tuning_n_iter": 15,
                "tuning_scoring": "f1_macro",  # Added default scoring
            },
        ],
        "GaussianNB": [  # Generally fast, little tuning needed
            {
                "name": "baseline",
                "initial_params": {},
                "use_smote": False,
                "perform_tuning": False,
            },
            {
                "name": "smote",
                "initial_params": {},  # Check effect of SMOTE
                "use_smote": True,
                "perform_tuning": False,
            },
            {
                "name": "tuned_varsmoothing_smote",
                "initial_params": {},
                "use_smote": True,
                "perform_tuning": True,
                "param_grid": DEFAULT_PARAM_GRIDS.get(
                    "GaussianNB", {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
                ),  # Use default from tuning.py or fallback
                "tuning_search_method": "grid",
                "tuning_scoring": "f1_macro",
            },
        ],
        # XGBoost and LightGBM can be added here if you have them installed and want GPU acceleration
    }

    # Filter models if specific ones are requested by the user
    if args.models_to_run:
        filtered_model_configurations = {
            model_name: configs
            for model_name, configs in model_configurations.items()
            if model_name in args.models_to_run
        }
        if not filtered_model_configurations:
            logger.error(
                f"None of the specified models are configured: {args.models_to_run}. Exiting."
            )
            logger.info(
                f"Available configured models are: {list(model_configurations.keys())}"
            )
            return
        model_configurations_to_run = filtered_model_configurations
    else:  # If no specific models are passed, run all defined above
        model_configurations_to_run = model_configurations

    logger.info(
        f"Models that will be processed: {list(model_configurations_to_run.keys())}"
    )

    for dataset_name in args.datasets_to_process:
        logger.info(f"===== Processing dataset: {dataset_name} =====")

        target_col = args.target_stage_column_name
        logger.info(f"Using target stage column: {target_col}")

        _ = run_classification_for_dataset(
            dataset_name=dataset_name,
            target_stage_col=target_col,
            model_configurations=model_configurations_to_run,
            n_cv_splits=args.num_cv_splits,
        )
        logger.info(f"===== Finished processing dataset: {dataset_name} =====")

    logger.info(
        f"--- Classification Script Completed --- Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Classification pipeline for CMAPSS dataset using custom cluster labels."
    )
    parser.add_argument(
        "--datasets_to_process",
        type=str,
        nargs="+",
        default=DATASET_NAMES,  # Default to ALL datasets for the final report
        choices=DATASET_NAMES,
        help=(
            "List of dataset identifiers (e.g., FD001 FD003). "
            f"Defaults to all: {DATASET_NAMES}."
        ),
    )
    parser.add_argument(
        "--target_stage_column_name",
        type=str,
        default=DEFAULT_TARGET_STAGE_COL_PREFIX,
        help=(
            f"Name of the stage column from clustering output file to use as target (e.g., kmeans_stage, agglomerative_stage). "
            f"This column must exist in the '*_clustered_all_algos.parquet' file. Default: {DEFAULT_TARGET_STAGE_COL_PREFIX}"
        ),
    )
    parser.add_argument(
        "--models_to_run",
        type=str,
        nargs="+",
        default=None,  # If None, runs all models defined in model_configurations_to_run
        help="List of specific model names to run (e.g., LogisticRegression GaussianNB). Runs all configured models by default.",
    )
    parser.add_argument(
        "--num_cv_splits",
        type=int,
        default=5,  # 5-fold CV is a common standard
        help="Number of cross-validation splits for GroupKFold. Default is 5.",
    )

    parsed_args = parser.parse_args()

    # Ensure DEFAULT_PARAM_GRIDS in src/classification/tuning.py has an entry for GaussianNB if you want to tune it this way
    # For example:
    # "GaussianNB": {
    #     'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    # },
    # The script provides a fallback if not found in DEFAULT_PARAM_GRIDS for GaussianNB.

    main(parsed_args)
