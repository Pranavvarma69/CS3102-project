# scripts/regression.py
import argparse
import logging
import os
import sys
from datetime import datetime

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.regression.orchestrator import run_regression_for_dataset  # noqa: E402
from src.regression.tuning import DEFAULT_REGRESSION_PARAM_GRIDS  # noqa: E402
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
DEFAULT_TARGET_STAGE_COL_PREFIX = "kmeans_stage"  # Example, from clustering
DEFAULT_MAX_STAGE_VALUE = 4  # Assuming stages 0, 1, 2, 3, 4


def main(args) -> None:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"--- Starting Regression Script --- Timestamp: {run_timestamp}")

    # Ensure base directories exist
    ensure_dir(os.path.join(REPORTS_DIR, "regression"))
    ensure_dir(os.path.join(IMAGE_DIR, "regression"))

    # Define regression model configurations to run
    # RandomForestRegressor has been removed.
    model_configurations = {
        "LinearRegression": [
            {
                "name": "baseline",
                "initial_params": {},
                "perform_tuning": False,
            }
        ],
        "Ridge": [
            {
                "name": "baseline_alpha1",
                "initial_params": {"alpha": 1.0, "random_state": args.random_seed},
                "perform_tuning": False,
            },
            {
                "name": "tuned_grid_search",
                "initial_params": {"random_state": args.random_seed},
                "perform_tuning": True,
                "param_grid": DEFAULT_REGRESSION_PARAM_GRIDS.get("Ridge"),
                "tuning_search_method": "grid",
                "tuning_scoring": "neg_root_mean_squared_error",
            },
        ],
        # SVR can be slow, especially with tuning. Add if needed.
        # "SVR": [
        #     {
        #         "name": "baseline_rbf",
        #         "initial_params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
        #         "perform_tuning": False,
        #     },
        # ],
    }
    if DEFAULT_REGRESSION_PARAM_GRIDS.get(
        "XGBRegressor"
    ):  # Check if XGBoost is available via its grid
        model_configurations["XGBRegressor"] = [
            {
                "name": "baseline_default",
                "initial_params": {
                    "n_estimators": 100,
                    "random_state": args.random_seed,
                },
                "perform_tuning": False,
            },
            {
                "name": "tuned_random_search_xgb",
                "initial_params": {"random_state": args.random_seed},
                "perform_tuning": True,
                "param_grid": DEFAULT_REGRESSION_PARAM_GRIDS.get("XGBRegressor"),
                "tuning_search_method": "random",
                "tuning_n_iter": args.tuning_n_iter,  # Number of iterations for random search
                "tuning_scoring": "neg_root_mean_squared_error",
            },
        ]

    # Filter models if specific ones are requested
    if args.models_to_run:
        filtered_model_configurations = {
            model_name: configs
            for model_name, configs in model_configurations.items()
            if model_name in args.models_to_run
        }
        if not filtered_model_configurations:
            logger.error(
                f"None of the specified models for regression are configured: {args.models_to_run}. Exiting."
            )
            logger.info(
                f"Available configured regressors are: {list(model_configurations.keys())}"
            )
            return
        model_configurations_to_run = filtered_model_configurations
    else:
        model_configurations_to_run = model_configurations

    logger.info(
        f"Regressors that will be processed: {list(model_configurations_to_run.keys())}"
    )

    for dataset_name in args.datasets_to_process:
        logger.info(f"===== Processing dataset for regression: {dataset_name} =====")
        target_stage_col = args.target_stage_column_name
        max_stage_val = args.max_cluster_stage_value
        logger.info(
            f"Using target stage column (from clustering): {target_stage_col} with max stage value: {max_stage_val}"
        )

        _ = run_regression_for_dataset(
            dataset_name=dataset_name,
            target_stage_col=target_stage_col,
            max_cluster_stage_value=max_stage_val,
            model_configurations=model_configurations_to_run,
            n_cv_splits=args.num_cv_splits,
        )
        logger.info(
            f"===== Finished regression processing for dataset: {dataset_name} ====="
        )

    logger.info(
        f"--- Regression Script Completed --- Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Regression pipeline for CMAPSS dataset using custom cluster labels to predict time to next stage."
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
        "--target_stage_column_name",
        type=str,
        default=DEFAULT_TARGET_STAGE_COL_PREFIX,
        help="Name of the stage column from clustering output file (e.g., kmeans_stage).",
    )
    parser.add_argument(
        "--max_cluster_stage_value",
        type=int,
        default=DEFAULT_MAX_STAGE_VALUE,
        help="The maximum value for a stage label (e.g., 4 if stages are 0-4).",
    )
    parser.add_argument(
        "--models_to_run",
        type=str,
        nargs="+",
        default=None,  # Run all configured if None
        help="List of specific regressor names to run (e.g., XGBRegressor Ridge).",
    )
    parser.add_argument(
        "--num_cv_splits",
        type=int,
        default=5,
        help="Number of cross-validation splits for GroupKFold. Default is 5.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility where applicable.",
    )
    parser.add_argument(
        "--tuning_n_iter",
        type=int,
        default=10,  # Default for RandomizedSearchCV iterations
        help="Number of iterations for RandomizedSearchCV hyperparameter tuning.",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
