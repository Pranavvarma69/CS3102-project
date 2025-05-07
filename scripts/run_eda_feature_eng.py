import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_loading import load_dataset  # noqa: E402
from src.eda.statistics import (  # noqa: E402
    get_dataset_info,
    get_sensor_statistics,
)
from src.eda.visualization import (  # noqa: E402
    plot_correlation_heatmap,
    plot_sensor_degradation_patterns,
    plot_sensor_distributions,
    plot_sensor_trends,
)

# Corrected block:
from src.feature_engineering.dimensionality_reduction import (  # noqa: E402
    generate_engineered_features,
)
from src.feature_engineering.visualization import plot_pca_components  # noqa: E402
from src.preprocessing import (  # noqa: E402
    check_missing_values,
    drop_features,
    identify_constant_features,
)
from src.utils import (  # noqa: E402
    DATA_DIR,
    IMAGE_DIR,
    OPERATIONAL_SETTINGS,
    PROCESSED_DATA_DIR,
    SENSOR_MEASUREMENTS,
    ensure_dir,
    save_plot,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATASET_NAMES = ["FD001", "FD002", "FD003", "FD004"]


def _run_eda_visualizations(
    df_cleaned: pd.DataFrame,
    dataset_name: str,
    current_sensor_cols: list[str],
    eda_image_dir_short: str,
    num_sample_engines: int,
) -> None:
    """Helper function to generate and save EDA visualizations."""
    logging.info(f"--- Generating EDA Visualizations for {dataset_name} ---")

    sample_engine_ids = np.random.choice(
        df_cleaned["engine_id"].unique(),
        min(num_sample_engines, df_cleaned["engine_id"].nunique()),
        replace=False,
    )
    for engine_id in sample_engine_ids:
        logging.info(f"Plotting sensor trends for engine {engine_id} of {dataset_name}")
        try:
            fig_trends = plot_sensor_trends(
                df_cleaned, engine_id=engine_id, sensor_cols=current_sensor_cols
            )
            save_plot(
                fig_trends, eda_image_dir_short, f"sensor_trends_engine_{engine_id}.png"
            )
        except Exception as e:
            logging.error(
                f"Failed to plot sensor trends for engine {engine_id} of {dataset_name}: {e}"
            )

    logging.info(f"Plotting sensor correlation heatmap for {dataset_name}")
    try:
        cols_for_corr = current_sensor_cols + OPERATIONAL_SETTINGS + ["cycle"]
        cols_for_corr = [c for c in cols_for_corr if c in df_cleaned.columns]
        fig_corr = plot_correlation_heatmap(df_cleaned, columns=cols_for_corr)
        save_plot(fig_corr, eda_image_dir_short, "sensor_correlation_heatmap.png")
    except Exception as e:
        logging.error(f"Failed to plot correlation heatmap for {dataset_name}: {e}")

    logging.info(f"Plotting sensor distributions for {dataset_name}")
    try:
        fig_dist = plot_sensor_distributions(
            df_cleaned, sensor_cols=current_sensor_cols
        )
        save_plot(fig_dist, eda_image_dir_short, "sensor_distributions.png")
    except Exception as e:
        logging.error(f"Failed to plot sensor distributions for {dataset_name}: {e}")

    logging.info(f"Plotting sensor degradation patterns for {dataset_name}")
    try:
        fig_degrad = plot_sensor_degradation_patterns(
            df_cleaned, sensor_cols=current_sensor_cols, n_samples=10, normalize=True
        )
        save_plot(fig_degrad, eda_image_dir_short, "sensor_degradation_patterns.png")
    except Exception as e:
        logging.error(f"Failed to plot degradation patterns for {dataset_name}: {e}")


def _run_feature_engineering_and_visualization(
    df_cleaned: pd.DataFrame,
    dataset_name: str,
    current_sensor_cols: list[str],
    feature_eng_image_dir_short: str,
    feature_output_dir: str,
    processed_filename: str,
) -> None:
    """Helper function for feature engineering, PCA visualization, and saving."""
    logging.info(f"--- Performing Feature Engineering for {dataset_name} ---")
    df_features = generate_engineered_features(
        df=df_cleaned.copy(),
        sensor_cols=current_sensor_cols,
        window_size=5,
        lag=3,
        apply_dimensionality_reduction=True,
        n_components=2,
    )

    logging.info(f"Generated features for {dataset_name}. Shape: {df_features.shape}")
    if not df_features.empty:
        logging.info(
            f"Feature names for {dataset_name}: " + ", ".join(df_features.columns)
        )
        logging.info(
            f"Engineered Features DataFrame (Top 5 rows) for {dataset_name}:\n{df_features.head().to_string()}"
        )

        if "pca_1" in df_features.columns and "pca_2" in df_features.columns:
            logging.info(f"Plotting PCA components for {dataset_name}")
            try:
                fig_pca_rul = plot_pca_components(
                    df_features, hue_col="RUL", sample_frac=0.2
                )
                save_plot(
                    fig_pca_rul,
                    feature_eng_image_dir_short,
                    "pca_components_scatter_by_RUL.png",
                )

                fig_pca_engine = plot_pca_components(
                    df_features, hue_col="engine_id", sample_frac=0.2
                )
                save_plot(
                    fig_pca_engine,
                    feature_eng_image_dir_short,
                    "pca_components_scatter_by_engine.png",
                )
            except Exception as e:
                logging.error(f"Failed to plot PCA components for {dataset_name}: {e}")
        else:
            logging.warning(
                f"PCA components not found in features for {dataset_name}. Skipping PCA plot."
            )

        output_path = os.path.join(feature_output_dir, processed_filename)
        logging.info(f"Saving engineered features for {dataset_name} to {output_path}")
        try:
            df_features.to_parquet(output_path, index=False)
            logging.info(f"Successfully saved engineered features for {dataset_name}.")
        except Exception as e:
            logging.error(f"Failed to save engineered features for {dataset_name}: {e}")
    else:
        logging.warning(
            f"Engineered features DataFrame for {dataset_name} is empty. Skipping PCA plots and saving."
        )


def _explore_test_data(dataset_name: str) -> None:
    """Helper function to explore test data and RUL files."""
    logging.info(
        f"--- Exploring Test Data and RUL File Structure (Sample) for {dataset_name} ---"
    )
    test_filename_short = f"test_{dataset_name}.txt"
    rul_filename_short = f"RUL_{dataset_name}.txt"

    try:
        df_test_sample = load_dataset(test_filename_short, data_dir=DATA_DIR)
        logging.info(
            f"Successfully loaded test data: {test_filename_short}. Shape: {df_test_sample.shape}"
        )
        logging.info(
            f"Test data ({test_filename_short}) head:\n{df_test_sample.head().to_string()}"
        )
        check_missing_values(df_test_sample)
        _ = identify_constant_features(df_test_sample)
    except FileNotFoundError:
        logging.warning(
            f"Test file {test_filename_short} not found for {dataset_name}. Skipping test data exploration."
        )
    except Exception as e:
        logging.error(
            f"Error loading or processing test file {test_filename_short} for {dataset_name}: {e}"
        )

    try:
        rul_file_path = os.path.join(DATA_DIR, rul_filename_short)
        if os.path.exists(rul_file_path):
            df_rul_sample = pd.read_csv(
                rul_file_path,
                sep=r"\s+",
                header=None,
                names=["RUL_actual_test"],
                engine="python",
            )
            logging.info(
                f"Successfully loaded RUL data: {rul_filename_short}. Shape: {df_rul_sample.shape}"
            )
            logging.info(
                f"RUL data ({rul_filename_short}) head:\n{df_rul_sample.head().to_string()}"
            )
            logging.info(
                f"RUL data basic stats:\n{df_rul_sample['RUL_actual_test'].describe().to_string()}"
            )
        else:
            logging.warning(
                f"RUL file {rul_filename_short} not found for {dataset_name}. Skipping RUL data exploration."
            )
    except Exception as e:
        logging.error(
            f"Error loading or processing RUL file {rul_filename_short} for {dataset_name}: {e}"
        )


def process_single_dataset(dataset_name: str, num_sample_engines: int) -> None:
    """Processes a single dataset for EDA and feature engineering."""
    logging.info(f"--- Processing dataset: {dataset_name} ---")

    train_filename = f"train_{dataset_name}.txt"
    processed_filename = f"{dataset_name}_features.parquet"
    eda_image_dir_short = f"eda/{dataset_name}"
    feature_eng_image_dir_short = f"feature_engineering/{dataset_name}"

    ensure_dir(os.path.join(IMAGE_DIR, eda_image_dir_short))
    ensure_dir(os.path.join(IMAGE_DIR, feature_eng_image_dir_short))

    feature_output_dir = os.path.join(PROCESSED_DATA_DIR, "features")
    ensure_dir(feature_output_dir)

    logging.info(f"Loading training data: {train_filename}")
    try:
        df_train = load_dataset(train_filename, data_dir=DATA_DIR)
    except Exception as e:
        logging.error(f"Failed to load data for {dataset_name}: {e}")
        return

    logging.info("--- Performing Initial EDA and Preprocessing ---")
    check_missing_values(df_train)
    constant_cols = identify_constant_features(df_train)
    features_to_drop = constant_cols
    df_train_cleaned = drop_features(df_train, features_to_drop)

    current_sensor_cols = [
        col for col in SENSOR_MEASUREMENTS if col in df_train_cleaned.columns
    ]
    logging.info(f"Using sensor columns for EDA/Feature Eng: {current_sensor_cols}")

    info = get_dataset_info(df_train_cleaned)
    logging.info(
        f"Dataset {dataset_name} Info: {info['num_engines']} engines, {info['total_cycles']} total cycles, "
        f"Avg Lifetime: {info['avg_lifetime']:.2f} cycles"
    )
    sensor_stats = get_sensor_statistics(df_train_cleaned, current_sensor_cols)
    logging.info(
        f"Sensor Statistics (Top 5 rows) for {dataset_name}:\n{sensor_stats.head().to_string()}"
    )

    _run_eda_visualizations(
        df_train_cleaned,
        dataset_name,
        current_sensor_cols,
        eda_image_dir_short,
        num_sample_engines,
    )
    _run_feature_engineering_and_visualization(
        df_train_cleaned,
        dataset_name,
        current_sensor_cols,
        feature_eng_image_dir_short,
        feature_output_dir,
        processed_filename,
    )
    _explore_test_data(dataset_name)


def main(
    dataset_names: list[str] = DATASET_NAMES, num_sample_engines: int = 3
) -> None:  # Added list[str] type hint
    """
    Main function to run EDA and Feature Engineering pipeline for multiple CMAPSS datasets.

    Args:
        dataset_names (list[str]): List of dataset identifiers (e.g., ["FD001", "FD002"]).
        num_sample_engines (int): Number of sample engines to use for trend visualizations per dataset.
    """
    logging.info(
        "--- Starting EDA and Feature Engineering for all specified datasets ---"
    )

    for dataset_name in dataset_names:
        process_single_dataset(dataset_name, num_sample_engines)

    logging.info(
        "--- EDA and Feature Engineering for all specified datasets complete ---"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EDA and Feature Engineering pipeline for CMAPSS dataset."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DATASET_NAMES,
        choices=DATASET_NAMES,  # type: ignore
        help="List of dataset identifiers (e.g., FD001 FD003). Defaults to all.",
    )
    parser.add_argument(
        "--num_engines",
        type=int,
        default=3,
        help="Number of sample engine IDs for trend visualization plots per dataset.",
    )
    args = parser.parse_args()

    main(dataset_names=args.datasets, num_sample_engines=args.num_engines)
