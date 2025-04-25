import logging
import os

import pandas as pd

from src.utils import COLUMNS, DATA_DIR


def load_dataset(filename: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Loads a single CMAPSS dataset file.

    Args:
        filename: The name of the file (e.g., 'train_FD001.txt').
        data_dir: The directory containing the data file.

    Returns:
        A pandas DataFrame with appropriate column names.
    """
    file_path = os.path.join(data_dir, filename)
    logging.info(f"Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path, sep="\s+", header=None, index_col=False)

        df = df.dropna(axis=1, how="all")

        if len(df.columns) == len(COLUMNS):
            df.columns = COLUMNS
            logging.info(f"Successfully loaded and assigned columns for {filename}.")
        elif len(df.columns) == len(COLUMNS) - 1:
            # Sometimes the last sensor column might be missing if all values were NaN etc.
            df.columns = COLUMNS[:-1]
            logging.warning(
                f"Loaded {filename} with {len(df.columns)} columns, expected {len(COLUMNS)}. Assigned names accordingly."
            )
        else:
            logging.error(
                f"Mismatch in column count for {filename}. Expected {len(COLUMNS)}, got {len(df.columns)}. Cannot assign names."
            )
            raise ValueError(f"Incorrect number of columns found in {filename}")

        logging.info(f"Dataset shape: {df.shape}")
        return df

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise
