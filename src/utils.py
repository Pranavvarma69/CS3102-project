import logging
import os

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "CMaps")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

COLUMNS = [
    "engine_id",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s19",
    "s20",
    "s21",
]

OPERATIONAL_SETTINGS = ["setting1", "setting2", "setting3"]
SENSOR_MEASUREMENTS = [f"s{i}" for i in range(1, 22)]


def ensure_dir(directory_path: str) -> None:
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)
    logging.debug(f"Ensured directory exists: {directory_path}")


def save_plot(figure: plt.Figure, state_dir: str, filename: str) -> None:
    """
    Saves a matplotlib figure to the specified subdirectory within the images folder.

    Args:
        figure: The matplotlib figure object to save.
        state_dir: The subdirectory name (e.g., 'data_exploration', 'feature_engineering').
        filename: The name for the saved file (e.g., 'sensor_trends.png').
    """
    output_subdir = os.path.join(IMAGE_DIR, state_dir)
    ensure_dir(output_subdir)
    full_path = os.path.join(output_subdir, filename)
    try:
        figure.savefig(full_path, bbox_inches="tight", dpi=300)
        logging.info(f"Plot saved successfully to {full_path}")
    except Exception as e:
        logging.error(f"Failed to save plot to {full_path}: {e}")
    plt.close(figure)
