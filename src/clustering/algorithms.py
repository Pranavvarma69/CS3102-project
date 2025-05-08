import logging

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

logger = logging.getLogger(__name__)


def apply_kmeans_clustering(
    data: pd.DataFrame, n_clusters: int, random_state: int = 42
) -> pd.Series:
    """
    Applies K-Means clustering to the provided data.

    Args:
        data: DataFrame with features for clustering.
        n_clusters: The number of clusters to form.
        random_state: Random state for reproducibility.

    Returns:
        A pandas Series containing the cluster labels for each sample.
    """
    logger.info(f"Applying K-Means clustering with {n_clusters} clusters.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    try:
        labels = kmeans.fit_predict(data)
        logger.info("K-Means clustering completed.")
        return pd.Series(labels, index=data.index, name="kmeans_raw_labels")
    except Exception as e:
        logger.error(f"Error during K-Means clustering: {e}")
        raise


def apply_agglomerative_clustering(data: pd.DataFrame, n_clusters: int) -> pd.Series:
    """
    Applies Agglomerative Clustering to the provided data.

    Args:
        data: DataFrame with features for clustering.
        n_clusters: The number of clusters to form.

    Returns:
        A pandas Series containing the cluster labels for each sample.
    """
    logger.info(f"Applying Agglomerative clustering with {n_clusters} clusters.")
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    try:
        labels = agg_clustering.fit_predict(data)
        logger.info("Agglomerative clustering completed.")
        return pd.Series(labels, index=data.index, name="agglomerative_raw_labels")
    except Exception as e:
        logger.error(f"Error during Agglomerative clustering: {e}")
        raise
