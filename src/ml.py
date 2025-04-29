from typing import Optional  # <-- Required for using Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(
    train_file: str, test_file: str, rul_file: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    col_names = ["unit", "time", "opt1", "opt2", "opt3"] + [
        f"s{i}" for i in range(1, 22)
    ]
    df_train = pd.read_csv(train_file, sep=" ", header=None)
    df_train.drop(columns=[26, 27], inplace=True)
    df_train.columns = col_names

    df_test = pd.read_csv(test_file, sep=" ", header=None)
    df_test.drop(columns=[26, 27], inplace=True)
    df_test.columns = col_names

    df_rul = pd.read_csv(rul_file, sep=" ", header=None)
    df_rul = df_rul[0]

    return df_train, df_test, df_rul


def explore(df: pd.DataFrame) -> None:
    print("Shape of data:", df.shape)
    print("Unique engines:", df["unit"].nunique())
    print("Number of NaNs:\n", df.isna().sum())
    print("Sensor stats:\n", df.describe().T)


def plot_sensor_trends(df: pd.DataFrame, sensor: str = "s2", engines: int = 5) -> None:
    sample = df["unit"].unique()[:engines]
    for unit in sample:
        sub = df[df["unit"] == unit]
        plt.plot(sub["time"], sub[sensor], label=f"Engine{unit}")
    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"{sensor} over time")
    plt.legend()
    plt.show()


def remove_constant_sensors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    sensor_cols = [col for col in df.columns if col.startswith("s")]
    std_deviation = df[sensor_cols].std()
    constant_sensors = std_deviation[std_deviation < 0.1].index.tolist()
    df = df.drop(columns=constant_sensors)
    return df, constant_sensors


def plot_correlation(df: pd.DataFrame) -> None:
    sensor_cols = [col for col in df.columns if col.startswith("s")]
    corr = df[sensor_cols].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation")
    plt.xticks(ticks=np.arange(len(sensor_cols)), labels=sensor_cols, rotation=90)
    plt.yticks(ticks=np.arange(len(sensor_cols)), labels=sensor_cols)
    plt.title("Sensor Correlation Matrix")
    plt.tight_layout()
    plt.show()


def explore_test_rul(df_test: pd.DataFrame, df_rul: pd.Series) -> None:
    print("Test Data Shape:", df_test.shape)
    print("Engine Count:", df_test["unit"].nunique())
    print("RUL entries:", df_rul.shape[0])
    print("Sample RUL values:\n", df_rul.head())


def rolling_window_stats(
    df: pd.DataFrame, sensor_cols: list[str], window: int = 5
) -> pd.DataFrame:
    df_original = df.copy()
    for sensor in sensor_cols:
        df_original[f"{sensor}_mean_{window}"] = (
            df.groupby("unit")[sensor]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )
        df_original[f"{sensor}_std_{window}"] = (
            df.groupby("unit")[sensor]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(drop=True)
            .fillna(0)
        )
    return df_original


def pca(df: pd.DataFrame, sensor_cols: list[str], n_components: int = 2) -> np.ndarray:
    sample = df.groupby("unit").last().reset_index()
    x = sample[sensor_cols].values
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_standardised = (x - x_mean) / (x_std + 1e-8)

    covariance_matrix = np.cov(x_standardised.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    x_pca = np.dot(x_standardised, eigenvectors)

    return x_pca


def initialize_centroids(x: np.ndarray, k: int) -> np.ndarray:
    np.random.seed(42)
    indices = np.random.choice(x.shape[0], k, replace=False)
    centroids = x[indices]
    return centroids


def assign_clusters(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.sqrt(((x[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)


def update_centroids(x: np.ndarray, cluster_labels: np.ndarray, k: int) -> np.ndarray:
    new_centroids = []
    for i in range(k):
        cluster_points = x[cluster_labels == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
        else:
            new_centroid = x[np.random.choice(x.shape[0])]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


def kmeans_numpy(
    x: np.ndarray, n_clusters: int = 2, n_iters: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    centroids = initialize_centroids(x, n_clusters)
    for _ in range(n_iters):
        cluster_labels = assign_clusters(x, centroids)
        new_centroids = update_centroids(x, cluster_labels, n_clusters)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, cluster_labels


def plot_clusters(x: np.ndarray, cluster_labels: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(x[:, 0], x[:, 1], c=cluster_labels, cmap="viridis", s=100)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("KMeans Clusters")
    plt.colorbar(label="Cluster")
    plt.show()


def compute_distance_matrix(x: np.ndarray) -> np.ndarray:
    n_samples = x.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance = np.linalg.norm(x[i] - x[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def agglomerative_clustering(x: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    n_samples = x.shape[0]
    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError(f"n_clusters must be between 1 and {n_samples}")

    clusters: dict[int, list[int]] = {i: [i] for i in range(n_samples)}
    distance_matrix = compute_distance_matrix(x)

    while len(clusters) > n_clusters:
        min_dist = float("inf")
        to_merge: tuple[Optional[int], Optional[int]] = (None, None)

        cluster_keys = list(clusters.keys())

        for i in range(len(cluster_keys)):
            key1 = cluster_keys[i]
            for j in range(i + 1, len(cluster_keys)):
                key2 = cluster_keys[j]
                points1 = clusters[key1]
                points2 = clusters[key2]
                sub_matrix = distance_matrix[np.ix_(points1, points2)]

                if sub_matrix.size == 0:
                    continue

                dist = np.min(sub_matrix)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (key1, key2)

        if to_merge[0] is None or to_merge[1] is None:
            break

        c1: int = to_merge[0]
        c2: int = to_merge[1]

        clusters[c1].extend(clusters[c2])
        del clusters[c2]

    cluster_labels = np.zeros(n_samples, dtype=int)
    label_map: dict[int, int] = {key: i for i, key in enumerate(clusters.keys())}
    for original_key, points in clusters.items():
        assigned_label = label_map[original_key]
        cluster_labels[points] = assigned_label

    return cluster_labels


def create_final_degradation_stage(
    df: pd.DataFrame, cluster_labels: list[int]
) -> pd.DataFrame:
    cluster_to_stage = {0: "Normal", 1: "Warning", 2: "Failure"}
    df = df.copy()
    df["degradation_stage"] = [cluster_to_stage[label] for label in cluster_labels]
    return df


def save_dataset_with_labels(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")
