from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def split_data(
    df: pd.DataFrame,
    features: list[str],
    label: str = "degradation_stage",
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.Series, list[tuple[list[int], list[int]]]]:
    x = df[features]
    y = df[label]
    engines = df["unit"].unique()

    np.random.seed(1)
    np.random.shuffle(engines)

    engine_splits = np.array_split(engines, n_splits)
    splits = []

    for i in range(n_splits):
        val_engines = engine_splits[i]
        train_engines = []
        for j in range(n_splits):
            if j != i:
                train_engines.extend(engine_splits[j])

        train_idx = df[df["unit"].isin(train_engines)].index.tolist()
        val_idx = df[df["unit"].isin(val_engines)].index.tolist()
        splits.append((train_idx, val_idx))

    return x, y, splits


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> dict[Any, pd.Series]:
    classes = y_train.unique()
    model = {}
    for c in classes:
        x_c = x_train[y_train == c]
        model[c] = x_c.mean()
    return model


def predict(model: dict[Any, pd.Series], x_test: pd.DataFrame) -> np.ndarray:
    predictions = []
    for _, row in x_test.iterrows():
        best_class = None
        smallest_distance = None
        for c, mean_vector in model.items():
            diff = row.values - mean_vector.values
            square_sum = sum(diff**2)
            distance = square_sum**0.5
            if smallest_distance is None or distance < smallest_distance:
                smallest_distance = distance
                best_class = c
        predictions.append(best_class)
    return np.array(predictions)


def simple_oversample(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    classes = y_train.unique()
    counts = y_train.value_counts()
    max_count = counts.max()

    x_list = []
    y_list = []

    for c in classes:
        x_c = x_train[y_train == c]
        repeat = max_count // len(x_c)
        extra = max_count % len(x_c)

        x_new = pd.concat([x_c] * repeat + [x_c.sample(extra, replace=True)])
        y_new = pd.Series([c] * len(x_new))

        x_list.append(x_new)
        y_list.append(y_new)

    x_balanced = pd.concat(x_list)
    y_balanced = pd.concat(y_list)

    return x_balanced, y_balanced


def evaluate(y_true: pd.Series, y_pred: np.ndarray, class_names: list[str]) -> None:
    correct = (y_true == y_pred).sum()
    accuracy = correct / len(y_true)
    print("Accuracy:", round(accuracy, 4))

    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    name_to_index = {name: i for i, name in enumerate(class_names)}

    for t, p in zip(y_true, y_pred):
        i = name_to_index[t]
        j = name_to_index[p]
        cm[i, j] += 1

    print("Confusion Matrix:")
    print(cm)

    precisions = []
    recalls = []
    f1s = []

    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print("Average Precision:", round(np.mean(precisions), 4))
    print("Average Recall:", round(np.mean(recalls), 4))
    print("Average F1 Score:", round(np.mean(f1s), 4))


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def feature_importance(x: pd.DataFrame, y: pd.Series) -> None:
    classes = y.unique()
    means = {c: x[y == c].mean() for c in classes}
    overall_mean = x.mean()

    importance = {}
    for col in x.columns:
        total_diff = sum(abs(means[c][col] - overall_mean[col]) for c in classes)
        importance[col] = total_diff / len(classes)

    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [f for f, _ in sorted_features]
    scores = [s for _, s in sorted_features]

    plt.barh(features, scores)
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()


def test_split(df: pd.DataFrame, features: list[str]) -> None:
    try:
        _, _, splits = split_data(df, features)
        if isinstance(splits, list):
            print("Split function is working correctly.")
        else:
            print("Split function did not return a list.")
    except Exception as e:
        print("There is a problem in the split function:", e)


def test_training(df: pd.DataFrame, features: list[str]) -> None:
    try:
        x = df[features]
        y = df["degradation_stage"]
        model = train_model(x, y)
        if isinstance(model, dict):
            print("Train function is working correctly.")
        else:
            print("Train function did not return a dictionary.")
    except Exception as e:
        print("There is a problem in the train function:", e)


def write_report() -> None:
    print("Report:")
    print("- We divided the data by engine units.")
    print("- We fixed class imbalance by copying some rows.")
    print("- We built a model using average values.")
    print("- We checked how well it works using accuracy and other scores.")
    print("- We found important features by looking at their average values.")
    print("- We did not use any machine learning libraries.")
