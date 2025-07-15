import numpy as np
import pandas as pd


def hitrate(row: pd.Series, k: int) -> bool:
    """Calculates hit rate for the given row and top k selections."""
    selected_columns = set(row["selected_columns"][:k])
    relevant_column = row["target_ind"]
    return relevant_column in selected_columns


def recall(row: pd.Series, k: int) -> float:
    """Calculates recall for the given row and top k selections."""
    selected_columns = set(row["selected_columns"][:k])
    relevant_columns = set(row["target_ind"])
    return len(selected_columns.intersection(relevant_columns)) / len(relevant_columns)


def compute_performance_metrics(df: pd.DataFrame, k: list) -> pd.DataFrame:
    """
    Computes performance metrics for the given DataFrame.

    :param df: DataFrame containing 'selected_columns' and 'target_ind'
    :param k: Number of top selections to consider for hitrate and recall
    :return: DataFrame with additional columns for hitrate and recall
    """
    first_row = df.iloc[0]
    if isinstance(first_row["target_ind"], list) or isinstance(
        first_row["target_ind"], np.ndarray
    ):
        metric = "Recall"
    elif (
        isinstance(first_row["target_ind"], int)
        or isinstance(first_row["target_ind"], np.integer)
        or isinstance(first_row["target_ind"], float)
        or isinstance(first_row["target_ind"], np.floating)
    ):
        metric = "HitRate"
    else:
        print(first_row["target_ind"])
        print(type(first_row["target_ind"]))
        raise ValueError("target_ind must be either a list/array or an integer.")

    preds = (
        df.groupby("query_ind")
        .agg(
            {
                "selected_columns": "first",
                "target_ind": "first",
            }
        )
        .reset_index()
    )

    results = {}
    for j in k:
        if metric == "HitRate":
            hitrate_i = preds.apply(lambda row: hitrate(row, j), axis=1).mean()
            results[f"{metric}@{j}"] = hitrate_i
        elif metric == "Recall":
            recall_i = preds.apply(lambda row: recall(row, j), axis=1).mean()
            results[f"{metric}@{j}"] = recall_i

    return results
