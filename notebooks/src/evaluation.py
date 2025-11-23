"""Evaluation utilities for concept drift experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class EvaluationResult:
    """Container for model evaluation metrics."""

    rmse: float
    mae: float
    r2: float


@dataclass
class DriftReport:
    """Summary of model performance across seasons to assess drift."""

    season_scores: Dict[int, EvaluationResult]


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Coefficient of determination."""
    return float(r2_score(y_true, y_pred))


def evaluate_split(model, features: pd.DataFrame, target: pd.Series) -> EvaluationResult:
    """Evaluate a fitted model on the provided dataset."""
    predictions = model.predict(features)
    return EvaluationResult(
        rmse=rmse(target, predictions),
        mae=mae(target, predictions),
        r2=r2(target, predictions),
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> EvaluationResult:
    """Compute RMSE, MAE, and R^2 metrics."""
    return EvaluationResult(rmse=rmse(y_true, y_pred), mae=mae(y_true, y_pred), r2=r2(y_true, y_pred))


def evaluate_by_position(
    model,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    position_column: str = "position",
) -> pd.DataFrame:
    """Evaluate a fitted model for each position subgroup.

    Parameters
    ----------
    model:
        Trained estimator with a ``predict`` method that accepts the provided feature columns.
    df:
        DataFrame that includes both the model-ready feature columns, the target column, and a
        categorical position column (e.g., QB/RB/WR/TE).
    feature_columns:
        Ordered collection of columns to feed into ``model.predict``.
    target_column:
        Column containing the true target values.
    position_column:
        Column indicating the player's position. Defaults to ``"position"``.

    Returns
    -------
    pd.DataFrame
        Table with MAE, RMSE, R², and row counts per position.
    """

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' is missing from the evaluation dataframe.")
    if position_column not in df.columns:
        raise KeyError(f"Position column '{position_column}' is required for subgroup evaluation.")

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"Feature columns missing from dataframe: {missing_features}")

    rows: list[dict[str, object]] = []
    ordered_features = list(feature_columns)

    for position, subset in df.groupby(position_column):
        if subset.empty:
            continue

        X_pos = subset.loc[:, ordered_features]
        y_pos = subset[target_column]
        preds = model.predict(X_pos)

        rows.append(
            {
                "Position": position,
                "Count": len(subset),
                "MAE": mae(y_pos, pd.Series(preds, index=y_pos.index)),
                "RMSE": rmse(y_pos, pd.Series(preds, index=y_pos.index)),
                "R²": r2(y_pos, pd.Series(preds, index=y_pos.index)),
            }
        )

    return pd.DataFrame(rows).sort_values("Position").reset_index(drop=True)


def aggregate_by_season(
    predictions: pd.DataFrame, target_column: str, pred_column: str, season_column: str
) -> DriftReport:
    """Compute evaluation metrics per season to measure concept drift."""
    season_scores: Dict[int, EvaluationResult] = {}
    for season, season_df in predictions.groupby(season_column):
        y_true = season_df[target_column]
        y_pred = season_df[pred_column]
        season_scores[int(season)] = compute_metrics(y_true, y_pred)
    return DriftReport(season_scores=season_scores)
