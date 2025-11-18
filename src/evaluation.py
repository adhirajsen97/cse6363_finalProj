"""Evaluation utilities for concept drift experiments."""

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


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


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> EvaluationResult:
    """Compute RMSE, MAE, and R^2 metrics."""
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return EvaluationResult(rmse=rmse, mae=mae, r2=r2)


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
