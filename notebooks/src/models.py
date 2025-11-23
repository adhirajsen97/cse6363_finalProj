"""Model training and persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
import xgboost as xgb
from xgboost import XGBRegressor

from .evaluation import EvaluationResult, evaluate_split, mae
from .features import build_feature_matrix


@dataclass
class XGBTrainingArtifacts:
    """Container for an XGBoost model plus its preprocessing pipeline."""

    model: XGBRegressor
    booster: xgb.Booster
    preprocessor: object
    feature_names: list[str]


ModelName = Literal["linear", "random_forest", "xgboost"]


def build_model(name: ModelName, random_state: int | None = 42):
    """Create a model by name with lightweight defaults suitable for prototyping."""
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(random_state=random_state, n_estimators=200)
    if name == "xgboost":
        return XGBRegressor(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    raise ValueError(f"Unsupported model name: {name}")


def train_model(model, features: pd.DataFrame, target: pd.Series) -> None:
    """Fit the provided model to features and target."""
    model.fit(features, target)


def evaluate_model(model, features: pd.DataFrame, target: pd.Series) -> float:
    """Return root mean squared error for model predictions."""
    predictions = model.predict(features)
    return float(np.sqrt(mean_squared_error(target, predictions)))


def save_model(model, path: Path) -> None:
    """Persist the trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    """Load a model from disk."""
    return joblib.load(path)


def train_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    alpha_grid: Sequence[float] | np.ndarray,
    *,
    target_column: str = "ppr_points",
    position_column: str = "position",
    drop_columns: Sequence[str] | None = None,
    scale_numeric: bool = True,
) -> dict[str, object]:
    """Train a Ridge baseline with time-aware cross-validation and return metrics."""

    if alpha_grid is None:
        raise ValueError("alpha_grid must contain at least one value.")

    train_sorted = train_df.sort_values(["season", "player_id"]).reset_index(drop=True)
    val_sorted = val_df.sort_values(["season", "player_id"]).reset_index(drop=True)

    X_train, y_train, preprocessor = build_feature_matrix(
        train_sorted,
        target_column=target_column,
        position_column=position_column,
        drop_columns=drop_columns,
        scale_numeric=scale_numeric,
        preprocessor=None,
        fit=True,
    )
    X_val, y_val, _ = build_feature_matrix(
        val_sorted,
        target_column=target_column,
        position_column=position_column,
        drop_columns=drop_columns,
        scale_numeric=scale_numeric,
        preprocessor=preprocessor,
        fit=False,
    )

    alpha_values = np.array(alpha_grid, dtype=float)

    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("alpha_grid must be a one-dimensional collection of floats.")

    if alpha_values.size == 1:
        model = Ridge(alpha=float(alpha_values[0]))
        model.fit(X_train, y_train)
    else:
        unique_seasons = train_sorted["season"].sort_values().unique()
        if unique_seasons.size < 3:
            raise ValueError("Need at least three training seasons for time-aware CV.")
        n_splits = min(5, unique_seasons.size - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = RidgeCV(alphas=alpha_values, cv=tscv, scoring="neg_mean_squared_error")
        model.fit(X_train, y_train)

    train_metrics: EvaluationResult = evaluate_split(model, X_train, y_train)
    val_metrics: EvaluationResult = evaluate_split(model, X_val, y_val)

    best_alpha = float(getattr(model, "alpha_", getattr(model, "alpha", alpha_values[0])))

    return {
        "model": model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "alpha": best_alpha,
    }


def train_xgb_time_cv(
    train_df: pd.DataFrame,
    seasons: Sequence[int],
    param_grid,
    *,
    target_column: str = "ppr_points",
    position_column: str = "position",
    drop_columns: Sequence[str] | None = None,
    scale_numeric: bool = False,
    min_train_seasons: int = 4,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, object]:
    """Perform rolling-origin CV for XGBoost and return the best parameters.

    Parameters
    ----------
    train_df:
        Historical player-season dataframe covering the provided seasons.
    seasons:
        Ordered collection of seasons to include in the rolling-origin splits. The final
        season in the list will serve as the last validation fold.
    param_grid:
        Dict or list of dicts compatible with :class:`sklearn.model_selection.ParameterGrid`.
    """

    if "season" not in train_df.columns:
        raise KeyError("train_df must contain a 'season' column for time-aware splitting.")

    if not seasons:
        raise ValueError("At least two seasons are required for rolling-origin CV.")

    if min_train_seasons < 1:
        raise ValueError("min_train_seasons must be at least 1.")

    unique_df_seasons = sorted(train_df["season"].unique().tolist())
    requested_seasons = sorted(dict.fromkeys(int(s) for s in seasons))

    missing = set(requested_seasons) - set(unique_df_seasons)
    if missing:
        raise ValueError(f"Seasons {sorted(missing)} are not present in the training dataframe.")

    n_required = min_train_seasons + 1
    if len(requested_seasons) < n_required:
        raise ValueError(
            f"Need at least {n_required} seasons to build one fold, received {len(requested_seasons)}."
        )

    df_sorted = train_df.sort_values(
        ["season"] + (["player_id"] if "player_id" in train_df.columns else [])
    ).reset_index(drop=True)

    grid = list(ParameterGrid(param_grid))
    if not grid:
        raise ValueError("param_grid produced no parameter combinations.")

    cv_results: list[dict[str, object]] = []

    for params in grid:
        fold_mae_scores: list[float] = []
        fold_summaries: list[dict[str, object]] = []

        for val_idx in range(min_train_seasons, len(requested_seasons)):
            val_season = requested_seasons[val_idx]
            train_seasons = requested_seasons[:val_idx]

            train_split = df_sorted[df_sorted["season"].isin(train_seasons)]
            val_split = df_sorted[df_sorted["season"] == val_season]

            if train_split.empty or val_split.empty:
                continue

            X_train, y_train, preprocessor = build_feature_matrix(
                train_split,
                target_column=target_column,
                position_column=position_column,
                drop_columns=drop_columns,
                scale_numeric=scale_numeric,
                preprocessor=None,
                fit=True,
            )
            X_val, y_val, _ = build_feature_matrix(
                val_split,
                target_column=target_column,
                position_column=position_column,
                drop_columns=drop_columns,
                scale_numeric=scale_numeric,
                preprocessor=preprocessor,
                fit=False,
            )

            model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                random_state=random_state,
                n_jobs=n_jobs,
                **params,
            )

            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)
            fold_mae = mae(y_val, pd.Series(val_predictions, index=y_val.index))
            fold_mae_scores.append(fold_mae)
            fold_summaries.append(
                {
                    "fold": len(fold_summaries) + 1,
                    "train_start": min(train_seasons),
                    "train_end": max(train_seasons),
                    "val_season": val_season,
                    "mae": fold_mae,
                }
            )

        if not fold_mae_scores:
            continue

        mean_mae_score = float(np.mean(fold_mae_scores))
        cv_results.append(
            {
                "params": params,
                "mean_mae": mean_mae_score,
                "fold_mae": fold_mae_scores,
                "folds": fold_summaries,
            }
        )

    if not cv_results:
        raise ValueError("Rolling-origin CV did not run any folds; check season coverage.")

    best_result = min(cv_results, key=lambda result: result["mean_mae"])

    return {
        "best_params": best_result["params"],
        "best_mean_mae": best_result["mean_mae"],
        "cv_results": cv_results,
        "seasons": requested_seasons,
        "min_train_seasons": min_train_seasons,
    }


def _prepare_retrain_matrix(
    df: pd.DataFrame,
    *,
    target_column: str,
    position_column: str,
    drop_columns: Sequence[str] | None,
    scale_numeric: bool,
) -> tuple[pd.DataFrame, pd.Series, object, pd.DataFrame]:
    """Sort dataframe, build features, and return the aligned training artifacts."""

    if df.empty:
        raise ValueError("Training dataframe is empty; cannot retrain XGBoost.")

    if "season" not in df.columns:
        raise KeyError("Training dataframe must include a 'season' column.")

    sorted_df = df.sort_values(
        ["season"] + (["player_id"] if "player_id" in df.columns else [])
    ).reset_index(drop=True)

    X_train, y_train, preprocessor = build_feature_matrix(
        sorted_df,
        target_column=target_column,
        position_column=position_column,
        drop_columns=drop_columns,
        scale_numeric=scale_numeric,
        preprocessor=None,
        fit=True,
    )

    return X_train, y_train, preprocessor, sorted_df


def retrain_full_xgb(
    df_train_2015_2023: pd.DataFrame,
    best_params: dict[str, object],
    *,
    target_column: str = "ppr_points",
    position_column: str = "position",
    drop_columns: Sequence[str] | None = None,
    scale_numeric: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
) -> XGBTrainingArtifacts:
    """Train a fresh XGBoost model on all seasons up to 2023.

    The function rebuilds the feature matrix (without leakage from 2024),
    then fits a new :class:`xgboost.XGBRegressor` using the best
    hyper-parameters discovered during the rolling-origin search.
    """

    if not best_params:
        raise ValueError("best_params must be a non-empty dictionary.")

    X_train, y_train, preprocessor, _ = _prepare_retrain_matrix(
        df_train_2015_2023,
        target_column=target_column,
        position_column=position_column,
        drop_columns=drop_columns,
        scale_numeric=scale_numeric,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=random_state,
        n_jobs=n_jobs,
        **best_params,
    )
    model.fit(X_train, y_train)

    booster = model.get_booster()
    return XGBTrainingArtifacts(
        model=model,
        booster=booster,
        preprocessor=preprocessor,
        feature_names=X_train.columns.tolist(),
    )


def finetune_xgb_with_sample_weights(
    df_train_2015_2023: pd.DataFrame,
    best_params: dict[str, object],
    *,
    target_column: str = "ppr_points",
    position_column: str = "position",
    drop_columns: Sequence[str] | None = None,
    scale_numeric: bool = False,
    focus_season: int = 2023,
    focus_weight: float = 2.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> XGBTrainingArtifacts:
    """Simulate fine-tuning by overweighting the most recent season (proxy strategy).

    XGBoost's sklearn wrapper does not expose incremental training hooks, so we
    approximate fine-tuning by re-training from scratch with higher sample
    weights on the target season (2023 by default). This biases the booster to
    fit late-era patterns without discarding older context.
    """

    if focus_weight <= 1.0:
        raise ValueError("focus_weight should be > 1.0 to emphasize the focus season.")

    X_train, y_train, preprocessor, sorted_df = _prepare_retrain_matrix(
        df_train_2015_2023,
        target_column=target_column,
        position_column=position_column,
        drop_columns=drop_columns,
        scale_numeric=scale_numeric,
    )

    if "season" not in df_train_2015_2023.columns:
        raise KeyError("Training dataframe must include 'season' to build sample weights.")

    season_series = sorted_df["season"]

    sample_weight = np.ones(len(season_series), dtype=float)
    mask = season_series == focus_season

    if not mask.any():
        raise ValueError(f"No rows found for focus_season={focus_season}; cannot apply weighting.")

    sample_weight[mask] = focus_weight

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=random_state,
        n_jobs=n_jobs,
        **best_params,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    booster = model.get_booster()
    return XGBTrainingArtifacts(
        model=model,
        booster=booster,
        preprocessor=preprocessor,
        feature_names=X_train.columns.tolist(),
    )
