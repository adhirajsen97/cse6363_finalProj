"""Model training and persistence utilities."""

from pathlib import Path
from typing import Literal, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


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
