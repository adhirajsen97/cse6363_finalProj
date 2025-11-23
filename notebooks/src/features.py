"""Feature engineering helpers for concept drift analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def encode_categorical(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode categorical columns and return transformed dataframe and encoder."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns), index=df.index)
    numeric_df = df.drop(columns=columns)
    return pd.concat([numeric_df, encoded_df], axis=1), encoder


def add_interactions(df: pd.DataFrame, feature_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Add simple multiplicative interaction terms for the provided feature pairs."""
    df = df.copy()
    for left, right in feature_pairs:
        interaction_name = f"{left}_x_{right}"
        df[interaction_name] = df[left] * df[right]
    return df


class RobustStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that handles zero-variance columns without producing NaNs.
    
    Zero-variance columns are set to 0 (their mean) instead of producing NaN.
    This prevents issues when StandardScaler encounters constant columns.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.zero_variance_mask_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        self.scaler.fit(X)
        
        # Identify zero-variance columns (where std is 0 or very close to 0)
        # StandardScaler sets scale_ to 1.0 for zero-variance columns, but transform produces NaN
        # We'll detect this by checking if transform produces NaN for constant columns
        try:
            if hasattr(self.scaler, 'scale_'):
                # Check for near-zero standard deviations
                self.zero_variance_mask_ = np.abs(self.scaler.scale_) < 1e-10
            else:
                self.zero_variance_mask_ = np.zeros(X.shape[1] if X.ndim > 1 else 1, dtype=bool)
        except AttributeError:
            self.zero_variance_mask_ = np.zeros(X.shape[1] if X.ndim > 1 else 1, dtype=bool)
        
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)
        
        # Handle all non-finite values (NaN, Inf, -Inf) by converting to 0
        # This is safe for standardized features which are centered at 0
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def build_feature_matrix(
    df: pd.DataFrame,
    target_column: str = "ppr_points",
    position_column: str = "position",
    drop_columns: Sequence[str] | None = None,
    scale_numeric: bool = True,
    preprocessor: ColumnTransformer | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Return model-ready feature matrix and target vector with consistent preprocessing.

    The function applies a `ColumnTransformer` that:
    - One-hot encodes the provided `position_column` with `handle_unknown="ignore"`.
    - Optionally scales numeric features via `StandardScaler`. When `scale_numeric` is False,
      numeric features pass through unchanged.

    Parameters
    ----------
    df:
        Player-season dataframe.
    target_column:
        Column to predict (defaults to PPR fantasy points).
    position_column:
        Categorical column to one-hot encode.
    drop_columns:
        Optional set of identifier columns to drop before modeling.
    scale_numeric:
        If True, fit/apply a `StandardScaler` to numeric columns.
    preprocessor:
        An optional, pre-fit `ColumnTransformer` to reuse across splits.
    fit:
        Whether to fit the provided/constructed preprocessor. Pass `False` when reusing an
        already fit transformer.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, ColumnTransformer]
        Feature matrix, target vector, and the fitted ColumnTransformer.
    """

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    features = df.drop(columns=[target_column])

    if drop_columns:
        valid_drop = [col for col in drop_columns if col in features.columns]
        features = features.drop(columns=valid_drop)

    if position_column not in features.columns:
        raise KeyError(f"Position column '{position_column}' not found in dataframe.")
    
    # Ensure no NaN values in input features before transformation
    # (clean_data should handle this, but this is a safety check)
    numeric_input_cols = features.select_dtypes(include=["number", "bool"]).columns
    if len(numeric_input_cols) > 0 and features[numeric_input_cols].isna().any().any():
        # Fill NaN in numeric columns with median before transformation
        for col in numeric_input_cols:
            if features[col].isna().any():
                median_val = features[col].median()
                if pd.notna(median_val):
                    features[col] = features[col].fillna(median_val)
                else:
                    # If all values are NaN, fill with 0
                    features[col] = features[col].fillna(0)

    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    if position_column in numeric_cols:
        numeric_cols.remove(position_column)

    transformers: list[tuple[str, object, Sequence[str]]] = []
    if preprocessor is None:
        if numeric_cols:
            num_transformer: object = RobustStandardScaler() if scale_numeric else "passthrough"
            transformers.append(("numeric", num_transformer, numeric_cols))
        transformers.append(
            (
                "position",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [position_column],
            )
        )
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    elif not fit:
        # When we're reusing an existing preprocessor, we keep the provided `scale_numeric`
        # flag only for compatibility; the behavior is governed by the fitted transformer.
        pass

    if preprocessor is None:
        raise ValueError("A ColumnTransformer is required to build features.")

    if fit:
        X_array = preprocessor.fit_transform(features)
    else:
        X_array = preprocessor.transform(features)

    feature_names = preprocessor.get_feature_names_out()
    X = pd.DataFrame(X_array, columns=feature_names, index=df.index)
    
    # Final safety check: ensure no NaN, Inf, or -Inf values remain
    # Convert to numpy array, handle all non-finite values, then convert back
    X_array_clean = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
    X = pd.DataFrame(X_array_clean, columns=feature_names, index=df.index)
    
    # Double-check: if any NaNs still exist (shouldn't happen), fill with 0
    if X.isna().any().any():
        X = X.fillna(0)
    
    y = df[target_column].copy()
    return X, y, preprocessor
