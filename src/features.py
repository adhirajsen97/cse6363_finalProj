"""Feature engineering helpers for concept drift analysis."""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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
