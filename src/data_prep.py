"""Data ingestion and preprocessing utilities for NFL datasets."""

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_raw_data(paths: Iterable[Path]) -> pd.DataFrame:
    """Load raw CSV files from the provided paths and concatenate them.

    Parameters
    ----------
    paths: Iterable[Path]
        Paths to CSV files containing raw NFL player statistics.

    Returns
    -------
    pd.DataFrame
        Combined dataframe of all loaded files.
    """
    dataframes = [pd.read_csv(path) for path in paths]
    if not dataframes:
        return pd.DataFrame()
    return pd.concat(dataframes, ignore_index=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by lowercasing and replacing spaces with underscores."""
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def split_train_test(df: pd.DataFrame, season_column: str, cutoff_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into historical training and recent evaluation sets based on season."""
    train = df[df[season_column] <= cutoff_season].copy()
    test = df[df[season_column] > cutoff_season].copy()
    return train, test
