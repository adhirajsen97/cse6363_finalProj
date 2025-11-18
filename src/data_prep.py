"""Data ingestion and preprocessing utilities for NFL datasets."""

from pathlib import Path

import pandas as pd


def load_data(path: Path | str) -> pd.DataFrame:
    """Load the canonical player-season dataset from a CSV file.

    Parameters
    ----------
    path: Path | str
        Location of the canonical CSV containing player-season statistics.

    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from ``path``.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and impute the canonical dataset for modeling.

    The cleaning steps include:
    - Keeping only skill positions (QB, RB, WR, TE).
    - Dropping players with fewer than four games played.
    - Dropping rows missing critical identifiers.
    - Median imputation for numeric columns and mode imputation for categoricals.

    Parameters
    ----------
    df: pd.DataFrame
        Raw dataframe of player-season statistics.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering and modeling.
    """

    cleaned = df.copy()
    allowed_positions = {"QB", "RB", "WR", "TE"}

    cleaned = cleaned[cleaned["position"].isin(allowed_positions)].copy()
    cleaned = cleaned[cleaned["games_played"] >= 4].copy()
    cleaned = cleaned.dropna(subset=["player_id", "season", "position"])

    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if cleaned[col].isna().any():
            if cleaned[col].dropna().empty:
                continue
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    categorical_cols = cleaned.select_dtypes(exclude=["number"]).columns
    for col in categorical_cols:
        if cleaned[col].isna().any():
            mode_series = cleaned[col].mode()
            if not mode_series.empty:
                cleaned[col] = cleaned[col].fillna(mode_series.iloc[0])

    return cleaned


def train_val_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train, validation, and two test sets by season.

    Splits use the following seasons:
    - Train: 2015–2021
    - Validation (old era): 2022
    - Test 2023: 2023
    - Test 2024: 2024

    Assertions ensure chronological ordering and no overlapping player-season
    rows across splits.
    """

    season_values = set(df["season"].unique())
    missing_years = [str(year) for year in (2022, 2023, 2024) if year not in season_values]
    if missing_years:
        raise ValueError(
            "Missing expected seasons for splitting: " + ", ".join(missing_years)
        )

    train_df = df[(df["season"] >= 2015) & (df["season"] <= 2021)].copy()
    val_df = df[df["season"] == 2022].copy()
    test_2023_df = df[df["season"] == 2023].copy()
    test_2024_df = df[df["season"] == 2024].copy()

    if train_df.empty:
        raise ValueError("Training split is empty; ensure seasons 2015-2021 are present after cleaning.")

    assert train_df["season"].max() < val_df["season"].min() < test_2023_df["season"].min() < test_2024_df["season"].min()

    def _player_season_keys(frame: pd.DataFrame) -> set[tuple]:
        return set(zip(frame["player_id"], frame["season"]))

    train_keys = _player_season_keys(train_df)
    val_keys = _player_season_keys(val_df)
    test_2023_keys = _player_season_keys(test_2023_df)
    test_2024_keys = _player_season_keys(test_2024_df)

    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_2023_keys)
    assert train_keys.isdisjoint(test_2024_keys)
    assert val_keys.isdisjoint(test_2023_keys)
    assert val_keys.isdisjoint(test_2024_keys)
    assert test_2023_keys.isdisjoint(test_2024_keys)

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test 2023 rows: {len(test_2023_df)}")
    print(f"Test 2024 rows: {len(test_2024_df)}")

    return train_df, val_df, test_2023_df, test_2024_df
