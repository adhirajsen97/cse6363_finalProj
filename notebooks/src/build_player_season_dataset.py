"""One-time pipeline to build a canonical player-season dataset (2015-2024).

The script is designed to pull weekly NFL player stats from nflfastR's public
player_stats releases, roll them up to season-level rows, and attach a concise
set of fantasy-friendly features. A fallback path lets you plug in an existing
season-level CSV (e.g., the bundled ``fantasy_merged_7_17.csv``) when offline.
"""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
import urllib.request

import pandas as pd

PLAYER_STATS_URL = (
    "https://raw.githubusercontent.com/"
    "nflverse/nflfastR-data/master/data/player_stats/player_stats_{season}.csv.gz"
)

WEEKLY_ALIAS_MAP = {
    "player_id": ["player_id", "gsis_id", "nfl_id"],
    "player_name": ["player_name", "name", "player"],
    "position": ["position", "pos", "position_group"],
    "team": ["recent_team", "team", "posteam", "Tm"],
    "season": ["season", "Year"],
    "week": ["week"],
    "age": ["age"],
    "years_experience": ["years_exp", "experience", "exp"],
    "passing_yards": ["pass_yds", "passing_yards", "pass_yards"],
    "passing_tds": ["pass_td", "pass_tds"],
    "interceptions": ["int", "pass_int"],
    "rushing_yards": ["rush_yds", "rushing_yards", "rush_yards", "RushYds"],
    "rushing_tds": ["rush_td", "rush_tds", "RushTD"],
    "carries": ["rush_att", "rushing_att", "carries", "RushAtt"],
    "receiving_yards": ["rec_yds", "receiving_yards", "receiving_yardage", "RecYds"],
    "receiving_tds": ["rec_td", "rec_tds", "RecTD"],
    "receptions": ["rec", "receptions", "Rec"],
    "targets": ["targets", "tgt", "Tgt"],
    "games_played": ["games", "G"],
    "pass_attempts": ["pass_attempts", "attempts", "pass_att", "Att", "att"],
}

SEASON_COLUMNS = [
    "player_id",
    "player_name",
    "season",
    "position",
    "team",
    "age",
    "years_experience",
    "games_played",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "targets",
    "receptions",
    "carries",
    "team_pass_rate",
    "team_rush_rate",
    "team_off_rank",
    "ppr_points",
    "season_total_yards",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-season",
        type=int,
        default=2015,
        help="First season to include (inclusive).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2024,
        help="Last season to include (inclusive).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/player_season_2015_2024.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/player_stats_cache"),
        help="Directory to cache downloaded weekly player stats.",
    )
    parser.add_argument(
        "--fallback-season-file",
        type=Path,
        default=None,
        help="Optional season-level CSV to use when downloads are unavailable.",
    )
    return parser.parse_args()


def _first_available(df: pd.DataFrame, options: list[str]) -> str | None:
    for opt in options:
        if opt in df.columns:
            return opt
    return None


def _normalize_weekly(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for target, options in WEEKLY_ALIAS_MAP.items():
        source = _first_available(df, options)
        if source:
            normalized[target] = df[source]
    numeric_fields = [
        "passing_yards",
        "passing_tds",
        "rushing_yards",
        "rushing_tds",
        "receiving_yards",
        "receiving_tds",
        "targets",
        "receptions",
        "carries",
        "pass_attempts",
        "interceptions",
    ]
    for numeric in numeric_fields:
        if numeric not in normalized.columns:
            normalized[numeric] = 0
    normalized[numeric_fields] = normalized[numeric_fields].fillna(0)
    normalized = normalized.dropna(subset=["season", "player_name", "team"], how="any")
    normalized = normalized[normalized["season"].between(2015, 2024)]
    if "week" in normalized.columns:
        normalized = normalized[normalized["week"] <= 18]
    return normalized


def _download_weekly(season: int, cache_dir: Path) -> pd.DataFrame | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"player_stats_{season}.csv.gz"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    url = PLAYER_STATS_URL.format(season=season)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not download %s: %s", url, exc)
        return None

    cache_path.write_bytes(content)
    return pd.read_csv(io.BytesIO(content))


def _aggregate_weekly(weekly_df: pd.DataFrame) -> pd.DataFrame:
    agg = weekly_df.groupby(
        ["player_id", "player_name", "position", "team", "season", "age", "years_experience"],
        dropna=False,
    ).agg(
        games_played=("week", "nunique"),
        passing_yards=("passing_yards", "sum"),
        passing_tds=("passing_tds", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rushing_tds", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("receiving_tds", "sum"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        carries=("carries", "sum"),
        interceptions=("interceptions", "sum"),
    )
    agg = agg.reset_index()
    return agg


def _compute_team_context(weekly_df: pd.DataFrame) -> pd.DataFrame:
    team = weekly_df.groupby(["team", "season"], dropna=False).agg(
        pass_attempts=("pass_attempts", "sum"),
        rush_attempts=("carries", "sum"),
        pass_yards=("passing_yards", "sum"),
        rush_yards=("rushing_yards", "sum"),
    )
    team["total_plays"] = team["pass_attempts"] + team["rush_attempts"]
    team["team_pass_rate"] = team["pass_attempts"] / team["total_plays"].replace({0: pd.NA})
    team["team_rush_rate"] = team["rush_attempts"] / team["total_plays"].replace({0: pd.NA})
    team["team_total_yards"] = team["pass_yards"] + team["rush_yards"]
    team["team_off_rank"] = (
        team.groupby("season")["team_total_yards"].rank(ascending=False, method="min")
    )
    team = team.reset_index()[["team", "season", "team_pass_rate", "team_rush_rate", "team_off_rank"]]
    return team


def _attach_ppr_and_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[[
        "passing_yards",
        "passing_tds",
        "rushing_yards",
        "rushing_tds",
        "receiving_yards",
        "receiving_tds",
        "targets",
        "receptions",
        "carries",
        "games_played",
    ]] = df[[
        "passing_yards",
        "passing_tds",
        "rushing_yards",
        "rushing_tds",
        "receiving_yards",
        "receiving_tds",
        "targets",
        "receptions",
        "carries",
        "games_played",
    ]].fillna(0)

    df[[
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "targets",
        "receptions",
        "carries",
    ]] = df[[
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "targets",
        "receptions",
        "carries",
    ]].clip(lower=0)

    df["ppr_points"] = (
        df["receptions"]
        + 0.1 * (df["receiving_yards"] + df["rushing_yards"])
        + 6 * (df["receiving_tds"] + df["rushing_tds"])
        + 0.04 * df["passing_yards"]
        + 4 * df["passing_tds"]
        - 2 * df.get("interceptions", 0)
    )
    df["season_total_yards"] = df["passing_yards"] + df["rushing_yards"] + df["receiving_yards"]
    return df


def _normalize_fallback_season(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {
        "Player": "player_name",
        "PlayerID": "player_id",
        "FantPos": "position",
        "Tm": "team",
        "Year": "season",
        "Age": "age",
        "G": "games_played",
        "Yds": "passing_yards",
        "TD": "passing_tds",
        "Att": "pass_attempts",
        "Int": "interceptions",
        "RushAtt": "carries",
        "RushYds": "rushing_yards",
        "RushTD": "rushing_tds",
        "Tgt": "targets",
        "Rec": "receptions",
        "RecYds": "receiving_yards",
        "RecTD": "receiving_tds",
        "PPR": "ppr_points",
    }
    renamed = renamed.rename(columns=rename_map)
    renamed = renamed[renamed["season"].between(2015, 2024)]
    numeric_fields = [
        "passing_yards",
        "passing_tds",
        "rushing_yards",
        "rushing_tds",
        "receiving_yards",
        "receiving_tds",
        "targets",
        "receptions",
        "carries",
        "pass_attempts",
        "interceptions",
        "games_played",
    ]
    for field in numeric_fields:
        if field not in renamed.columns:
            renamed[field] = 0
    renamed[numeric_fields] = renamed[numeric_fields].fillna(0)
    renamed["years_experience"] = renamed.get("years_experience", pd.NA)
    renamed = _attach_ppr_and_totals(renamed)

    team = renamed.groupby(["team", "season"], dropna=False).agg(
        pass_attempts=("pass_attempts", "sum"),
        rush_attempts=("carries", "sum"),
        team_total_yards=("season_total_yards", "sum"),
    )
    team["total_plays"] = team["pass_attempts"] + team["rush_attempts"]
    team["team_pass_rate"] = team["pass_attempts"] / team["total_plays"].replace({0: pd.NA})
    team["team_rush_rate"] = team["rush_attempts"] / team["total_plays"].replace({0: pd.NA})
    team["team_off_rank"] = (
        team.groupby("season")["team_total_yards"].rank(ascending=False, method="min")
    )
    team = team.reset_index()[["team", "season", "team_pass_rate", "team_rush_rate", "team_off_rank"]]
    merged = renamed.merge(team, on=["team", "season"], how="left")
    return merged[SEASON_COLUMNS]


def build_dataset(
    start_season: int, end_season: int, cache_dir: Path, fallback_path: Path | None
) -> pd.DataFrame:
    weekly_frames: list[pd.DataFrame] = []
    for season in range(start_season, end_season + 1):
        df = _download_weekly(season, cache_dir)
        if df is None:
            continue
        weekly_frames.append(_normalize_weekly(df))

    season_frames: list[pd.DataFrame] = []
    if weekly_frames:
        weekly_all = pd.concat(weekly_frames, ignore_index=True)
        aggregated = _aggregate_weekly(weekly_all)
        team_context = _compute_team_context(weekly_all)
        aggregated = aggregated.merge(team_context, on=["team", "season"], how="left")
        aggregated = _attach_ppr_and_totals(aggregated)
        season_frames.append(aggregated[SEASON_COLUMNS])

    if fallback_path and fallback_path.exists():
        fallback_df = pd.read_csv(fallback_path)
        season_frames.append(_normalize_fallback_season(fallback_df))

    if not season_frames:
        raise RuntimeError("No data sources were available to build the dataset.")

    combined = pd.concat(season_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_id", "season"], keep="first")
    combined = combined.sort_values(["season", "player_name"]).reset_index(drop=True)
    return combined


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.start_season, args.end_season, args.cache_dir, args.fallback_season_file)
    dataset.to_csv(output_path, index=False)
    logging.info("Wrote %s rows to %s", len(dataset), output_path)


if __name__ == "__main__":
    main()
