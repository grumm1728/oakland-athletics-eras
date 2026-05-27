"""Build Oakland Athletics roster-era data and static exports.

The pipeline starts from Lahman CSV files, filters Oakland seasons, computes
season-normalized player importance, and writes both analysis CSVs and compact
JSON used by the web app.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "lahman"
BREF_RAW_DIR = ROOT / "data" / "raw" / "baseball_reference"
PROCESSED_DIR = ROOT / "data" / "processed"
PUBLIC_DATA_DIR = ROOT / "public" / "data"
EXPORT_DIR = ROOT / "public" / "exports"

TEAM_ID = "OAK"
START_YEAR = 1968
END_YEAR = 2024

REQUIRED_TABLES = [
    "Appearances.csv",
    "Batting.csv",
    "Pitching.csv",
    "Fielding.csv",
    "People.csv",
    "Teams.csv",
]

SABR_LAHMAN_PAGE = "https://sabr.org/lahman-database/"
BREF_WAR_BATTING_URL = "https://www.baseball-reference.com/data/war_daily_bat.txt"
BREF_WAR_PITCHING_URL = "https://www.baseball-reference.com/data/war_daily_pitch.txt"
BREF_WAR_FILES = {
    "war_daily_bat.txt": BREF_WAR_BATTING_URL,
    "war_daily_pitch.txt": BREF_WAR_PITCHING_URL,
}
BOX_SHARED_NAME = "y1prhc795jk8zvmelfd3jq7tl389y6cd"
BOX_SHARED_URL = f"https://sabr.app.box.com/s/{BOX_SHARED_NAME}"
BOX_DOWNLOAD_URL = (
    "https://sabr.app.box.com/index.php"
    "?rm=box_download_shared_file"
    f"&shared_name={BOX_SHARED_NAME}"
    "&file_id=f_{file_id}"
)


def ensure_dirs() -> None:
    for directory in [RAW_DIR, BREF_RAW_DIR, PROCESSED_DIR, PUBLIC_DATA_DIR, EXPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def discover_box_file_ids() -> dict[str, str]:
    """Return a CSV filename -> Box file id mapping from the SABR folder page."""
    urls = [
        BOX_SHARED_URL,
        f"{BOX_SHARED_URL}?page=2",
        f"{BOX_SHARED_URL}?sortColumn=name&sortDirection=asc",
    ]
    found: dict[str, str] = {}
    pattern = re.compile(
        r'"typedID":"f_(?P<id>\d+)".{0,3000}?"name":"(?P<name>[^"]+\.csv)"',
        flags=re.DOTALL,
    )

    for url in urls:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        for match in pattern.finditer(response.text):
            found[match.group("name")] = match.group("id")

    missing = [name for name in REQUIRED_TABLES if name not in found]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Could not discover Box ids for: {missing_list}")
    return found


def download_lahman_tables(force: bool = False) -> None:
    """Download required Lahman CSVs if they are not already cached."""
    ensure_dirs()
    needed = [name for name in REQUIRED_TABLES if force or not (RAW_DIR / name).exists()]
    if not needed:
        return

    print("Discovering official SABR Lahman CSV file ids...")
    file_ids = discover_box_file_ids()
    for name in needed:
        file_id = file_ids[name]
        url = BOX_DOWNLOAD_URL.format(file_id=file_id)
        target = RAW_DIR / name
        tmp = target.with_suffix(".tmp")
        print(f"Downloading {name}...")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(target)


def download_bref_war_files(force: bool = False) -> None:
    """Download Baseball-Reference daily WAR files if they are not cached."""
    ensure_dirs()
    for filename, url in BREF_WAR_FILES.items():
        target = BREF_RAW_DIR / filename
        if target.exists() and not force:
            continue
        tmp = target.with_suffix(".tmp")
        print(f"Downloading Baseball-Reference {filename}...")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(target)


def read_table(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run with --download or place Lahman CSVs there.")
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def oak_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (pd.to_numeric(df["yearID"], errors="coerce").between(START_YEAR, END_YEAR))
        & (df["teamID"] == TEAM_ID)
    ].copy()


def season_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[pd.to_numeric(df["yearID"], errors="coerce").between(START_YEAR, END_YEAR)].copy()


def group_sum(
    df: pd.DataFrame,
    columns: list[str],
    prefixes: dict[str, str] | None = None,
    key_columns: list[str] | None = None,
) -> pd.DataFrame:
    prefixes = prefixes or {}
    key_columns = key_columns or ["playerID", "yearID"]
    present = [col for col in columns if col in df.columns]
    if not present:
        return df[key_columns].drop_duplicates()
    grouped = df.groupby(key_columns, as_index=False)[present].sum(numeric_only=True)
    return grouped.rename(columns={col: prefixes.get(col, col) for col in present})


def normalize_within_group(df: pd.DataFrame, group_columns: list[str], column: str, out_column: str) -> None:
    maxes = df.groupby(group_columns)[column].transform("max").replace(0, np.nan)
    df[out_column] = (df[column] / maxes).fillna(0).clip(lower=0, upper=1)


def normalize_within_year(df: pd.DataFrame, column: str, out_column: str) -> None:
    normalize_within_group(df, ["yearID"], column, out_column)


def people_name_table(people: pd.DataFrame) -> pd.DataFrame:
    if {"nameFirst", "nameLast"}.issubset(people.columns):
        people_names = people[["playerID", "nameFirst", "nameLast"]].copy()
        people_names["nameFirst"] = people_names["nameFirst"].fillna("")
        people_names["nameLast"] = people_names["nameLast"].fillna("")
        people_names["player_name"] = (people_names["nameFirst"] + " " + people_names["nameLast"]).str.strip()
    else:
        people_names = people[["playerID"]].copy()
        people_names["player_name"] = people_names["playerID"]
    return people_names[["playerID", "player_name"]]


def read_bref_war_file(filename: str) -> pd.DataFrame:
    path = BREF_RAW_DIR / filename
    if not path.exists():
        download_bref_war_files(force=False)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Could not download Baseball-Reference WAR data.")
    usecols = {"player_ID", "year_ID", "team_ID", "WAR"}
    return pd.read_csv(path, usecols=lambda column: column in usecols, low_memory=False)


def bref_war_source(
    filename: str,
    value_column: str,
    people_map: pd.DataFrame,
    team_map: pd.DataFrame,
) -> pd.DataFrame:
    source = read_bref_war_file(filename)
    source = source.rename(
        columns={
            "player_ID": "bbrefID",
            "year_ID": "yearID",
            "team_ID": "teamIDBR",
            "WAR": value_column,
        }
    )
    source["yearID"] = pd.to_numeric(source["yearID"], errors="coerce").astype("Int64")
    source = source[source["yearID"].between(START_YEAR, END_YEAR).fillna(False)].copy()
    source = source[source["teamIDBR"].fillna("") != "TOT"].copy()
    source[value_column] = pd.to_numeric(source[value_column], errors="coerce").fillna(0)

    source = source.merge(people_map, how="inner", on="bbrefID")
    source = source.merge(team_map, how="left", on=["yearID", "teamIDBR"])

    unmapped = source[source["teamID"].isna() & source["teamIDBR"].notna()]
    if not unmapped.empty:
        examples = ", ".join(sorted(unmapped["teamIDBR"].astype(str).unique())[:8])
        print(f"Warning: dropped {len(unmapped)} WAR rows with unmapped BRef team ids: {examples}")
    source = source[source["teamID"].notna()].copy()
    source["yearID"] = source["yearID"].astype(int)

    return (
        source.groupby(["playerID", "yearID", "teamID"], as_index=False)[value_column]
        .sum(numeric_only=True)
    )


def build_bref_war_table(people: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    download_bref_war_files(force=False)
    if "bbrefID" not in people.columns:
        raise RuntimeError("Lahman People.csv does not include bbrefID, which is required for BRef WAR.")

    people_map = (
        people[["playerID", "bbrefID"]]
        .dropna()
        .drop_duplicates()
    )
    team_map = (
        teams[["yearID", "teamID", "teamIDBR"]]
        .dropna(subset=["yearID", "teamID", "teamIDBR"])
        .drop_duplicates()
    )
    team_map["yearID"] = pd.to_numeric(team_map["yearID"], errors="coerce").astype("Int64")
    team_map = team_map[team_map["yearID"].between(START_YEAR, END_YEAR).fillna(False)].copy()

    batting = bref_war_source("war_daily_bat.txt", "batting_bwar", people_map, team_map)
    pitching = bref_war_source("war_daily_pitch.txt", "pitching_bwar", people_map, team_map)
    war = batting.merge(pitching, how="outer", on=["playerID", "yearID", "teamID"])
    for column in ["batting_bwar", "pitching_bwar"]:
        war[column] = pd.to_numeric(war[column], errors="coerce").fillna(0)
    war["bwar"] = war["batting_bwar"] + war["pitching_bwar"]
    war["bwar_positive"] = war["bwar"].clip(lower=0)
    return war[["playerID", "yearID", "teamID", "bwar", "batting_bwar", "pitching_bwar", "bwar_positive"]]


def add_war_metrics(
    player_seasons: pd.DataFrame,
    war: pd.DataFrame,
    key_columns: list[str],
    normalization_columns: list[str],
    fallback_score_column: str,
    prefix: str = "",
) -> pd.DataFrame:
    war_columns = ["bwar", "batting_bwar", "pitching_bwar", "bwar_positive"]
    merged = player_seasons.merge(war[key_columns + war_columns], how="left", on=key_columns)
    rename_map = {column: f"{prefix}{column}" for column in war_columns}
    merged = merged.rename(columns=rename_map)

    bwar_col = f"{prefix}bwar"
    batting_col = f"{prefix}batting_bwar"
    pitching_col = f"{prefix}pitching_bwar"
    positive_col = f"{prefix}bwar_positive"
    score_col = f"{prefix}war_score"
    rank_col = f"{prefix}war_rank"
    metric_col = f"{prefix}metric_score"
    metric_rank_col = f"{prefix}metric_rank"
    metric_source_col = f"{prefix}metric_source"

    for column in [bwar_col, batting_col, pitching_col, positive_col]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)
    normalize_within_group(merged, normalization_columns, positive_col, score_col)

    group_has_war = merged.groupby(normalization_columns)[positive_col].transform("max") > 0
    merged[metric_col] = np.where(group_has_war, merged[score_col], merged[fallback_score_column])
    merged[metric_source_col] = np.where(group_has_war, "Baseball-Reference bWAR", "Lahman playing-time fallback")

    merged = merged.sort_values(
        normalization_columns + [score_col, bwar_col, fallback_score_column],
        ascending=[True] * len(normalization_columns) + [False, False, False],
    )
    merged[rank_col] = merged.groupby(normalization_columns)[score_col].rank(method="first", ascending=False).astype(int)
    merged = merged.sort_values(
        normalization_columns + [metric_col, fallback_score_column],
        ascending=[True] * len(normalization_columns) + [False, False],
    )
    merged[metric_rank_col] = merged.groupby(normalization_columns)[metric_col].rank(method="first", ascending=False).astype(int)
    return merged


def compute_player_seasons(
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    fielding: pd.DataFrame,
    appearances: pd.DataFrame,
    people: pd.DataFrame,
    key_columns: list[str],
    normalization_columns: list[str],
) -> pd.DataFrame:
    batting_cols = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP"]
    batting_agg = group_sum(batting, batting_cols, {col: f"bat_{col}" for col in batting_cols}, key_columns)

    pitching_cols = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB", "WP", "HBP", "BK", "BFP", "GF", "R", "SH", "SF", "GIDP"]
    pitching_agg = group_sum(pitching, pitching_cols, {col: f"pit_{col}" for col in pitching_cols}, key_columns)

    app_cols = ["G_all", "GS", "G_batting", "G_defense", "G_p", "G_c", "G_1b", "G_2b", "G_3b", "G_ss", "G_lf", "G_cf", "G_rf", "G_of", "G_dh", "G_ph", "G_pr"]
    app_agg = group_sum(appearances, app_cols, {col: f"app_{col}" for col in app_cols}, key_columns)

    field_cols = ["G", "GS", "InnOuts", "PO", "A", "E", "DP"]
    field_agg = group_sum(fielding, field_cols, {col: f"fld_{col}" for col in field_cols}, key_columns)
    if not fielding.empty and "POS" in fielding.columns:
        pos_rows = fielding.copy()
        pos_rows["G_num"] = numeric(pos_rows, "G")
        pos_rows = pos_rows.sort_values(key_columns + ["G_num"], ascending=[True] * len(key_columns) + [False])
        primary_pos = pos_rows.groupby(key_columns, as_index=False).first()[key_columns + ["POS"]]
        primary_pos = primary_pos.rename(columns={"POS": "primary_pos"})
    else:
        primary_pos = pd.DataFrame(columns=key_columns + ["primary_pos"])

    keys = pd.concat(
        [
            batting_agg[key_columns],
            pitching_agg[key_columns],
            app_agg[key_columns],
            field_agg[key_columns],
        ],
        ignore_index=True,
    ).drop_duplicates()

    player_seasons = keys.merge(batting_agg, how="left", on=key_columns)
    player_seasons = player_seasons.merge(pitching_agg, how="left", on=key_columns)
    player_seasons = player_seasons.merge(app_agg, how="left", on=key_columns)
    player_seasons = player_seasons.merge(field_agg, how="left", on=key_columns)
    player_seasons = player_seasons.merge(primary_pos, how="left", on=key_columns)

    num_cols = player_seasons.select_dtypes(include=["number"]).columns
    player_seasons[num_cols] = player_seasons[num_cols].fillna(0)
    player_seasons["yearID"] = player_seasons["yearID"].astype(int)

    for column in [
        "bat_G",
        "bat_AB",
        "bat_R",
        "bat_H",
        "bat_2B",
        "bat_3B",
        "bat_HR",
        "bat_RBI",
        "bat_SB",
        "bat_BB",
        "bat_HBP",
        "bat_SH",
        "bat_SF",
        "pit_G",
        "pit_GS",
        "pit_SV",
        "pit_IPouts",
        "app_G_all",
        "app_GS",
        "fld_G",
    ]:
        if column not in player_seasons.columns:
            player_seasons[column] = 0.0

    player_seasons["bat_PA_est"] = (
        player_seasons["bat_AB"]
        + player_seasons["bat_BB"]
        + player_seasons["bat_HBP"]
        + player_seasons["bat_SH"]
        + player_seasons["bat_SF"]
    )
    player_seasons["bat_TB_est"] = (
        player_seasons["bat_H"]
        + player_seasons["bat_2B"]
        + 2 * player_seasons["bat_3B"]
        + 3 * player_seasons["bat_HR"]
    )
    player_seasons["pit_IP"] = player_seasons["pit_IPouts"] / 3.0

    for source, out in [
        ("bat_PA_est", "bat_pa_norm"),
        ("bat_G", "bat_g_norm"),
        ("bat_TB_est", "bat_tb_norm"),
        ("pit_IP", "pit_ip_norm"),
        ("pit_G", "pit_g_norm"),
        ("pit_GS", "pit_gs_norm"),
        ("pit_SV", "pit_sv_norm"),
        ("app_G_all", "app_g_norm"),
        ("app_GS", "app_gs_norm"),
        ("fld_G", "fld_g_norm"),
    ]:
        normalize_within_group(player_seasons, normalization_columns, source, out)

    player_seasons["hitter_component"] = (
        0.70 * player_seasons["bat_pa_norm"]
        + 0.20 * player_seasons["bat_g_norm"]
        + 0.10 * player_seasons["bat_tb_norm"]
    )
    player_seasons["pitcher_component"] = (
        0.55 * player_seasons["pit_ip_norm"]
        + 0.18 * player_seasons["pit_g_norm"]
        + 0.17 * player_seasons["pit_gs_norm"]
        + 0.10 * player_seasons["pit_sv_norm"]
    )
    player_seasons["fielding_component"] = (
        0.62 * player_seasons["app_g_norm"]
        + 0.28 * player_seasons["app_gs_norm"]
        + 0.10 * player_seasons["fld_g_norm"]
    )
    player_seasons["raw_core_score"] = np.maximum.reduce(
        [
            player_seasons["hitter_component"],
            player_seasons["pitcher_component"],
            0.90 * player_seasons["fielding_component"],
        ]
    )
    normalize_within_group(player_seasons, normalization_columns, "raw_core_score", "core_score")

    player_seasons["primary_role"] = np.select(
        [
            (player_seasons["pit_IP"] > 0) & (player_seasons["pitcher_component"] >= player_seasons["hitter_component"] * 0.90),
            player_seasons["bat_PA_est"] > 0,
        ],
        ["Pitcher", "Hitter"],
        default="Fielder",
    )
    fallback_pos = pd.Series(
        np.where(player_seasons["primary_role"] == "Pitcher", "P", ""),
        index=player_seasons.index,
    )
    player_seasons["primary_pos"] = player_seasons["primary_pos"].fillna(fallback_pos)

    people_names = people_name_table(people)
    player_seasons = player_seasons.merge(people_names, how="left", on="playerID")
    player_seasons["player_name"] = player_seasons["player_name"].fillna(player_seasons["playerID"])

    player_seasons = player_seasons.sort_values(normalization_columns + ["core_score"], ascending=[True] * len(normalization_columns) + [False])
    player_seasons["core_rank"] = player_seasons.groupby(normalization_columns)["core_score"].rank(method="first", ascending=False).astype(int)
    player_seasons["is_top_10_core"] = player_seasons["core_rank"] <= 10
    return player_seasons


def build_player_seasons(war_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    batting = oak_filter(read_table("Batting.csv"))
    pitching = oak_filter(read_table("Pitching.csv"))
    fielding = oak_filter(read_table("Fielding.csv"))
    appearances = oak_filter(read_table("Appearances.csv"))
    people = read_table("People.csv")
    teams = oak_filter(read_table("Teams.csv"))

    player_seasons = compute_player_seasons(
        batting,
        pitching,
        fielding,
        appearances,
        people,
        key_columns=["playerID", "yearID"],
        normalization_columns=["yearID"],
    )
    oak_war = (
        war_table[war_table["teamID"] == TEAM_ID]
        .groupby(["playerID", "yearID"], as_index=False)[["bwar", "batting_bwar", "pitching_bwar", "bwar_positive"]]
        .sum(numeric_only=True)
    )
    player_seasons = add_war_metrics(
        player_seasons,
        oak_war,
        key_columns=["playerID", "yearID"],
        normalization_columns=["yearID"],
        fallback_score_column="core_score",
    )
    player_seasons["is_top_10_war"] = player_seasons["war_rank"] <= 10
    player_seasons["is_top_10_metric"] = player_seasons["metric_rank"] <= 10

    team_columns = [
        col
        for col in [
            "yearID",
            "teamID",
            "lgID",
            "divID",
            "Rank",
            "G",
            "W",
            "L",
            "DivWin",
            "WCWin",
            "LgWin",
            "WSWin",
            "R",
            "RA",
            "attendance",
            "name",
            "park",
        ]
        if col in teams.columns
    ]
    team_seasons = teams[team_columns].copy().sort_values("yearID")
    team_seasons["yearID"] = team_seasons["yearID"].astype(int)

    return player_seasons, team_seasons


def build_career_player_seasons(oak_player_ids: set[str], war_table: pd.DataFrame) -> pd.DataFrame:
    batting = season_filter(read_table("Batting.csv"))
    pitching = season_filter(read_table("Pitching.csv"))
    fielding = season_filter(read_table("Fielding.csv"))
    appearances = season_filter(read_table("Appearances.csv"))
    people = read_table("People.csv")
    teams = season_filter(read_table("Teams.csv"))

    batting = batting[batting["playerID"].isin(oak_player_ids)].copy()
    pitching = pitching[pitching["playerID"].isin(oak_player_ids)].copy()
    fielding = fielding[fielding["playerID"].isin(oak_player_ids)].copy()
    appearances = appearances[appearances["playerID"].isin(oak_player_ids)].copy()

    career = compute_player_seasons(
        batting,
        pitching,
        fielding,
        appearances,
        people,
        key_columns=["playerID", "yearID", "teamID"],
        normalization_columns=["yearID", "teamID"],
    )
    career = career.rename(
        columns={
            "core_score": "career_core_score",
            "core_rank": "career_core_rank",
            "raw_core_score": "career_raw_core_score",
        }
    )
    career = add_war_metrics(
        career,
        war_table,
        key_columns=["playerID", "yearID", "teamID"],
        normalization_columns=["yearID", "teamID"],
        fallback_score_column="career_core_score",
        prefix="career_",
    )
    career["is_oakland"] = career["teamID"] == TEAM_ID

    team_names = teams[["yearID", "teamID", "name"]].drop_duplicates().rename(columns={"name": "team_name"})
    career = career.merge(team_names, how="left", on=["yearID", "teamID"])
    career["team_name"] = career["team_name"].fillna(career["teamID"])
    career = career.sort_values(["playerID", "yearID", "is_oakland", "career_metric_score"], ascending=[True, True, False, False])
    return career


def compute_similarity(player_seasons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = list(range(START_YEAR, END_YEAR + 1))
    pivot = player_seasons.pivot_table(index="yearID", columns="playerID", values="metric_score", fill_value=0, aggfunc="sum")
    pivot = pivot.reindex(years, fill_value=0)
    values = pivot.to_numpy(dtype=float)
    rows = []
    for i, year_a in enumerate(years):
        a = values[i]
        for j, year_b in enumerate(years):
            b = values[j]
            denom = np.maximum(a, b).sum()
            similarity = float(np.minimum(a, b).sum() / denom) if denom else 0.0
            rows.append({"year_a": year_a, "year_b": year_b, "similarity": similarity})
    similarity = pd.DataFrame(rows)

    adjacent_rows = []
    for i in range(len(years) - 1):
        sim = similarity.loc[
            (similarity["year_a"] == years[i]) & (similarity["year_b"] == years[i + 1]),
            "similarity",
        ].iloc[0]
        adjacent_rows.append(
            {
                "year": years[i],
                "next_year": years[i + 1],
                "similarity": sim,
                "turnover": 1.0 - sim,
            }
        )
    adjacent = pd.DataFrame(adjacent_rows)
    return similarity, adjacent


def segment_eras(adjacent: pd.DataFrame) -> pd.DataFrame:
    threshold = max(adjacent["turnover"].quantile(0.75), adjacent["turnover"].mean())
    break_after = set(adjacent.loc[adjacent["turnover"] >= threshold, "year"].astype(int))
    rows = []
    era_index = 1
    for year in range(START_YEAR, END_YEAR + 1):
        rows.append({"yearID": year, "era_id": f"C{era_index:02d}"})
        if year in break_after:
            era_index += 1
    eras = pd.DataFrame(rows)
    spans = eras.groupby("era_id")["yearID"].agg(["min", "max", "count"]).reset_index()

    # Merge single-season fragments into the previous segment when possible.
    replacements: dict[str, str] = {}
    previous = None
    for _, row in spans.iterrows():
        era = row["era_id"]
        if row["count"] == 1 and previous is not None:
            replacements[era] = previous
        else:
            previous = era
    if replacements:
        eras["era_id"] = eras["era_id"].replace(replacements)

    unique = {era: f"C{i + 1:02d}" for i, era in enumerate(eras["era_id"].drop_duplicates())}
    eras["era_id"] = eras["era_id"].map(unique)
    return eras


def list_names(df: pd.DataFrame, limit: int = 10) -> str:
    return " | ".join(df.head(limit)["player_name"].tolist())


def build_season_summary(player_seasons: pd.DataFrame, team_seasons: pd.DataFrame, eras: pd.DataFrame) -> pd.DataFrame:
    rows = []
    top10_by_year = {
        year: set(group.loc[group["metric_rank"] <= 10, "playerID"])
        for year, group in player_seasons.groupby("yearID")
    }

    for year in range(START_YEAR, END_YEAR + 1):
        season = player_seasons[player_seasons["yearID"] == year].sort_values("metric_score", ascending=False)
        top10 = season.head(10)
        hitters = season[season["bat_PA_est"] > 0].copy()
        hitters["hitter_sort"] = hitters["batting_bwar"].clip(lower=0)
        if hitters["hitter_sort"].max() <= 0:
            hitters["hitter_sort"] = hitters["hitter_component"]
        hitters = hitters.sort_values(["hitter_sort", "hitter_component"], ascending=False).head(8)
        pitchers = season[season["pit_IP"] > 0].copy()
        pitchers["pitcher_sort"] = pitchers["pitching_bwar"].clip(lower=0)
        if pitchers["pitcher_sort"].max() <= 0:
            pitchers["pitcher_sort"] = pitchers["pitcher_component"]
        pitchers = pitchers.sort_values(["pitcher_sort", "pitcher_component"], ascending=False).head(8)
        prev = top10_by_year.get(year - 1, set())
        curr = top10_by_year.get(year, set())
        returning_ids = prev & curr
        departing_ids = prev - curr
        returning = season[season["playerID"].isin(returning_ids)].sort_values("metric_rank")
        previous_season = player_seasons[player_seasons["yearID"] == year - 1]
        departing = previous_season[previous_season["playerID"].isin(departing_ids)].sort_values("metric_rank")

        rows.append(
            {
                "yearID": year,
                "top10_players": list_names(top10, 10),
                "top_hitters": list_names(hitters, 8),
                "top_pitchers": list_names(pitchers, 8),
                "returning_core_players": list_names(returning, 10),
                "departing_core_players": list_names(departing, 10),
                "returning_core_count": len(returning_ids),
                "departing_core_count": len(departing_ids),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.merge(eras, how="left", on="yearID")
    summary = summary.merge(team_seasons, how="left", on="yearID")
    return summary


def player_totals(player_seasons: pd.DataFrame, eras: pd.DataFrame) -> pd.DataFrame:
    merged = player_seasons.merge(eras, how="left", on="yearID")
    grouped = merged.groupby(["playerID", "player_name"], as_index=False).agg(
        first_year=("yearID", "min"),
        last_year=("yearID", "max"),
        oak_seasons=("yearID", "nunique"),
        total_core=("core_score", "sum"),
        max_core=("core_score", "max"),
        avg_core=("core_score", "mean"),
        total_bwar=("bwar", "sum"),
        total_bwar_positive=("bwar_positive", "sum"),
        max_bwar=("bwar", "max"),
        total_metric_score=("metric_score", "sum"),
        max_metric_score=("metric_score", "max"),
        total_pa=("bat_PA_est", "sum"),
        total_ip=("pit_IP", "sum"),
    )
    peak_rows = (
        merged.sort_values(["playerID", "metric_score", "bwar", "core_score"], ascending=[True, False, False, False])
        .groupby("playerID", as_index=False)
        .first()
    )
    grouped = grouped.merge(
        peak_rows[["playerID", "yearID", "primary_role", "primary_pos", "era_id", "bwar", "metric_score"]],
        on="playerID",
        how="left",
    )
    grouped = grouped.rename(
        columns={
            "yearID": "peak_year",
            "era_id": "peak_era_id",
            "bwar": "peak_bwar",
            "metric_score": "peak_metric_score",
        }
    )
    grouped = grouped.sort_values(["total_bwar_positive", "total_metric_score", "total_core"], ascending=False)
    grouped["overall_rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def label_propagation(nodes: list[str], edges: pd.DataFrame) -> dict[str, int]:
    labels = {node: node for node in nodes}
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for edge in edges.itertuples(index=False):
        adjacency[edge.source].append((edge.target, float(edge.weighted_core)))
        adjacency[edge.target].append((edge.source, float(edge.weighted_core)))

    degree_order = sorted(nodes, key=lambda node: (-sum(w for _, w in adjacency[node]), node))
    for _ in range(30):
        changed = False
        for node in degree_order:
            scores: dict[str, float] = defaultdict(float)
            for neighbor, weight in adjacency[node]:
                scores[labels[neighbor]] += weight
            if not scores:
                continue
            best_label = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if best_label != labels[node]:
                labels[node] = best_label
                changed = True
        if not changed:
            break

    community_totals: dict[str, float] = defaultdict(float)
    for node in nodes:
        community_totals[labels[node]] += sum(w for _, w in adjacency[node])
    ordered = {label: i + 1 for i, (label, _) in enumerate(sorted(community_totals.items(), key=lambda item: (-item[1], item[0])))}
    return {node: ordered[labels[node]] for node in nodes}


def stable_jitter(text: str, scale: float = 0.035) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * scale


def build_network(player_seasons: pd.DataFrame, totals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = totals.head(180)["playerID"].tolist()
    pool_set = set(pool)
    source = player_seasons[player_seasons["playerID"].isin(pool_set)].copy()
    scores = source.set_index(["yearID", "playerID"])["metric_score"].to_dict()

    edge_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"shared_seasons": 0, "weighted_core": 0.0})
    for year, group in source.groupby("yearID"):
        ids = sorted(group["playerID"].unique())
        for a, b in itertools.combinations(ids, 2):
            key = (a, b)
            edge_stats[key]["shared_seasons"] += 1
            edge_stats[key]["weighted_core"] += min(scores.get((year, a), 0.0), scores.get((year, b), 0.0))

    edges = pd.DataFrame(
        [
            {
                "source": a,
                "target": b,
                "shared_seasons": stats["shared_seasons"],
                "weighted_core": stats["weighted_core"],
            }
            for (a, b), stats in edge_stats.items()
            if stats["weighted_core"] > 0
        ]
    )

    communities = label_propagation(pool, edges) if not edges.empty else {player_id: i + 1 for i, player_id in enumerate(pool)}
    nodes = totals[totals["playerID"].isin(pool_set)].copy()
    nodes["community_id"] = nodes["playerID"].map(communities).fillna(0).astype(int)

    community_order = (
        nodes.groupby("community_id")["total_bwar_positive"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    community_y = {
        community: (i + 0.5) / max(len(community_order), 1)
        for i, community in enumerate(community_order)
    }
    nodes["layout_x"] = ((nodes["peak_year"] - START_YEAR) / (END_YEAR - START_YEAR)).clip(0, 1)
    nodes["layout_y"] = nodes["community_id"].map(community_y).fillna(0.5)
    nodes["layout_x"] = (nodes["layout_x"] + nodes["playerID"].map(lambda x: stable_jitter(x, 0.055))).clip(0.02, 0.98)
    nodes["layout_y"] = (nodes["layout_y"] + nodes["playerID"].map(lambda x: stable_jitter(x[::-1], 0.075))).clip(0.03, 0.97)

    reps = (
        nodes.sort_values("total_bwar_positive", ascending=False)
        .groupby("community_id")["player_name"]
        .apply(lambda names: " / ".join(names.head(3)))
        .reset_index(name="community_label")
    )
    nodes = nodes.merge(reps, on="community_id", how="left")

    node_lookup = nodes.set_index("playerID")[["player_name", "community_id"]].to_dict("index")
    if not edges.empty:
        edges["source_name"] = edges["source"].map(lambda pid: node_lookup.get(pid, {}).get("player_name", pid))
        edges["target_name"] = edges["target"].map(lambda pid: node_lookup.get(pid, {}).get("player_name", pid))
        edges["source_community"] = edges["source"].map(lambda pid: node_lookup.get(pid, {}).get("community_id", 0))
        edges["target_community"] = edges["target"].map(lambda pid: node_lookup.get(pid, {}).get("community_id", 0))
        edges = edges.sort_values("weighted_core", ascending=False)

    return nodes.sort_values("overall_rank"), edges


def build_ribbon(player_seasons: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    ribbon = player_seasons[player_seasons["metric_rank"] <= top_n].copy()
    return ribbon[
        [
            "yearID",
            "playerID",
            "player_name",
            "metric_rank",
            "metric_score",
            "war_rank",
            "war_score",
            "bwar",
            "core_rank",
            "core_score",
            "primary_role",
            "primary_pos",
        ]
    ].sort_values(["yearID", "metric_rank"])


def top_similarity_pairs(similarity: pd.DataFrame, limit: int = 15) -> list[dict[str, Any]]:
    pairs = similarity[similarity["year_a"] < similarity["year_b"]].copy()
    return pairs.sort_values("similarity", ascending=False).head(limit).to_dict("records")


def player_lookup(totals: pd.DataFrame, name: str) -> str | None:
    exact = totals[totals["player_name"].str.lower() == name.lower()]
    if not exact.empty:
        return exact.iloc[0]["playerID"]
    contains = totals[totals["player_name"].str.lower().str.contains(name.lower(), regex=False)]
    if not contains.empty:
        return contains.iloc[0]["playerID"]
    return None


def overlap_check(player_seasons: pd.DataFrame, totals: pd.DataFrame, name_a: str, name_b: str) -> dict[str, Any]:
    id_a = player_lookup(totals, name_a)
    id_b = player_lookup(totals, name_b)
    if not id_a or not id_b:
        return {"pair": f"{name_a} / {name_b}", "found": False}

    a = player_seasons[player_seasons["playerID"] == id_a].set_index("yearID")
    b = player_seasons[player_seasons["playerID"] == id_b].set_index("yearID")
    overlap_years = sorted(set(a.index) & set(b.index))
    details = []
    for year in overlap_years:
        arow = a.loc[year]
        brow = b.loc[year]
        details.append(
            {
                "yearID": int(year),
                "a_score": round(float(arow["metric_score"]), 3),
                "b_score": round(float(brow["metric_score"]), 3),
                "a_bwar": round(float(arow["bwar"]), 1),
                "b_bwar": round(float(brow["bwar"]), 1),
                "a_core": round(float(arow["core_score"]), 3),
                "b_core": round(float(brow["core_score"]), 3),
                "a_pa": round(float(arow["bat_PA_est"]), 1),
                "b_pa": round(float(brow["bat_PA_est"]), 1),
                "a_ip": round(float(arow["pit_IP"]), 1),
                "b_ip": round(float(brow["pit_IP"]), 1),
            }
        )
    return {
        "pair": f"{a.iloc[0]['player_name']} / {b.iloc[0]['player_name']}",
        "found": True,
        "a_years": f"{int(a.index.min())}-{int(a.index.max())}",
        "b_years": f"{int(b.index.min())}-{int(b.index.max())}",
        "overlapped": bool(overlap_years),
        "overlap_years": overlap_years,
        "overlap_seasons": len(overlap_years),
        "weighted_overlap": round(sum(min(item["a_score"], item["b_score"]) for item in details), 3),
        "details": details,
    }


def build_analysis(
    player_seasons: pd.DataFrame,
    totals: pd.DataFrame,
    similarity: pd.DataFrame,
    adjacent: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    eras: pd.DataFrame,
) -> dict[str, Any]:
    most_similar = top_similarity_pairs(similarity, 12)
    biggest_turnover = adjacent.sort_values("turnover", ascending=False).head(12).to_dict("records")

    checks = [
        overlap_check(player_seasons, totals, "Jason Giambi", "Ray Durham"),
        overlap_check(player_seasons, totals, "Jason Giambi", "Mark McGwire"),
    ]

    # Specific bridge check requested by the prompt. This is not used to name eras.
    mcgwire_window = player_seasons[player_seasons["yearID"].between(1988, 1997)]
    moneyball_window = player_seasons[player_seasons["yearID"].between(2000, 2004)]
    early = mcgwire_window.groupby("playerID")["metric_score"].sum()
    late = moneyball_window.groupby("playerID")["metric_score"].sum()
    bridge_ids = sorted(set(early.index) & set(late.index))
    bridge_rows = []
    name_map = totals.set_index("playerID")["player_name"].to_dict()
    for player_id in bridge_ids:
        bridge_rows.append(
            {
                "playerID": player_id,
                "player_name": name_map.get(player_id, player_id),
                "late_mcgwire_window_score": round(float(early[player_id]), 3),
                "early_moneyball_window_score": round(float(late[player_id]), 3),
                "bridge_score": round(float(min(early[player_id], late[player_id])), 3),
            }
        )
    bridge_players = sorted(bridge_rows, key=lambda row: (-row["bridge_score"], row["player_name"]))[:12]

    node_set = set(nodes.head(140)["playerID"])
    edge_lookup = {
        tuple(sorted((row.source, row.target))): row
        for row in edges.itertuples(index=False)
        if row.source in node_set and row.target in node_set
    }
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_weight: dict[tuple[str, str], float] = {}
    for key, row in edge_lookup.items():
        a, b = key
        adjacency[a].add(b)
        adjacency[b].add(a)
        edge_weight[key] = float(row.weighted_core)

    node_info = nodes.set_index("playerID").to_dict("index")
    no_overlap_same_cluster = []
    top_ids = nodes.head(100)["playerID"].tolist()
    for a, b in itertools.combinations(top_ids, 2):
        if tuple(sorted((a, b))) in edge_lookup:
            continue
        if node_info[a]["community_id"] != node_info[b]["community_id"]:
            continue
        common = adjacency[a] & adjacency[b]
        if not common:
            continue
        common_score = sum(
            min(edge_weight.get(tuple(sorted((a, c))), 0), edge_weight.get(tuple(sorted((b, c))), 0))
            for c in common
        )
        no_overlap_same_cluster.append(
            {
                "player_a": node_info[a]["player_name"],
                "player_b": node_info[b]["player_name"],
                "community_id": int(node_info[a]["community_id"]),
                "common_teammate_score": round(common_score, 3),
                "peak_years": f"{int(node_info[a]['peak_year'])} / {int(node_info[b]['peak_year'])}",
            }
        )
    no_overlap_same_cluster = sorted(
        no_overlap_same_cluster, key=lambda row: (-row["common_teammate_score"], row["player_a"])
    )[:12]

    peak_eras = totals.set_index("playerID")["peak_era_id"].to_dict()
    technical_overlap_different_cluster = []
    for row in edges.itertuples(index=False):
        if row.source not in node_set or row.target not in node_set:
            continue
        source = node_info.get(row.source)
        target = node_info.get(row.target)
        if not source or not target:
            continue
        if peak_eras.get(row.source) == peak_eras.get(row.target):
            continue
        peak_gap = abs(int(source["peak_year"]) - int(target["peak_year"]))
        if peak_gap < 7:
            continue
        technical_overlap_different_cluster.append(
            {
                "player_a": source["player_name"],
                "player_b": target["player_name"],
                "shared_seasons": int(row.shared_seasons),
                "weighted_core": round(float(row.weighted_core), 3),
                "peak_year_gap": peak_gap,
                "peak_eras": f"{peak_eras.get(row.source)} / {peak_eras.get(row.target)}",
            }
        )
    technical_overlap_different_cluster = sorted(
        technical_overlap_different_cluster,
        key=lambda row: (-row["peak_year_gap"], row["weighted_core"]),
    )[:12]

    era_spans = eras.groupby("era_id")["yearID"].agg(["min", "max"]).reset_index()
    era_spans["label"] = era_spans.apply(lambda row: f"{row['era_id']}: {int(row['min'])}-{int(row['max'])}", axis=1)

    return {
        "generated_from": {
            "primary_source": "SABR Lahman Baseball Database CSV release plus Baseball-Reference daily WAR",
            "source_page": SABR_LAHMAN_PAGE,
            "box_folder": BOX_SHARED_URL,
            "bref_batting_war": BREF_WAR_BATTING_URL,
            "bref_pitching_war": BREF_WAR_PITCHING_URL,
            "teamID": TEAM_ID,
            "year_start": START_YEAR,
            "year_end": END_YEAR,
            "primary_metric": "Baseball-Reference bWAR, clamped to zero for visual weights",
        },
        "most_similar_seasons": most_similar,
        "biggest_adjacent_turnover": biggest_turnover,
        "specific_checks": checks,
        "bridge_players_late_mcgwire_to_early_moneyball": bridge_players,
        "same_cluster_no_overlap": no_overlap_same_cluster,
        "technical_overlap_different_peak_eras": technical_overlap_different_cluster,
        "candidate_eras": era_spans.to_dict("records"),
    }


def clean_records(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    frame = df[columns].copy() if columns else df.copy()
    frame = frame.replace({np.nan: None})
    records = frame.to_dict("records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, (np.integer,)):
                record[key] = int(value)
            elif isinstance(value, (np.floating,)):
                record[key] = round(float(value), 6)
    return records


def write_app_json(
    player_seasons: pd.DataFrame,
    career_player_seasons: pd.DataFrame,
    totals: pd.DataFrame,
    season_summary: pd.DataFrame,
    similarity: pd.DataFrame,
    adjacent: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    ribbon: pd.DataFrame,
    analysis: dict[str, Any],
) -> None:
    app_data = {
        "meta": {
            "teamID": TEAM_ID,
            "year_start": START_YEAR,
            "year_end": END_YEAR,
            "source_page": SABR_LAHMAN_PAGE,
            "source_box_folder": BOX_SHARED_URL,
            "bref_batting_war": BREF_WAR_BATTING_URL,
            "bref_pitching_war": BREF_WAR_PITCHING_URL,
            "primary_metric": "Baseball-Reference bWAR",
            "primary_metric_definition": "Positive bWAR, normalized within season/team, with Lahman playing-time core as fallback only when WAR is unavailable.",
            "core_definition": "Within-season normalized max of hitter, pitcher, and fielding/appearance playing-time components.",
        },
        "playerSeasons": clean_records(
            player_seasons,
            [
                "playerID",
                "player_name",
                "yearID",
                "bwar",
                "batting_bwar",
                "pitching_bwar",
                "bwar_positive",
                "war_score",
                "war_rank",
                "metric_score",
                "metric_rank",
                "metric_source",
                "core_score",
                "core_rank",
                "primary_role",
                "primary_pos",
                "bat_PA_est",
                "pit_IP",
                "app_G_all",
            ],
        ),
        "careerSeasons": clean_records(
            career_player_seasons,
            [
                "playerID",
                "player_name",
                "yearID",
                "teamID",
                "team_name",
                "is_oakland",
                "career_bwar",
                "career_batting_bwar",
                "career_pitching_bwar",
                "career_bwar_positive",
                "career_war_score",
                "career_war_rank",
                "career_metric_score",
                "career_metric_rank",
                "career_metric_source",
                "career_core_score",
                "career_core_rank",
                "primary_role",
                "primary_pos",
                "bat_PA_est",
                "pit_IP",
                "app_G_all",
            ],
        ),
        "playerTotals": clean_records(
            totals.head(260),
            [
                "playerID",
                "player_name",
                "first_year",
                "last_year",
                "oak_seasons",
                "total_bwar",
                "total_bwar_positive",
                "max_bwar",
                "peak_bwar",
                "total_metric_score",
                "max_metric_score",
                "total_core",
                "max_core",
                "peak_year",
                "peak_metric_score",
                "primary_role",
                "primary_pos",
                "peak_era_id",
                "overall_rank",
            ],
        ),
        "seasonSummary": clean_records(season_summary),
        "similarity": clean_records(similarity),
        "adjacentTurnover": clean_records(adjacent),
        "networkNodes": clean_records(nodes.head(140)),
        "networkEdges": clean_records(edges.head(1600)),
        "ribbon": clean_records(ribbon),
        "analysis": analysis,
    }
    with (PUBLIC_DATA_DIR / "app_data.json").open("w", encoding="utf-8") as handle:
        json.dump(app_data, handle, indent=2)


def oak_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "oak",
        ["#f5f7f2", "#d5e2c6", "#8fb47a", "#31734d", "#003831"],
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(EXPORT_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(EXPORT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def visual_score(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return float(np.clip(max(float(value), 0.0) / max_value, 0.0, 1.0))


def yes_flag(value: Any) -> bool:
    return str(value).upper() == "Y"


def shade_playoff_columns(ax: plt.Axes, season_summary: pd.DataFrame) -> None:
    for row in season_summary.itertuples(index=False):
        playoff = (
            yes_flag(getattr(row, "DivWin", ""))
            or yes_flag(getattr(row, "WCWin", ""))
            or yes_flag(getattr(row, "LgWin", ""))
            or yes_flag(getattr(row, "WSWin", ""))
        )
        if not playoff:
            continue
        is_ws_winner = yes_flag(getattr(row, "WSWin", ""))
        ax.axvspan(
            float(row.yearID) - 0.5,
            float(row.yearID) + 0.5,
            color="#f1d36b" if is_ws_winner else "#dcead2",
            alpha=0.48 if is_ws_winner else 0.42,
            linewidth=0,
            zorder=0,
        )


def export_timeline(player_seasons: pd.DataFrame, totals: pd.DataFrame, season_summary: pd.DataFrame) -> None:
    top_ids = totals.head(60)["playerID"].tolist()
    source = player_seasons[player_seasons["playerID"].isin(top_ids)].copy()
    order = totals[totals["playerID"].isin(top_ids)].sort_values(["first_year", "last_year", "total_bwar_positive"])
    y_lookup = {pid: i for i, pid in enumerate(order["playerID"])}
    name_lookup = order.set_index("playerID")["player_name"].to_dict()
    role_colors = {"Hitter": "#EFB21E", "Pitcher": "#006B54", "Fielder": "#5B6770"}
    visual_max = max(float(player_seasons["bwar_positive"].max()), 1.0)

    fig, ax = plt.subplots(figsize=(17, 14))
    ax.set_facecolor("#fbfbf7")
    shade_playoff_columns(ax, season_summary)
    for row in source.itertuples(index=False):
        y = y_lookup[row.playerID]
        score = visual_score(row.bwar_positive, visual_max)
        height = 0.18 + 0.55 * score
        ax.add_patch(
            Rectangle(
                (row.yearID - 0.46, y - height / 2),
                0.92,
                height,
                facecolor=role_colors.get(row.primary_role, "#5B6770"),
                edgecolor="none",
                alpha=0.22 + 0.78 * score,
            )
        )
    ax.set_xlim(START_YEAR - 0.8, END_YEAR + 0.8)
    ax.set_ylim(-1, len(order))
    ax.set_yticks(list(y_lookup.values()))
    ax.set_yticklabels([name_lookup[pid] for pid in order["playerID"]], fontsize=8)
    ax.set_xticks(list(range(1970, END_YEAR + 1, 5)))
    ax.grid(axis="x", color="#d9ded5", linewidth=0.7)
    ax.set_title("Oakland A's Player Timeline: Top 60 by Total Oakland bWAR, Bars on Overall bWAR Scale", loc="left", fontsize=16, weight="bold")
    ax.set_xlabel("Season")
    ax.invert_yaxis()
    save_fig(fig, "player_timeline_top60")


def aggregate_career_overlay_rows(career_source: pd.DataFrame) -> pd.DataFrame:
    if career_source.empty:
        return career_source.copy()

    keys = ["playerID", "yearID"]
    sorted_source = career_source.sort_values(keys + ["career_metric_score"], ascending=[True, True, False])
    overlay = sorted_source.groupby(keys, as_index=False).first()

    sum_columns = [
        "career_bwar",
        "career_batting_bwar",
        "career_pitching_bwar",
        "career_bwar_positive",
        "bat_PA_est",
        "pit_IP",
    ]
    present_sum_columns = [column for column in sum_columns if column in career_source.columns]
    if present_sum_columns:
        sums = career_source.groupby(keys, as_index=False)[present_sum_columns].sum(numeric_only=True)
        overlay = overlay.drop(columns=present_sum_columns, errors="ignore").merge(sums, on=keys, how="left")

    for score_column in ["career_war_score", "career_metric_score", "career_core_score"]:
        if score_column not in career_source.columns:
            continue
        score_sums = (
            career_source.groupby(keys, as_index=False)[score_column]
            .sum(numeric_only=True)
            .rename(columns={score_column: f"{score_column}_combined"})
        )
        overlay = overlay.merge(score_sums, on=keys, how="left")
        overlay[score_column] = overlay[f"{score_column}_combined"].clip(upper=1).fillna(0)
        overlay = overlay.drop(columns=[f"{score_column}_combined"])

    if {"teamID", "team_name"}.issubset(career_source.columns):
        teams = (
            career_source.groupby(keys, as_index=False)
            .agg(
                teamID=("teamID", lambda values: " + ".join(dict.fromkeys(values.dropna().astype(str)))),
                team_name=("team_name", lambda values: " + ".join(dict.fromkeys(values.dropna().astype(str)))),
            )
        )
        overlay = overlay.drop(columns=["teamID", "team_name"], errors="ignore").merge(teams, on=keys, how="left")

    return overlay


def export_timeline_career(
    player_seasons: pd.DataFrame,
    career_player_seasons: pd.DataFrame,
    totals: pd.DataFrame,
    season_summary: pd.DataFrame,
) -> None:
    top_ids = totals.head(60)["playerID"].tolist()
    oak_source = player_seasons[player_seasons["playerID"].isin(top_ids)].copy()
    all_non_oak_career = aggregate_career_overlay_rows(career_player_seasons[~career_player_seasons["is_oakland"]].copy())
    career_source = aggregate_career_overlay_rows(
        career_player_seasons[
            career_player_seasons["playerID"].isin(top_ids) & ~career_player_seasons["is_oakland"]
        ].copy()
    )
    order = totals[totals["playerID"].isin(top_ids)].sort_values("total_bwar_positive", ascending=False)
    y_lookup = {pid: i for i, pid in enumerate(order["playerID"])}
    name_lookup = order.set_index("playerID")["player_name"].to_dict()
    role_colors = {"Hitter": "#EFB21E", "Pitcher": "#006B54", "Fielder": "#5B6770"}
    career_max = float(all_non_oak_career["career_bwar_positive"].max()) if not all_non_oak_career.empty else 0.0
    visual_max = max(float(player_seasons["bwar_positive"].max()), career_max, 1.0)

    fig, ax = plt.subplots(figsize=(17, 14))
    ax.set_facecolor("#fbfbf7")
    shade_playoff_columns(ax, season_summary)
    for row in career_source.itertuples(index=False):
        y = y_lookup[row.playerID]
        score = visual_score(row.career_bwar_positive, visual_max)
        height = 0.18 + 0.55 * score
        ax.add_patch(
            Rectangle(
                (row.yearID - 0.43, y - height / 2),
                0.86,
                height,
                facecolor="#8a8f8b",
                edgecolor="none",
                alpha=0.22 + 0.68 * score,
            )
        )
    for row in oak_source.itertuples(index=False):
        y = y_lookup[row.playerID]
        score = visual_score(row.bwar_positive, visual_max)
        height = 0.18 + 0.55 * score
        ax.add_patch(
            Rectangle(
                (row.yearID - 0.46, y - height / 2),
                0.92,
                height,
                facecolor=role_colors.get(row.primary_role, "#5B6770"),
                edgecolor="none",
                alpha=0.24 + 0.76 * score,
            )
        )
    ax.set_xlim(START_YEAR - 0.8, END_YEAR + 0.8)
    ax.set_ylim(-1, len(order))
    ax.set_yticks(list(y_lookup.values()))
    ax.set_yticklabels([name_lookup[pid] for pid in order["playerID"]], fontsize=8)
    ax.set_xticks(list(range(1970, END_YEAR + 1, 5)))
    ax.grid(axis="x", color="#d9ded5", linewidth=0.7)
    ax.set_title("Oakland A's Player Timeline with Full MLB Career Overlay, Bars on Overall bWAR Scale", loc="left", fontsize=16, weight="bold")
    ax.set_xlabel("Season")
    ax.invert_yaxis()
    save_fig(fig, "player_timeline_career_top60")


def export_heatmap(similarity: pd.DataFrame) -> None:
    years = list(range(START_YEAR, END_YEAR + 1))
    matrix = similarity.pivot(index="year_a", columns="year_b", values="similarity").reindex(index=years, columns=years)
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(matrix.values, cmap=oak_cmap(), vmin=0, vmax=1)
    tick_positions = [i for i, year in enumerate(years) if year % 5 == 0]
    tick_labels = [years[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_title("Oakland A's Season Similarity Heatmap: bWAR-Weighted Core", loc="left", fontsize=16, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="bWAR-weighted Jaccard similarity")
    save_fig(fig, "season_similarity_heatmap")


def export_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    top_nodes = nodes.head(110).copy()
    node_ids = set(top_nodes["playerID"])
    top_edges = edges[(edges["source"].isin(node_ids)) & (edges["target"].isin(node_ids))].head(900)
    coords = top_nodes.set_index("playerID")[["layout_x", "layout_y"]].to_dict("index")
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor("#fbfbf7")
    for edge in top_edges.itertuples(index=False):
        a = coords.get(edge.source)
        b = coords.get(edge.target)
        if not a or not b:
            continue
        ax.plot(
            [a["layout_x"], b["layout_x"]],
            [a["layout_y"], b["layout_y"]],
            color="#6f7a6f",
            alpha=min(0.08 + edge.weighted_core / 8, 0.45),
            linewidth=0.2 + min(edge.weighted_core, 5) * 0.28,
        )
    size_metric = top_nodes["total_bwar_positive"].replace(0, np.nan)
    size_metric = size_metric.fillna(top_nodes["total_metric_score"]).replace(0, np.nan).fillna(top_nodes["total_core"])
    sizes = 30 + 270 * (size_metric / size_metric.max())
    scatter = ax.scatter(
        top_nodes["layout_x"],
        top_nodes["layout_y"],
        s=sizes,
        c=top_nodes["peak_year"],
        cmap="viridis",
        edgecolors="#12352f",
        linewidths=0.4,
        alpha=0.92,
    )
    for row in top_nodes.head(35).itertuples(index=False):
        ax.text(row.layout_x + 0.006, row.layout_y, row.player_name, fontsize=7, va="center")
    ax.set_xticks(np.linspace(0, 1, 8))
    ax.set_xticklabels([str(round(START_YEAR + x * (END_YEAR - START_YEAR))) for x in np.linspace(0, 1, 8)])
    ax.set_yticks([])
    ax.set_title("Oakland A's Player Overlap Network: Top 110 by Oakland bWAR", loc="left", fontsize=16, weight="bold")
    ax.set_xlabel("Peak Oakland season")
    fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02, label="Peak year")
    save_fig(fig, "player_overlap_network")


def export_ribbon(ribbon: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_facecolor("#fbfbf7")
    years = list(range(START_YEAR, END_YEAR + 1))
    source = ribbon[ribbon["metric_rank"] <= 12].copy()
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_map = {}
    for i, player_id in enumerate(source["playerID"].drop_duplicates()):
        color_map[player_id] = colors[i % len(colors)]
    for player_id, group in source.groupby("playerID"):
        group = group.sort_values("yearID")
        xs = group["yearID"].to_numpy()
        ys = group["metric_rank"].to_numpy()
        ax.plot(xs, ys, color=color_map[player_id], linewidth=0.5 + 2.5 * group["metric_score"].mean(), alpha=0.65)
    for year in [1968, 1972, 1989, 2002, 2012, 2024]:
        labels = source[source["yearID"] == year].sort_values("metric_rank").head(8)
        for row in labels.itertuples(index=False):
            ax.text(year + 0.15, row.metric_rank, row.player_name, fontsize=6, va="center")
    ax.set_xlim(START_YEAR - 0.5, END_YEAR + 1.5)
    ax.set_ylim(12.7, 0.3)
    ax.set_yticks(range(1, 13))
    ax.set_xticks(list(range(1970, END_YEAR + 1, 5)))
    ax.grid(axis="x", color="#d9ded5", linewidth=0.7)
    ax.set_title("Oakland A's Era Ribbon: Top 12 bWAR Core Players per Season", loc="left", fontsize=16, weight="bold")
    ax.set_ylabel("bWAR rank within season")
    save_fig(fig, "era_ribbon_top12")


def write_static_exports(
    player_seasons: pd.DataFrame,
    career_player_seasons: pd.DataFrame,
    totals: pd.DataFrame,
    season_summary: pd.DataFrame,
    similarity: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    ribbon: pd.DataFrame,
) -> None:
    export_timeline(player_seasons, totals, season_summary)
    export_timeline_career(player_seasons, career_player_seasons, totals, season_summary)
    export_heatmap(similarity)
    export_network(nodes, edges)
    export_ribbon(ribbon)


def write_readme_summary(analysis: dict[str, Any]) -> None:
    lines = [
        "# Oakland A's Roster-Era Visualization Data Notes",
        "",
        "Generated by `scripts/build_oakland_eras.py` from Lahman CSV files plus Baseball-Reference daily WAR.",
        "",
        "## Quick Analysis Outputs",
        "",
        "### Most Roster-Similar Seasons",
    ]
    for row in analysis["most_similar_seasons"][:8]:
        lines.append(f"- {int(row['year_a'])} and {int(row['year_b'])}: {row['similarity']:.3f}")
    lines.extend(["", "### Biggest Adjacent-Year Core Turnover"])
    for row in analysis["biggest_adjacent_turnover"][:8]:
        lines.append(f"- {int(row['year'])} to {int(row['next_year'])}: turnover {row['turnover']:.3f}")
    lines.extend(["", "### Specific Checks"])
    for check in analysis["specific_checks"]:
        if not check.get("found"):
            lines.append(f"- {check['pair']}: player lookup failed")
            continue
        years = ", ".join(map(str, check["overlap_years"])) or "none"
        lines.append(f"- {check['pair']}: overlap years {years}; weighted overlap {check['weighted_overlap']}")
    lines.extend(["", "### Bridge Players: Late McGwire Window to Early Moneyball Window"])
    for row in analysis["bridge_players_late_mcgwire_to_early_moneyball"][:8]:
        lines.append(f"- {row['player_name']}: bridge score {row['bridge_score']}")

    (PROCESSED_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_csvs_to_public() -> None:
    for name in [
        "player_seasons.csv",
        "career_player_seasons.csv",
        "season_summary.csv",
        "season_similarity.csv",
        "adjacent_turnover.csv",
        "network_nodes.csv",
        "network_edges.csv",
        "ribbon_players.csv",
    ]:
        shutil.copy2(PROCESSED_DIR / name, PUBLIC_DATA_DIR / name)


def run_pipeline(download: bool = False, force_download: bool = False, force_war_download: bool = False) -> None:
    ensure_dirs()
    if download or force_download:
        download_lahman_tables(force=force_download)

    missing = [name for name in REQUIRED_TABLES if not (RAW_DIR / name).exists()]
    if missing:
        download_lahman_tables(force=False)

    if force_war_download:
        download_bref_war_files(force=True)

    print("Loading Baseball-Reference WAR...")
    people = read_table("People.csv")
    teams = read_table("Teams.csv")
    war_table = build_bref_war_table(people, teams)

    print("Building player-season table...")
    player_seasons, team_seasons = build_player_seasons(war_table)
    print("Building full-career player-season table...")
    career_player_seasons = build_career_player_seasons(set(player_seasons["playerID"]), war_table)
    print("Computing season similarity...")
    similarity, adjacent = compute_similarity(player_seasons)
    eras = segment_eras(adjacent)
    season_summary = build_season_summary(player_seasons, team_seasons, eras)
    totals = player_totals(player_seasons, eras)
    print("Building player overlap network...")
    nodes, edges = build_network(player_seasons, totals)
    ribbon = build_ribbon(player_seasons)
    analysis = build_analysis(player_seasons, totals, similarity, adjacent, nodes, edges, eras)

    print("Writing processed CSVs and app JSON...")
    player_seasons.to_csv(PROCESSED_DIR / "player_seasons.csv", index=False)
    career_player_seasons.to_csv(PROCESSED_DIR / "career_player_seasons.csv", index=False)
    season_summary.to_csv(PROCESSED_DIR / "season_summary.csv", index=False)
    similarity.to_csv(PROCESSED_DIR / "season_similarity.csv", index=False)
    adjacent.to_csv(PROCESSED_DIR / "adjacent_turnover.csv", index=False)
    totals.to_csv(PROCESSED_DIR / "player_totals.csv", index=False)
    nodes.to_csv(PROCESSED_DIR / "network_nodes.csv", index=False)
    edges.to_csv(PROCESSED_DIR / "network_edges.csv", index=False)
    ribbon.to_csv(PROCESSED_DIR / "ribbon_players.csv", index=False)
    with (PROCESSED_DIR / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    write_readme_summary(analysis)
    copy_csvs_to_public()
    write_app_json(player_seasons, career_player_seasons, totals, season_summary, similarity, adjacent, nodes, edges, ribbon, analysis)
    print("Rendering static exports...")
    write_static_exports(player_seasons, career_player_seasons, totals, season_summary, similarity, nodes, edges, ribbon)
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing Lahman CSV files before processing.")
    parser.add_argument("--force-download", action="store_true", help="Re-download Lahman CSV files even if cached.")
    parser.add_argument("--force-war-download", action="store_true", help="Re-download Baseball-Reference WAR files even if cached.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(download=args.download, force_download=args.force_download, force_war_download=args.force_war_download)
