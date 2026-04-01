from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() == ".sas7bdat":
        return pd.read_sas(path, format="sas7bdat", encoding="latin1")
    raise ValueError(f"Unsupported file type for {path}. Use .csv, .txt, or .parquet.")


def _require_columns(df: pd.DataFrame, required: list[str], *, label: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")
    return df


def load_synthetic_population(path: str | Path) -> pd.DataFrame:
    df = read_table(path).copy()
    required = ["area", "sex"]
    _require_columns(df, required, label="Synthetic population")

    if "person_id" not in df.columns:
        df["person_id"] = df.index.astype(str)
    if "household_id" not in df.columns and "HID" in df.columns:
        df["household_id"] = df["HID"]
    if "age" not in df.columns and "agegrp" in df.columns:
        # Downstream code can still work with agegrp-only, but age is better.
        df["age"] = pd.NA
    return df


def load_harmonized_tus_respondents(path: str | Path) -> pd.DataFrame:
    df = read_table(path).copy()
    required = ["respondent_id", "sex", "age"]
    _require_columns(df, required, label="TUS respondents")

    if "person_weight" not in df.columns:
        df["person_weight"] = 1.0
    if "day_type" not in df.columns:
        df["day_type"] = "weekday"
    return df


def load_harmonized_tus_episodes(path: str | Path) -> pd.DataFrame:
    df = read_table(path).copy()
    required = ["respondent_id", "start_minute", "end_minute", "activity_code"]
    _require_columns(df, required, label="TUS episodes")
    return df
