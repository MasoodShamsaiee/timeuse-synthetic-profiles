from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _copy_and_rename(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    missing = [src for src in column_map if src not in df.columns]
    if missing:
        raise KeyError(f"Raw TUS file is missing mapped source columns: {missing}")
    return df.rename(columns=column_map).copy()


def _apply_value_map(series: pd.Series, value_map: dict[Any, Any] | None) -> pd.Series:
    if not value_map:
        return series
    return series.map(lambda x: value_map.get(x, value_map.get(str(x), x)))


def hhmm_to_minutes(value: Any) -> int | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None

    if ":" in s:
        parts = s.split(":")
        if len(parts) >= 2:
            try:
                hh = int(parts[0])
                mm = int(parts[1])
                return hh * 60 + mm
            except Exception:
                return None

    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    if len(digits) <= 2:
        try:
            return int(digits) * 60
        except Exception:
            return None
    if len(digits) == 3:
        digits = f"0{digits}"
    try:
        hh = int(digits[:2])
        mm = int(digits[2:4])
    except Exception:
        return None
    return hh * 60 + mm


def harmonize_tus_respondents(
    raw_df: pd.DataFrame,
    *,
    column_map: dict[str, str],
    sex_map: dict[Any, str] | None = None,
    day_type_map: dict[Any, str] | None = None,
    yes_no_maps: dict[str, dict[Any, int]] | None = None,
    categorical_maps: dict[str, dict[Any, Any]] | None = None,
    defaults: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Convert a raw TUS respondent file into the standardized schema expected by the PoC.

    Required standardized columns after renaming:
    - respondent_id
    - sex
    - age

    Common optional standardized columns:
    - day_type
    - person_weight
    - household_size
    - has_children
    - is_student
    - is_worker
    - works_from_home
    - employment_cat
    - student_cat
    - commute_cat
    - tenure_cat
    """
    work = _copy_and_rename(raw_df, column_map)

    required = ["respondent_id", "sex", "age"]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise KeyError(f"Harmonized respondent table is missing required columns: {missing}")

    work["respondent_id"] = work["respondent_id"].astype(str)
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work["sex"] = _apply_value_map(work["sex"], sex_map).astype("string")

    if "day_type" in work.columns:
        work["day_type"] = _apply_value_map(work["day_type"], day_type_map).astype("string")

    if yes_no_maps:
        for col, value_map in yes_no_maps.items():
            if col in work.columns:
                work[col] = pd.to_numeric(_apply_value_map(work[col], value_map), errors="coerce")

    if categorical_maps:
        for col, value_map in categorical_maps.items():
            if col in work.columns:
                work[col] = _apply_value_map(work[col], value_map).astype("string")

    if defaults:
        for col, value in defaults.items():
            if col not in work.columns:
                work[col] = value

    if "person_weight" in work.columns:
        work["person_weight"] = pd.to_numeric(work["person_weight"], errors="coerce")

    return work


def harmonize_tus_episodes(
    raw_df: pd.DataFrame,
    *,
    column_map: dict[str, str],
    activity_map: dict[Any, Any] | None = None,
    location_map: dict[Any, Any] | None = None,
    start_format: str = "minutes",
    end_format: str = "minutes",
) -> pd.DataFrame:
    """
    Convert a raw TUS episode file into the standardized schema expected by the PoC.

    Required standardized columns after renaming:
    - respondent_id
    - start_minute
    - end_minute
    - activity_code
    """
    work = _copy_and_rename(raw_df, column_map)

    required = ["respondent_id", "start_minute", "end_minute", "activity_code"]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise KeyError(f"Harmonized episode table is missing required columns: {missing}")

    work["respondent_id"] = work["respondent_id"].astype(str)

    if start_format == "hhmm":
        work["start_minute"] = work["start_minute"].map(hhmm_to_minutes)
    else:
        work["start_minute"] = pd.to_numeric(work["start_minute"], errors="coerce")

    if end_format == "hhmm":
        work["end_minute"] = work["end_minute"].map(hhmm_to_minutes)
    else:
        work["end_minute"] = pd.to_numeric(work["end_minute"], errors="coerce")

    work["activity_code"] = _apply_value_map(work["activity_code"], activity_map)
    if "location_code" in work.columns:
        work["location_code"] = _apply_value_map(work["location_code"], location_map)

    work = work.dropna(subset=["respondent_id", "start_minute", "end_minute", "activity_code"]).copy()
    work["start_minute"] = work["start_minute"].astype(int)
    work["end_minute"] = work["end_minute"].astype(int)
    work = work.loc[work["end_minute"] > work["start_minute"]].reset_index(drop=True)
    return work


def save_harmonized_tables(
    respondents: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    out_dir: str | Path,
    respondents_name: str = "tus_respondents_harmonized.parquet",
    episodes_name: str = "tus_episodes_harmonized.parquet",
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_path = out_dir / respondents_name
    epi_path = out_dir / episodes_name
    respondents.to_parquet(resp_path, index=False)
    episodes.to_parquet(epi_path, index=False)
    return resp_path, epi_path


def _tus2022_agegr10_to_midpoint(series: pd.Series) -> pd.Series:
    mapping = {
        1: 20,
        2: 30,
        3: 40,
        4: 50,
        5: 60,
        6: 70,
        7: 80,
    }
    return pd.to_numeric(series, errors="coerce").map(mapping)


def _tus2022_day_type(series: pd.Series) -> pd.Series:
    mapping = {
        1: "weekday",
        2: "weekend",
        3: "weekend",
    }
    return pd.to_numeric(series, errors="coerce").map(mapping).astype("string")


def _tus2022_yes_no(series: pd.Series) -> pd.Series:
    mapping = {1: 1, 2: 0}
    return pd.to_numeric(series, errors="coerce").map(mapping)


def harmonize_tus2022_main_pumf(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    First-pass harmonization for the 2022 TUS PUMF main file.

    This uses stable columns observed in the official 2022 layout files:
    - PUMFID
    - WGHT_PER
    - AGEGR10
    - GENDER2
    - HSDSIZEC
    - CHH0017C
    - DVTDAY
    - ACT7DAYC
    - MRW_D40B
    - TLWK_01A
    - CTW_140I
    """
    required = [
        "PUMFID",
        "WGHT_PER",
        "AGEGR10",
        "GENDER2",
        "HSDSIZEC",
        "CHH0017C",
        "DVTDAY",
    ]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise KeyError(f"TUS 2022 main PUMF is missing expected columns: {missing}")

    out = pd.DataFrame()
    out["respondent_id"] = raw_df["PUMFID"].astype(str)
    out["person_weight"] = pd.to_numeric(raw_df["WGHT_PER"], errors="coerce")
    out["sex"] = pd.to_numeric(raw_df["GENDER2"], errors="coerce").map({1: "male", 2: "female"}).astype("string")
    out["age"] = _tus2022_agegr10_to_midpoint(raw_df["AGEGR10"])
    out["age_group_10"] = pd.to_numeric(raw_df["AGEGR10"], errors="coerce")
    out["household_size"] = pd.to_numeric(raw_df["HSDSIZEC"], errors="coerce")
    out["has_children"] = pd.to_numeric(raw_df["CHH0017C"], errors="coerce").map({0: 0, 1: 1, 2: 1, 3: 1})
    out["day_type"] = _tus2022_day_type(raw_df["DVTDAY"])
    out["worked_last_week"] = _tus2022_yes_no(raw_df["MRW_D40B"]) if "MRW_D40B" in raw_df.columns else pd.NA
    out["teleworked_from_home_last_week"] = _tus2022_yes_no(raw_df["TLWK_01A"]) if "TLWK_01A" in raw_df.columns else pd.NA
    out["worked_or_studied_at_home"] = _tus2022_yes_no(raw_df["CTW_140I"]) if "CTW_140I" in raw_df.columns else pd.NA

    if "ACT7DAYC" in raw_df.columns:
        act7 = pd.to_numeric(raw_df["ACT7DAYC"], errors="coerce")
        out["main_activity_last_week"] = act7
        out["employment_cat"] = act7.map(
            {
                1: "employed",
                2: "student",
                3: "home_care",
                4: "retired",
                5: "other",
            }
        ).astype("string")
        out["is_student"] = act7.map({2: 1}).fillna(0).astype(int)
        out["is_worker"] = act7.map({1: 1}).fillna(0).astype(int)
    else:
        out["employment_cat"] = "unknown"
        out["is_student"] = 0
        out["is_worker"] = 0

    if "CTW_150G" in raw_df.columns:
        out["commute_cat"] = pd.to_numeric(raw_df["CTW_150G"], errors="coerce").astype("Int64").astype("string")
    else:
        out["commute_cat"] = "unknown"

    return out


def harmonize_tus2022_episode_pumf(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    First-pass harmonization for the 2022 TUS PUMF episode file.

    Uses:
    - PUMFID
    - INSTANCE
    - STARTMIN / ENDMIN
    - STARTIME / ENDTIME
    - TUI_01 for detailed activity code
    - ACTIVITY for grouped activity code
    - LOCATION
    - WGHT_EPI
    """
    required = ["PUMFID", "INSTANCE", "STARTMIN", "ENDMIN", "TUI_01", "ACTIVITY", "LOCATION"]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise KeyError(f"TUS 2022 episode PUMF is missing expected columns: {missing}")

    out = pd.DataFrame()
    out["respondent_id"] = raw_df["PUMFID"].astype(str)
    out["episode_id"] = pd.to_numeric(raw_df["INSTANCE"], errors="coerce")
    out["start_minute"] = pd.to_numeric(raw_df["STARTMIN"], errors="coerce")
    out["end_minute"] = pd.to_numeric(raw_df["ENDMIN"], errors="coerce")
    out["start_time_raw"] = raw_df["STARTIME"] if "STARTIME" in raw_df.columns else pd.NA
    out["end_time_raw"] = raw_df["ENDTIME"] if "ENDTIME" in raw_df.columns else pd.NA
    out["activity_code"] = pd.to_numeric(raw_df["TUI_01"], errors="coerce")
    out["activity_group"] = pd.to_numeric(raw_df["ACTIVITY"], errors="coerce")
    out["location_code"] = pd.to_numeric(raw_df["LOCATION"], errors="coerce")
    out["episode_weight"] = pd.to_numeric(raw_df["WGHT_EPI"], errors="coerce") if "WGHT_EPI" in raw_df.columns else pd.NA

    out = out.dropna(subset=["respondent_id", "start_minute", "end_minute", "activity_code"]).copy()
    out["start_minute"] = out["start_minute"].astype(int)
    out["end_minute"] = out["end_minute"].astype(int)
    out = out.loc[(out["start_minute"] >= 0) & (out["end_minute"] > out["start_minute"])].reset_index(drop=True)
    return out
