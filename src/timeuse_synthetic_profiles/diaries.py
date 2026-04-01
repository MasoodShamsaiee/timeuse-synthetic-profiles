from __future__ import annotations

import numpy as np
import pandas as pd


def classify_episode_states(
    episodes: pd.DataFrame,
    *,
    home_activity_codes: set | None = None,
    sleep_activity_codes: set | None = None,
    home_location_values: set | None = None,
    activity_col: str = "activity_code",
    location_col: str | None = None,
) -> pd.DataFrame:
    work = episodes.copy()
    home_activity_codes = set() if home_activity_codes is None else set(home_activity_codes)
    sleep_activity_codes = set() if sleep_activity_codes is None else set(sleep_activity_codes)
    home_location_values = set() if home_location_values is None else set(home_location_values)

    is_sleep = work[activity_col].isin(sleep_activity_codes)
    if location_col is not None and location_col in work.columns and home_location_values:
        is_home = work[location_col].isin(home_location_values) | work[activity_col].isin(home_activity_codes)
    else:
        is_home = work[activity_col].isin(home_activity_codes) | is_sleep

    work["state"] = np.select(
        [is_sleep, is_home],
        ["sleep", "home_awake"],
        default="away",
    )
    return work


def expand_episodes_to_hourly_states(
    episodes: pd.DataFrame,
    *,
    respondent_id_col: str = "respondent_id",
    start_col: str = "start_minute",
    end_col: str = "end_minute",
    state_col: str = "state",
) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in episodes.itertuples(index=False):
        respondent_id = getattr(rec, respondent_id_col)
        start_minute = int(getattr(rec, start_col))
        end_minute = int(getattr(rec, end_col))
        state = getattr(rec, state_col)

        if end_minute <= start_minute:
            continue

        for hour in range(start_minute // 60, (end_minute - 1) // 60 + 1):
            hour_start = hour * 60
            hour_end = hour_start + 60
            overlap = max(0, min(end_minute, hour_end) - max(start_minute, hour_start))
            if overlap <= 0:
                continue
            rows.append(
                {
                    "respondent_id": respondent_id,
                    "hour": int(hour % 24),
                    "state": state,
                    "minutes": int(overlap),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["respondent_id", "hour", "state", "minutes"])

    hourly = pd.DataFrame(rows)
    hourly = (
        hourly.groupby(["respondent_id", "hour", "state"], as_index=False)["minutes"]
        .sum()
        .sort_values(["respondent_id", "hour", "state"])
        .reset_index(drop=True)
    )
    return hourly


def summarize_respondent_profiles(hourly_states: pd.DataFrame) -> pd.DataFrame:
    if hourly_states.empty:
        return pd.DataFrame(
            columns=[
                "respondent_id",
                "hour",
                "minutes_home_awake",
                "minutes_sleep",
                "minutes_away",
                "share_home_awake",
                "share_sleep",
                "share_away",
            ]
        )

    pivot = (
        hourly_states.pivot_table(
            index=["respondent_id", "hour"],
            columns="state",
            values="minutes",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in ["away", "home_awake", "sleep"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["minutes_total"] = pivot[["away", "home_awake", "sleep"]].sum(axis=1)
    pivot["share_away"] = pivot["away"] / pivot["minutes_total"].replace(0, np.nan)
    pivot["share_home_awake"] = pivot["home_awake"] / pivot["minutes_total"].replace(0, np.nan)
    pivot["share_sleep"] = pivot["sleep"] / pivot["minutes_total"].replace(0, np.nan)
    pivot = pivot.rename(
        columns={
            "away": "minutes_away",
            "home_awake": "minutes_home_awake",
            "sleep": "minutes_sleep",
        }
    )
    return pivot
