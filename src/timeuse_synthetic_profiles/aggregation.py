from __future__ import annotations

import pandas as pd


def aggregate_profiles_by_district(
    assigned_profiles: pd.DataFrame,
    *,
    district_col: str = "district_id",
    hour_col: str = "hour",
) -> pd.DataFrame:
    required = [district_col, hour_col, "share_home_awake", "share_sleep", "share_away"]
    missing = [c for c in required if c not in assigned_profiles.columns]
    if missing:
        raise KeyError(f"Assigned profiles are missing required columns: {missing}")

    work = assigned_profiles.copy()
    group_cols = [district_col, hour_col]
    district_hourly = (
        work.groupby(group_cols, as_index=False)[["share_home_awake", "share_sleep", "share_away"]]
        .mean()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    district_hourly["share_home_total"] = district_hourly["share_home_awake"] + district_hourly["share_sleep"]
    return district_hourly


def aggregate_profiles_by_district_from_assignments(
    synth_people: pd.DataFrame,
    donor_assignments: pd.DataFrame,
    respondent_profiles: pd.DataFrame,
    *,
    district_col: str = "district_id",
    hour_col: str = "hour",
    person_id_col: str = "person_id",
    respondent_id_col: str = "respondent_id",
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_group_cols = [] if extra_group_cols is None else list(extra_group_cols)

    missing_synth = [c for c in [person_id_col, district_col] if c not in synth_people.columns]
    if missing_synth:
        raise KeyError(f"Synthetic people are missing required columns: {missing_synth}")

    missing_assign = [c for c in [person_id_col, respondent_id_col] + extra_group_cols if c not in donor_assignments.columns]
    if missing_assign:
        raise KeyError(f"Donor assignments are missing required columns: {missing_assign}")

    required_profile_cols = [respondent_id_col, hour_col, "share_home_awake", "share_sleep", "share_away"]
    missing_profiles = [c for c in required_profile_cols if c not in respondent_profiles.columns]
    if missing_profiles:
        raise KeyError(f"Respondent profiles are missing required columns: {missing_profiles}")

    synth_base = synth_people[[person_id_col, district_col]].copy()
    synth_base[person_id_col] = synth_base[person_id_col].astype(str)

    assignment_cols = [person_id_col, respondent_id_col, *extra_group_cols]
    assigned = synth_base.merge(
        donor_assignments[assignment_cols].copy(),
        on=person_id_col,
        how="left",
        validate="1:1",
    )
    assigned[respondent_id_col] = assigned[respondent_id_col].astype(str)

    count_group_cols = [district_col, respondent_id_col, *extra_group_cols]
    assigned_counts = (
        assigned.groupby(count_group_cols, as_index=False)
        .size()
        .rename(columns={"size": "n_people"})
    )

    profile_cols = [respondent_id_col, hour_col, "share_home_awake", "share_sleep", "share_away"]
    profile_base = respondent_profiles[profile_cols].copy()
    profile_base[respondent_id_col] = profile_base[respondent_id_col].astype(str)

    weighted = assigned_counts.merge(
        profile_base,
        on=respondent_id_col,
        how="left",
        validate="m:m",
    )

    for col in ["share_home_awake", "share_sleep", "share_away"]:
        weighted[col] = pd.to_numeric(weighted[col], errors="coerce")
        weighted[f"{col}_weighted"] = weighted[col] * pd.to_numeric(weighted["n_people"], errors="coerce").fillna(0)

    agg_group_cols = [district_col, hour_col, *extra_group_cols]
    district_hourly = (
        weighted.groupby(agg_group_cols, as_index=False)[
            ["n_people", "share_home_awake_weighted", "share_sleep_weighted", "share_away_weighted"]
        ]
        .sum()
        .sort_values(agg_group_cols)
        .reset_index(drop=True)
    )

    denom = pd.to_numeric(district_hourly["n_people"], errors="coerce").replace(0, pd.NA)
    district_hourly["share_home_awake"] = district_hourly["share_home_awake_weighted"] / denom
    district_hourly["share_sleep"] = district_hourly["share_sleep_weighted"] / denom
    district_hourly["share_away"] = district_hourly["share_away_weighted"] / denom
    district_hourly["share_home_total"] = district_hourly["share_home_awake"] + district_hourly["share_sleep"]
    return district_hourly.drop(
        columns=[
            "share_home_awake_weighted",
            "share_sleep_weighted",
            "share_away_weighted",
        ]
    )
