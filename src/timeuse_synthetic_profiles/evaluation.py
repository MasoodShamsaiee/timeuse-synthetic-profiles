from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_PROFILE_COLS = ["share_home_awake", "share_sleep", "share_away", "share_home_total"]
DEFAULT_SUBGROUP_COLS = ["age_band", "sex_std", "hhsize_cat", "has_children"]


def compare_distribution(
    df_left: pd.DataFrame,
    left_col: str,
    df_right: pd.DataFrame,
    right_col: str,
    *,
    left_name: str = "synthetic",
    right_name: str = "tus",
    right_weight_col: str | None = None,
) -> pd.DataFrame:
    left = df_left[left_col].astype(str).value_counts(normalize=True).rename(left_name)
    if right_weight_col is not None and right_weight_col in df_right.columns:
        weights = pd.to_numeric(df_right[right_weight_col], errors="coerce").fillna(0)
        right = (
            df_right.assign(_weight=weights, _category=df_right[right_col].astype(str))
            .groupby("_category")["_weight"]
            .sum()
        )
        right = (right / right.sum()).rename(right_name) if right.sum() > 0 else right.astype(float).rename(right_name)
    else:
        right = df_right[right_col].astype(str).value_counts(normalize=True).rename(right_name)
    out = pd.concat([left, right], axis=1).fillna(0).rename_axis("category").reset_index()
    out["abs_diff"] = (out[left_name] - out[right_name]).abs()
    return out.sort_values("category").reset_index(drop=True)


def weighted_distribution(df: pd.DataFrame, value_col: str, *, weight_col: str | None = None) -> pd.Series:
    work = df[[value_col]].copy()
    work[value_col] = work[value_col].astype(str).fillna("missing")
    if weight_col is None or weight_col not in df.columns:
        counts = work[value_col].value_counts(dropna=False).sort_index()
    else:
        weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
        counts = work.assign(_weight=weights).groupby(value_col)["_weight"].sum().sort_index()
    return counts / counts.sum() if counts.sum() else counts.astype(float)


def _normalize_prob(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    total = arr.sum()
    return np.zeros_like(arr, dtype=float) if total <= 0 else arr / total


def js_divergence(p: np.ndarray | pd.Series | list[float], q: np.ndarray | pd.Series | list[float], *, eps: float = 1e-12) -> float:
    p_norm = _normalize_prob(p)
    q_norm = _normalize_prob(q)
    m = 0.5 * (p_norm + q_norm)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * _kl(p_norm, m) + 0.5 * _kl(q_norm, m)


def weighted_hourly_mean(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
    *,
    weight_col: str | None = None,
) -> pd.DataFrame:
    work = df[[group_col, "hour", *value_cols]].copy()
    work[group_col] = work[group_col].astype(str)
    for col in value_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if weight_col is None or weight_col not in df.columns:
        return work.groupby([group_col, "hour"], as_index=False)[value_cols].mean()

    work[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0).values
    rows: list[dict] = []
    for (group_value, hour), sub in work.groupby([group_col, "hour"], dropna=False):
        weights = sub[weight_col].to_numpy(dtype=float)
        means = (
            sub[value_cols].mean()
            if weights.sum() <= 0
            else pd.Series({col: np.average(sub[col].fillna(0).to_numpy(dtype=float), weights=weights) for col in value_cols})
        )
        row = {group_col: group_value, "hour": hour}
        row.update(means.to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values([group_col, "hour"]).reset_index(drop=True)


def profile_error_summary(
    synth_hourly: pd.DataFrame,
    tus_hourly: pd.DataFrame,
    *,
    group_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    merged = synth_hourly.merge(
        tus_hourly,
        on=[group_col, "hour"],
        how="inner",
        suffixes=("_synth", "_tus"),
    )
    rows: list[dict] = []
    for group_value, sub in merged.groupby(group_col, dropna=False):
        row = {"group_col": group_col, "group_value": group_value, "n_hours": len(sub)}
        for col in value_cols:
            diff = pd.to_numeric(sub[f"{col}_synth"], errors="coerce") - pd.to_numeric(sub[f"{col}_tus"], errors="coerce")
            row[f"{col}_mae"] = float(diff.abs().mean())
            row[f"{col}_rmse"] = float(np.sqrt((diff**2).mean()))
            row[f"{col}_bias"] = float(diff.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group_col", "group_value"]).reset_index(drop=True)


def build_match_summary(assigned_eval: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "n_synthetic_people",
                "n_unique_tus_donors_used",
                "share_exact_matches",
                "share_fallback_matches",
                "share_global_fallback",
            ],
            "value": [
                len(assigned_eval),
                assigned_eval["respondent_id"].astype(str).nunique(),
                float((assigned_eval["match_level"] == 0).mean()),
                float((assigned_eval["match_level"] > 0).mean()),
                float((assigned_eval["match_level"] == 999).mean()),
            ],
        }
    )


def build_match_level_summary(assigned_eval: pd.DataFrame) -> pd.DataFrame:
    return assigned_eval["match_level"].value_counts(dropna=False).rename_axis("match_level").reset_index(name="n_people")


def build_household_metrics(
    syn_raw: pd.DataFrame,
    *,
    household_id_col: str,
    area_col: str = "area",
    person_id_col: str = "person_id",
    age_col: str = "age",
) -> pd.DataFrame:
    hh_df = syn_raw.copy()
    hh_df[household_id_col] = hh_df[household_id_col].astype(str)
    hh_df["age_num"] = pd.to_numeric(hh_df.get(age_col), errors="coerce")
    household_metrics = hh_df.groupby(household_id_col).agg(
        n_people=(person_id_col, "size"),
        n_children=("age_num", lambda s: int((pd.to_numeric(s, errors="coerce") < 18).sum())),
        n_seniors=("age_num", lambda s: int((pd.to_numeric(s, errors="coerce") >= 65).sum())),
        area=(area_col, "first"),
    ).reset_index()
    household_metrics["has_children"] = (household_metrics["n_children"] > 0).astype(int)
    household_metrics["has_seniors"] = (household_metrics["n_seniors"] > 0).astype(int)
    return household_metrics


def build_household_summary(household_metrics: pd.DataFrame, *, household_id_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "n_households",
                "mean_household_size",
                "share_households_with_children",
                "share_households_with_seniors",
            ],
            "value": [
                household_metrics[household_id_col].nunique(),
                household_metrics["n_people"].mean(),
                household_metrics["has_children"].mean(),
                household_metrics["has_seniors"].mean(),
            ],
        }
    )


def build_district_peak_home(district_profiles: pd.DataFrame) -> pd.DataFrame:
    work = district_profiles.copy()
    work["district_id"] = work["district_id"].astype(str)
    peak_idx = work.groupby("district_id")["share_home_total"].idxmax()
    return work.loc[peak_idx, ["district_id", "hour", "share_home_total"]].rename(
        columns={"hour": "peak_home_hour", "share_home_total": "peak_home_share"}
    )


def build_mean_profiles(district_profiles: pd.DataFrame, *, profile_cols: list[str] | None = None) -> pd.DataFrame:
    profile_cols = profile_cols or DEFAULT_PROFILE_COLS
    return district_profiles.groupby("hour", as_index=False)[profile_cols].mean()


def build_district_summary(district_profiles: pd.DataFrame) -> pd.DataFrame:
    return district_profiles.groupby("district_id").agg(
        mean_home_total=("share_home_total", "mean"),
        max_home_total=("share_home_total", "max"),
        mean_away=("share_away", "mean"),
        std_home_total=("share_home_total", "std"),
    ).reset_index()


def build_subgroup_jsd_summary(
    assigned_eval: pd.DataFrame,
    tus_eval: pd.DataFrame,
    *,
    subgroup_cols: list[str] | None = None,
) -> pd.DataFrame:
    subgroup_cols = subgroup_cols or DEFAULT_SUBGROUP_COLS
    rows: list[dict] = []
    for col in subgroup_cols:
        synth_dist = weighted_distribution(assigned_eval, col)
        tus_dist = weighted_distribution(tus_eval, col, weight_col="person_weight")
        categories = sorted(set(synth_dist.index).union(set(tus_dist.index)))
        p = synth_dist.reindex(categories, fill_value=0).to_numpy(dtype=float)
        q = tus_dist.reindex(categories, fill_value=0).to_numpy(dtype=float)
        p_norm = _normalize_prob(p)
        q_norm = _normalize_prob(q)
        rows.append(
            {
                "group_col": col,
                "n_categories": len(categories),
                "js_divergence": js_divergence(p_norm, q_norm),
                "total_variation": float(0.5 * np.abs(p_norm - q_norm).sum()),
                "max_abs_share_diff": float(np.abs(p_norm - q_norm).max()),
            }
        )
    return pd.DataFrame(rows).sort_values("js_divergence", ascending=False).reset_index(drop=True)


def build_subgroup_hourly_profile_errors(
    assigned_eval: pd.DataFrame,
    tus_eval: pd.DataFrame,
    respondent_profiles: pd.DataFrame,
    *,
    subgroup_cols: list[str] | None = None,
    profile_cols: list[str] | None = None,
) -> pd.DataFrame:
    subgroup_cols = subgroup_cols or DEFAULT_SUBGROUP_COLS
    profile_cols = profile_cols or DEFAULT_PROFILE_COLS

    assigned_hourly = assigned_eval[["person_id", "respondent_id", *subgroup_cols]].merge(
        respondent_profiles,
        on="respondent_id",
        how="left",
    )
    tus_hourly = tus_eval[["respondent_id", "person_weight", *subgroup_cols]].merge(
        respondent_profiles,
        on="respondent_id",
        how="left",
    )

    frames: list[pd.DataFrame] = []
    for col in subgroup_cols:
        synth_hourly = weighted_hourly_mean(assigned_hourly, col, profile_cols)
        tus_hourly_ref = weighted_hourly_mean(tus_hourly, col, profile_cols, weight_col="person_weight")
        frames.append(profile_error_summary(synth_hourly, tus_hourly_ref, group_col=col, value_cols=profile_cols))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_weak_group_summary(
    assigned_eval: pd.DataFrame,
    tus_eval: pd.DataFrame,
    *,
    weak_age_groups: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for age_band in weak_age_groups:
        synth_sub = assigned_eval.loc[assigned_eval["age_band"].astype(str) == str(age_band)].copy()
        tus_sub = tus_eval.loc[tus_eval["age_band"].astype(str) == str(age_band)].copy()
        rows.append(
            {
                "age_band": age_band,
                "n_synthetic_people": len(synth_sub),
                "n_tus_respondents": len(tus_sub),
                "n_unique_donors_used": synth_sub["respondent_id"].astype(str).nunique(),
                "share_exact_matches": float((synth_sub["match_level"] == 0).mean()) if len(synth_sub) else np.nan,
                "share_fallback_matches": float((synth_sub["match_level"] > 0).mean()) if len(synth_sub) else np.nan,
                "share_global_fallback": float((synth_sub["match_level"] == 999).mean()) if len(synth_sub) else np.nan,
                "donor_reuse_ratio": float(len(synth_sub) / max(synth_sub["respondent_id"].astype(str).nunique(), 1)) if len(synth_sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class EvaluationBundle:
    match_summary: pd.DataFrame
    match_level_summary: pd.DataFrame
    household_metrics: pd.DataFrame
    household_summary: pd.DataFrame
    district_peak_home: pd.DataFrame
    district_mean_profiles: pd.DataFrame
    district_summary: pd.DataFrame
    subgroup_jsd_summary: pd.DataFrame
    subgroup_hourly_profile_errors: pd.DataFrame


def build_evaluation_bundle(
    *,
    syn_raw: pd.DataFrame,
    assigned_eval: pd.DataFrame,
    tus_eval: pd.DataFrame,
    district_profiles: pd.DataFrame,
    respondent_profiles: pd.DataFrame,
    household_id_col: str,
    subgroup_cols: list[str] | None = None,
    profile_cols: list[str] | None = None,
) -> EvaluationBundle:
    profile_cols = profile_cols or DEFAULT_PROFILE_COLS
    household_metrics = build_household_metrics(syn_raw, household_id_col=household_id_col)
    return EvaluationBundle(
        match_summary=build_match_summary(assigned_eval),
        match_level_summary=build_match_level_summary(assigned_eval),
        household_metrics=household_metrics,
        household_summary=build_household_summary(household_metrics, household_id_col=household_id_col),
        district_peak_home=build_district_peak_home(district_profiles),
        district_mean_profiles=build_mean_profiles(district_profiles, profile_cols=profile_cols),
        district_summary=build_district_summary(district_profiles),
        subgroup_jsd_summary=build_subgroup_jsd_summary(assigned_eval, tus_eval, subgroup_cols=subgroup_cols),
        subgroup_hourly_profile_errors=build_subgroup_hourly_profile_errors(
            assigned_eval,
            tus_eval,
            respondent_profiles,
            subgroup_cols=subgroup_cols,
            profile_cols=profile_cols,
        ),
    )
