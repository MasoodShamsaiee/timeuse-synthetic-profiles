from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from timeuse_synthetic_profiles.aggregation import (
    aggregate_profiles_by_district,
    aggregate_profiles_by_district_from_assignments,
)
from timeuse_synthetic_profiles.archetypes import (
    ArchetypeFit,
    assign_tus_donors_by_archetype,
    assign_tus_donors_by_archetype_draws,
    fit_schedule_archetypes,
    summarize_profile_draw_uncertainty,
    summarize_weekday_weekend_stability,
)
from timeuse_synthetic_profiles.diaries import (
    classify_episode_states,
    expand_episodes_to_hourly_states,
    summarize_respondent_profiles,
)
from timeuse_synthetic_profiles.features import derive_synth_matching_features, derive_tus_matching_features
from timeuse_synthetic_profiles.io import (
    load_harmonized_tus_episodes,
    load_harmonized_tus_respondents,
    load_synthetic_population,
)
from timeuse_synthetic_profiles.matching import assign_tus_donors, attach_profiles_to_synthpop


DEFAULT_PROFILE_COLS = ["share_home_awake", "share_sleep", "share_away", "share_home_total"]


@dataclass(frozen=True)
class TimeUsePaths:
    synthetic_population: Path
    tus_respondents: Path
    tus_episodes: Path
    out_dir: Path


@dataclass(frozen=True)
class TimeUseConfig:
    district_col: str = "area"
    household_id_eval_col: str = "household_id_eval"
    household_id_cols: tuple[str, ...] = ("household_id", "HID")
    day_types: tuple[str, ...] = ("weekday", "weekend")
    random_seed: int = 42
    n_archetypes: int = 8
    n_draws: int = 5
    chunk_size: int = 100_000
    sleep_activity_codes: frozenset[int] = frozenset({101, 102, 103})
    home_activity_codes: frozenset[int] = frozenset()
    home_location_values: frozenset[int] = frozenset({3300})
    location_col: str = "location_code"
    exact_match_cols: tuple[str, ...] = ("sex_std", "age_band", "hhsize_cat", "has_children")
    fallback_match_sets: tuple[tuple[str, ...], ...] = (
        ("sex_std", "age_band", "has_children"),
        ("sex_std", "age_band"),
        ("age_band",),
    )
    base_feature_cols: tuple[str, ...] = ("sex_std", "age_band", "hhsize_cat", "has_children")
    profile_cols: tuple[str, ...] = ("share_home_awake", "share_sleep", "share_away", "share_home_total")


@dataclass(frozen=True)
class PreparedTimeUseData:
    syn_raw: pd.DataFrame
    syn_people_all: pd.DataFrame
    tus_resp_raw: pd.DataFrame
    tus_ep_raw: pd.DataFrame
    selected_das: list[str]
    paths: TimeUsePaths
    config: TimeUseConfig


def project_root(cwd: Path | None = None) -> Path:
    here = (cwd or Path.cwd()).resolve()
    return here.parent if here.name.lower() == "notebooks" else here


def default_timeuse_paths(root: str | Path | None = None) -> TimeUsePaths:
    root = project_root(Path(root)) if root is not None else project_root()
    raw_time_use = root / "data" / "raw" / "time_use"
    proc_time_use = root / "data" / "processed" / "timeuse_profiles"
    proc_syn = root / "data" / "processed" / "synthetic_population"
    return TimeUsePaths(
        synthetic_population=proc_syn / "syn_inds_with_hh_montreal_p24_seed42_all.parquet",
        tus_respondents=raw_time_use / "tus_respondents_harmonized.parquet",
        tus_episodes=raw_time_use / "tus_episodes_harmonized.parquet",
        out_dir=proc_time_use,
    )


def build_household_eval_id(
    syn_raw: pd.DataFrame,
    *,
    district_col: str = "area",
    output_col: str = "household_id_eval",
    household_id_cols: tuple[str, ...] = ("household_id", "HID"),
) -> pd.DataFrame:
    work = syn_raw.copy()
    base_household_col = next((col for col in household_id_cols if col in work.columns), None)
    if base_household_col is not None:
        work[output_col] = work[district_col].astype(str) + "::" + work[base_household_col].astype(str)
    else:
        work[output_col] = work.index.astype(str)
    return work


def _coerce_realized_hhsize_cat(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    if "hhsize_actual" in out.columns:
        out["hhsize_cat"] = (
            pd.to_numeric(out["hhsize_actual"], errors="coerce")
            .fillna(1)
            .clip(lower=1, upper=5)
            .astype(int)
            .astype(str)
        )
    return out


def _select_districts(
    syn_people_all: pd.DataFrame,
    *,
    limit_to_selected_das: bool,
    n_selected_das: int,
    selection_method: str,
    random_seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    work = syn_people_all.copy()
    available_das = sorted(work["district_id"].astype(str).dropna().unique().tolist())
    if not limit_to_selected_das:
        return work, available_das

    if len(available_das) < int(n_selected_das):
        raise ValueError(f"Requested {n_selected_das} DAs but only found {len(available_das)}.")

    if selection_method == "random":
        selected_das = sorted(
            pd.Series(available_das)
            .sample(n=int(n_selected_das), random_state=int(random_seed), replace=False)
            .astype(str)
            .tolist()
        )
    elif selection_method == "top_population":
        selected_das = (
            work.assign(district_id=work["district_id"].astype(str))
            .groupby("district_id")
            .size()
            .sort_values(ascending=False)
            .head(int(n_selected_das))
            .index.astype(str)
            .tolist()
        )
        selected_das = sorted(selected_das)
    else:
        raise ValueError(f"Unsupported selection_method: {selection_method}")

    work = work.loc[work["district_id"].astype(str).isin(selected_das)].copy()
    return work, selected_das


def prepare_timeuse_data(
    *,
    paths: TimeUsePaths | None = None,
    config: TimeUseConfig | None = None,
    limit_to_selected_das: bool = False,
    n_selected_das: int = 50,
    da_selection_method: str = "random",
) -> PreparedTimeUseData:
    paths = paths or default_timeuse_paths()
    config = config or TimeUseConfig()
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "synthetic population": paths.synthetic_population,
        "TUS respondents": paths.tus_respondents,
        "TUS episodes": paths.tus_episodes,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required time-use inputs:\n" + "\n".join(missing))

    syn_raw = load_synthetic_population(paths.synthetic_population)
    syn_raw = build_household_eval_id(
        syn_raw,
        district_col=config.district_col,
        output_col=config.household_id_eval_col,
        household_id_cols=config.household_id_cols,
    )
    tus_resp_raw = load_harmonized_tus_respondents(paths.tus_respondents)
    tus_ep_raw = load_harmonized_tus_episodes(paths.tus_episodes)

    syn_people_all = derive_synth_matching_features(
        syn_raw,
        district_col=config.district_col,
        household_id_col=config.household_id_eval_col,
    )
    syn_people_all = _coerce_realized_hhsize_cat(syn_people_all)
    syn_people_all, selected_das = _select_districts(
        syn_people_all,
        limit_to_selected_das=limit_to_selected_das,
        n_selected_das=n_selected_das,
        selection_method=da_selection_method,
        random_seed=config.random_seed,
    )
    return PreparedTimeUseData(
        syn_raw=syn_raw,
        syn_people_all=syn_people_all,
        tus_resp_raw=tus_resp_raw,
        tus_ep_raw=tus_ep_raw,
        selected_das=selected_das,
        paths=paths,
        config=config,
    )


def filter_day_type_inputs(
    tus_resp_raw: pd.DataFrame,
    tus_ep_raw: pd.DataFrame,
    *,
    day_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_resp = tus_resp_raw.copy()
    day_ep = tus_ep_raw.copy()
    if "day_type" in day_resp.columns:
        keep_ids = set(day_resp.loc[day_resp["day_type"].astype(str) == str(day_type), "respondent_id"].astype(str))
        day_resp = day_resp.loc[day_resp["respondent_id"].astype(str).isin(keep_ids)].copy()
        day_ep = day_ep.loc[day_ep["respondent_id"].astype(str).isin(keep_ids)].copy()
    return day_resp, day_ep


def build_respondent_profiles(
    episodes: pd.DataFrame,
    *,
    config: TimeUseConfig | None = None,
) -> pd.DataFrame:
    config = config or TimeUseConfig()
    classified = classify_episode_states(
        episodes,
        home_activity_codes=config.home_activity_codes,
        sleep_activity_codes=config.sleep_activity_codes,
        home_location_values=config.home_location_values,
        activity_col="activity_code",
        location_col=config.location_col if config.location_col in episodes.columns else None,
    )
    hourly_states = expand_episodes_to_hourly_states(classified)
    respondent_profiles = summarize_respondent_profiles(hourly_states)
    respondent_profiles["respondent_id"] = respondent_profiles["respondent_id"].astype(str)
    respondent_profiles["share_home_total"] = (
        pd.to_numeric(respondent_profiles["share_home_awake"], errors="coerce").fillna(0.0)
        + pd.to_numeric(respondent_profiles["share_sleep"], errors="coerce").fillna(0.0)
    )
    return respondent_profiles


def prepare_matching_inputs(
    prepared: PreparedTimeUseData,
    *,
    day_type: str,
) -> dict[str, Any]:
    day_resp, day_ep = filter_day_type_inputs(prepared.tus_resp_raw, prepared.tus_ep_raw, day_type=day_type)
    tus_people = derive_tus_matching_features(day_resp)
    tus_people["respondent_id"] = tus_people["respondent_id"].astype(str)
    respondent_profiles = build_respondent_profiles(day_ep, config=prepared.config)

    tus_supported_age_bands = sorted(tus_people["age_band"].astype(str).dropna().unique().tolist())
    syn_people = prepared.syn_people_all.loc[
        prepared.syn_people_all["age_band"].astype(str).isin(tus_supported_age_bands)
    ].copy()
    syn_people["person_id"] = syn_people["person_id"].astype(str)
    syn_people["district_id"] = syn_people["district_id"].astype(str)

    exact_match_cols = [
        col for col in prepared.config.exact_match_cols if col in syn_people.columns and col in tus_people.columns
    ]
    fallback_match_sets = [
        [col for col in cols if col in syn_people.columns and col in tus_people.columns]
        for cols in prepared.config.fallback_match_sets
    ]
    fallback_match_sets = [cols for cols in fallback_match_sets if cols]

    return {
        "day_type": day_type,
        "syn_people": syn_people,
        "tus_people": tus_people,
        "respondent_profiles": respondent_profiles,
        "exact_match_cols": exact_match_cols,
        "fallback_match_sets": fallback_match_sets,
    }


def run_baseline_donor_workflow(
    prepared: PreparedTimeUseData,
    *,
    day_type: str,
) -> dict[str, pd.DataFrame]:
    inputs = prepare_matching_inputs(prepared, day_type=day_type)
    donor_assignments = assign_tus_donors(
        inputs["syn_people"],
        inputs["tus_people"],
        exact_match_cols=inputs["exact_match_cols"],
        fallback_match_sets=inputs["fallback_match_sets"],
        weight_col="person_weight",
        random_seed=prepared.config.random_seed,
    )
    assigned_profiles = attach_profiles_to_synthpop(
        inputs["syn_people"],
        donor_assignments,
        inputs["respondent_profiles"],
    )
    district_profiles = aggregate_profiles_by_district(assigned_profiles, district_col="district_id")
    return {
        **inputs,
        "donor_assignments": donor_assignments,
        "assigned_profiles": assigned_profiles,
        "district_profiles": district_profiles,
    }


def run_archetype_assignment_workflow(
    prepared: PreparedTimeUseData,
    *,
    day_type: str,
) -> dict[str, pd.DataFrame]:
    inputs = prepare_matching_inputs(prepared, day_type=day_type)
    fit = fit_schedule_archetypes(
        inputs["respondent_profiles"],
        n_archetypes=prepared.config.n_archetypes,
        random_seed=prepared.config.random_seed,
    )
    donor_draws = assign_tus_donors_by_archetype_draws(
        inputs["syn_people"],
        inputs["tus_people"],
        fit.respondent_archetypes,
        exact_match_cols=inputs["exact_match_cols"],
        fallback_match_sets=inputs["fallback_match_sets"],
        weight_col="person_weight",
        random_seed=prepared.config.random_seed,
        n_draws=prepared.config.n_draws,
    )
    donor_draws["day_type"] = day_type

    district_draws = aggregate_profiles_by_district_from_assignments(
        inputs["syn_people"],
        donor_draws,
        inputs["respondent_profiles"],
        district_col="district_id",
        extra_group_cols=["draw_id", "day_type"],
    )
    reference_assignments = donor_draws.loc[pd.to_numeric(donor_draws["draw_id"], errors="coerce") == 0].copy()
    reference_district_profiles = district_draws.loc[
        pd.to_numeric(district_draws["draw_id"], errors="coerce") == 0
    ].copy()

    donor_draw_summary = (
        donor_draws.groupby(["day_type", "draw_id", "match_level"], as_index=False)
        .size()
        .rename(columns={"size": "n_people"})
    )
    hourly_uncertainty = summarize_profile_draw_uncertainty(
        district_draws.groupby(["day_type", "draw_id", "hour"], as_index=False)[list(prepared.config.profile_cols)].mean(),
        group_cols=["day_type", "hour"],
        draw_col="draw_id",
        value_cols=list(prepared.config.profile_cols),
    )
    district_uncertainty = summarize_profile_draw_uncertainty(
        district_draws,
        group_cols=["day_type", "district_id", "hour"],
        draw_col="draw_id",
        value_cols=list(prepared.config.profile_cols),
    )

    fit_summary = pd.DataFrame([{"day_type": day_type, **fit.fit_summary}])
    respondent_archetypes = fit.respondent_archetypes.copy()
    respondent_archetypes["day_type"] = day_type
    centroid_profiles = fit.centroid_profiles.copy()
    centroid_profiles["day_type"] = day_type

    return {
        **inputs,
        "reference_assignments": reference_assignments,
        "reference_district_profiles": reference_district_profiles,
        "donor_draws": donor_draws,
        "district_draws": district_draws,
        "donor_draw_summary": donor_draw_summary,
        "hourly_uncertainty": hourly_uncertainty,
        "district_uncertainty": district_uncertainty,
        "respondent_archetypes": respondent_archetypes,
        "centroid_profiles": centroid_profiles,
        "fit_summary": fit_summary,
        "fit": fit,
    }


def fit_archetype_classifier(
    tus_labeled: pd.DataFrame,
    *,
    feature_cols: list[str],
    random_seed: int = 42,
) -> tuple[Pipeline, pd.DataFrame, np.ndarray]:
    X = tus_labeled[feature_cols].copy()
    y = tus_labeled["archetype_id"].astype(int)
    sample_weight = pd.to_numeric(tus_labeled.get("person_weight"), errors="coerce").fillna(1.0).to_numpy(dtype=float)

    stratify_y = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        sample_weight,
        test_size=0.25,
        random_state=int(random_seed),
        stratify=stratify_y,
    )

    preprocessor = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), feature_cols)])
    classifier = LogisticRegression(max_iter=500)
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    model.fit(X_train, y_train, classifier__sample_weight=w_train)

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)
    classes = model.named_steps["classifier"].classes_
    metrics = pd.DataFrame(
        [
            {
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "n_classes": int(len(classes)),
                "accuracy": float(accuracy_score(y_test, test_pred, sample_weight=w_test)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
                "log_loss": float(log_loss(y_test, test_prob, sample_weight=w_test, labels=classes)),
            }
        ]
    )
    return model, metrics, classes


def predict_expected_district_archetype_counts(
    syn_people: pd.DataFrame,
    model: Pipeline,
    *,
    feature_cols: list[str],
    classes: np.ndarray,
    chunk_size: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    district_tables: list[pd.DataFrame] = []
    for start in range(0, len(syn_people), int(chunk_size)):
        chunk = syn_people.iloc[start:start + int(chunk_size)].copy()
        probs = model.predict_proba(chunk[feature_cols])
        prob_df = pd.DataFrame(probs, columns=[int(c) for c in classes], index=chunk.index)
        prob_df.insert(0, "district_id", chunk["district_id"].astype(str).values)
        district_tables.append(prob_df.groupby("district_id", as_index=False).sum())

    district_counts = pd.concat(district_tables, ignore_index=True)
    district_counts = district_counts.groupby("district_id", as_index=False).sum()
    long = district_counts.melt(id_vars="district_id", var_name="archetype_id", value_name="expected_n_people")
    long["archetype_id"] = pd.to_numeric(long["archetype_id"], errors="coerce").astype(int)
    return district_counts, long


def district_profiles_from_expected_counts(
    district_archetype_counts: pd.DataFrame,
    centroid_profiles: pd.DataFrame,
    *,
    profile_cols: list[str] | None = None,
) -> pd.DataFrame:
    profile_cols = profile_cols or DEFAULT_PROFILE_COLS
    centroids = centroid_profiles.copy()
    if "share_home_total" in profile_cols and "share_home_total" not in centroids.columns:
        centroids["share_home_total"] = (
            pd.to_numeric(centroids.get("share_home_awake"), errors="coerce").fillna(0.0)
            + pd.to_numeric(centroids.get("share_sleep"), errors="coerce").fillna(0.0)
        )
    centroids = centroids[["archetype_id", "hour", *profile_cols]].copy()
    centroids["archetype_id"] = pd.to_numeric(centroids["archetype_id"], errors="coerce").astype(int)
    work = district_archetype_counts.copy()
    work["archetype_id"] = pd.to_numeric(work["archetype_id"], errors="coerce").astype(int)
    work["expected_n_people"] = pd.to_numeric(work["expected_n_people"], errors="coerce").fillna(0.0)

    merged = work.merge(centroids, on="archetype_id", how="inner", validate="m:m")
    for col in profile_cols:
        merged[f"{col}_weighted"] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0) * merged["expected_n_people"]

    district_profiles = (
        merged.groupby(["district_id", "hour"], as_index=False)[
            ["expected_n_people", *[f"{col}_weighted" for col in profile_cols]]
        ]
        .sum()
        .sort_values(["district_id", "hour"])
        .reset_index(drop=True)
    )
    denom = district_profiles["expected_n_people"].replace(0, np.nan)
    for col in profile_cols:
        district_profiles[col] = district_profiles[f"{col}_weighted"] / denom
    return district_profiles.drop(columns=[f"{col}_weighted" for col in profile_cols])


def run_archetype_classifier_workflow(
    prepared: PreparedTimeUseData,
    *,
    day_type: str,
) -> dict[str, Any]:
    inputs = prepare_matching_inputs(prepared, day_type=day_type)
    fit: ArchetypeFit = fit_schedule_archetypes(
        inputs["respondent_profiles"],
        n_archetypes=prepared.config.n_archetypes,
        random_seed=prepared.config.random_seed,
    )
    tus_labeled = inputs["tus_people"].merge(
        fit.respondent_archetypes,
        on="respondent_id",
        how="inner",
        validate="1:1",
    )
    feature_cols = [
        col for col in prepared.config.base_feature_cols if col in tus_labeled.columns and col in inputs["syn_people"].columns
    ]
    if not feature_cols:
        raise ValueError("No common feature columns were available for archetype-classifier training.")
    for col in feature_cols:
        tus_labeled[col] = tus_labeled[col].astype(str).fillna("missing")
        inputs["syn_people"][col] = inputs["syn_people"][col].astype(str).fillna("missing")

    model, metrics, classes = fit_archetype_classifier(
        tus_labeled,
        feature_cols=feature_cols,
        random_seed=prepared.config.random_seed,
    )
    district_archetype_counts, district_archetype_long = predict_expected_district_archetype_counts(
        inputs["syn_people"],
        model,
        feature_cols=feature_cols,
        classes=classes,
        chunk_size=prepared.config.chunk_size,
    )
    district_profiles = district_profiles_from_expected_counts(
        district_archetype_long,
        fit.centroid_profiles,
        profile_cols=list(prepared.config.profile_cols),
    )

    metrics = metrics.copy()
    metrics["day_type"] = day_type
    metrics["n_synthetic_people"] = int(len(inputs["syn_people"]))
    metrics["n_tus_respondents"] = int(len(tus_labeled))
    metrics["n_archetypes"] = int(len(classes))

    centroid_profiles = fit.centroid_profiles.copy()
    centroid_profiles["day_type"] = day_type
    district_archetype_counts["day_type"] = day_type
    district_profiles["day_type"] = day_type

    return {
        **inputs,
        "tus_labeled": tus_labeled,
        "feature_cols": feature_cols,
        "model": model,
        "metrics": metrics,
        "classes": classes,
        "district_archetype_counts": district_archetype_counts,
        "district_archetype_long": district_archetype_long,
        "district_profiles": district_profiles,
        "centroid_profiles": centroid_profiles,
        "respondent_archetypes": fit.respondent_archetypes,
        "fit": fit,
    }


def run_multiday_classifier_workflow(
    prepared: PreparedTimeUseData,
    *,
    day_types: list[str] | None = None,
) -> dict[str, Any]:
    day_types = day_types or list(prepared.config.day_types)
    results_by_day: dict[str, Any] = {}
    metrics_frames: list[pd.DataFrame] = []
    district_profile_frames: list[pd.DataFrame] = []
    district_archetype_frames: list[pd.DataFrame] = []
    centroid_frames: list[pd.DataFrame] = []

    for day_type in day_types:
        result = run_archetype_classifier_workflow(prepared, day_type=day_type)
        results_by_day[day_type] = result
        metrics_frames.append(result["metrics"])
        district_profile_frames.append(result["district_profiles"])
        district_archetype_frames.append(result["district_archetype_counts"])
        centroid_frames.append(result["centroid_profiles"])

    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    all_district_profiles = pd.concat(district_profile_frames, ignore_index=True)
    all_district_archetypes = pd.concat(district_archetype_frames, ignore_index=True)
    all_centroids = pd.concat(centroid_frames, ignore_index=True)

    stability = pd.DataFrame()
    stability_summary = pd.DataFrame()
    if len(day_types) >= 2 and "weekday" in results_by_day and "weekend" in results_by_day:
        weekday_profiles = results_by_day["weekday"]["district_profiles"][["district_id", "hour", *prepared.config.profile_cols]].copy()
        weekend_profiles = results_by_day["weekend"]["district_profiles"][["district_id", "hour", *prepared.config.profile_cols]].copy()
        stability = summarize_weekday_weekend_stability(
            weekday_profiles,
            weekend_profiles,
            group_cols=["district_id", "hour"],
            value_cols=list(prepared.config.profile_cols),
        )
        stability_summary = pd.DataFrame(
            [
                {
                    "metric": col,
                    "mean_abs_delta": float(pd.to_numeric(stability[f"{col}_abs_delta"], errors="coerce").mean()),
                    "p95_abs_delta": float(pd.to_numeric(stability[f"{col}_abs_delta"], errors="coerce").quantile(0.95)),
                }
                for col in prepared.config.profile_cols
            ]
        )

    return {
        "results_by_day": results_by_day,
        "all_metrics": all_metrics,
        "all_district_profiles": all_district_profiles,
        "all_district_archetypes": all_district_archetypes,
        "all_centroids": all_centroids,
        "stability": stability,
        "stability_summary": stability_summary,
    }
