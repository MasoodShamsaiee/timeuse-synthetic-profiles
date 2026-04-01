from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at import time
    tqdm = None


DEFAULT_PROFILE_COLS = ("share_home_awake", "share_sleep", "share_away")


@dataclass(frozen=True)
class ArchetypeFit:
    respondent_archetypes: pd.DataFrame
    centroid_profiles: pd.DataFrame
    fit_summary: dict


def _ensure_profile_columns(df: pd.DataFrame, profile_cols: tuple[str, ...]) -> pd.DataFrame:
    work = df.copy()
    for col in profile_cols:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    return work


def _build_profile_matrix(
    respondent_profiles: pd.DataFrame,
    *,
    profile_cols: tuple[str, ...] = DEFAULT_PROFILE_COLS,
) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"respondent_id", "hour"}
    missing = required.difference(respondent_profiles.columns)
    if missing:
        raise KeyError(f"Respondent profiles are missing required columns: {sorted(missing)}")

    work = _ensure_profile_columns(respondent_profiles, profile_cols)
    work["respondent_id"] = work["respondent_id"].astype(str)
    work["hour"] = pd.to_numeric(work["hour"], errors="coerce").fillna(0).astype(int) % 24

    full_index = pd.MultiIndex.from_product(
        [sorted(work["respondent_id"].unique()), list(range(24))],
        names=["respondent_id", "hour"],
    )
    wide = (
        work.set_index(["respondent_id", "hour"])[list(profile_cols)]
        .reindex(full_index, fill_value=0.0)
        .unstack("hour")
    )
    wide.columns = [f"{col}_h{int(hour):02d}" for col, hour in wide.columns]
    matrix = wide.to_numpy(dtype=float)
    return wide.reset_index(), matrix


def _run_kmeans(
    X: np.ndarray,
    k: int,
    *,
    n_init: int = 10,
    max_iter: int = 100,
    random_seed: int = 42,
) -> dict:
    if len(X) == 0:
        raise ValueError("Cannot cluster an empty matrix.")
    if k < 1 or k > len(X):
        raise ValueError("k must be between 1 and the number of rows in X.")

    rng = np.random.default_rng(int(random_seed))
    best: dict | None = None
    for _ in range(int(n_init)):
        centers = X[rng.choice(len(X), size=k, replace=False)].copy()
        for _ in range(int(max_iter)):
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster_id in range(k):
                mask = labels == cluster_id
                if mask.any():
                    new_centers[cluster_id] = X[mask].mean(axis=0)
                else:
                    new_centers[cluster_id] = X[rng.integers(len(X))]
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best["inertia"]:
            best = {"labels": labels.copy(), "centers": centers.copy(), "inertia": inertia}
    assert best is not None
    return best


def fit_schedule_archetypes(
    respondent_profiles: pd.DataFrame,
    *,
    n_archetypes: int = 8,
    profile_cols: tuple[str, ...] = DEFAULT_PROFILE_COLS,
    random_seed: int = 42,
    n_init: int = 10,
    max_iter: int = 100,
) -> ArchetypeFit:
    wide, matrix = _build_profile_matrix(respondent_profiles, profile_cols=profile_cols)
    k = max(1, min(int(n_archetypes), len(wide)))
    result = _run_kmeans(matrix, k, n_init=n_init, max_iter=max_iter, random_seed=random_seed)

    respondent_archetypes = wide[["respondent_id"]].copy()
    respondent_archetypes["archetype_id"] = result["labels"].astype(int)

    center_rows: list[dict] = []
    for archetype_id, center in enumerate(result["centers"]):
        offset = 0
        for col in profile_cols:
            for hour in range(24):
                if offset >= len(center):
                    break
                center_rows.append(
                    {
                        "archetype_id": int(archetype_id),
                        "hour": int(hour),
                        "state": col,
                        "share": float(center[offset]),
                    }
                )
                offset += 1
    centroid_long = pd.DataFrame(center_rows)
    centroid_profiles = (
        centroid_long.pivot_table(index=["archetype_id", "hour"], columns="state", values="share", fill_value=0.0)
        .reset_index()
        .rename_axis(None, axis=1)
    )

    fit_summary = {
        "n_respondents": int(len(wide)),
        "n_archetypes": int(k),
        "inertia": float(result["inertia"]),
    }
    return ArchetypeFit(
        respondent_archetypes=respondent_archetypes,
        centroid_profiles=centroid_profiles,
        fit_summary=fit_summary,
    )


def _sample_weighted(values: list[str], weights: list[float], *, rng: random.Random) -> str:
    if not values:
        raise ValueError("Cannot sample from an empty list.")
    if not weights or float(sum(weights)) <= 0:
        return str(values[rng.randrange(len(values))])
    return str(rng.choices(values, weights=weights, k=1)[0])


def _sample_weighted_respondent(pool: pd.DataFrame, *, weight_col: str, rng: random.Random) -> str:
    if pool.empty:
        raise ValueError("Cannot sample from an empty donor pool.")
    weights = pd.to_numeric(pool[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return _sample_weighted(
        pool["respondent_id"].astype(str).tolist(),
        weights.tolist(),
        rng=rng,
    )


def assign_tus_donors_by_archetype(
    synth_people: pd.DataFrame,
    tus_people: pd.DataFrame,
    respondent_archetypes: pd.DataFrame,
    *,
    exact_match_cols: list[str],
    fallback_match_sets: list[list[str]] | None = None,
    weight_col: str = "person_weight",
    random_seed: int = 42,
    archetype_col: str = "archetype_id",
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_position: int | None = None,
) -> pd.DataFrame:
    if "respondent_id" not in tus_people.columns:
        raise KeyError("Expected 'respondent_id' column in TUS respondents.")
    if "person_id" not in synth_people.columns:
        raise KeyError("Expected 'person_id' column in synthetic people.")
    if archetype_col not in respondent_archetypes.columns:
        raise KeyError(f"Expected '{archetype_col}' in respondent_archetypes.")

    fallback_match_sets = [] if fallback_match_sets is None else fallback_match_sets
    match_sets = [list(exact_match_cols)] + [list(cols) for cols in fallback_match_sets]
    rng = random.Random(int(random_seed))

    tus_with_archetypes = tus_people.merge(
        respondent_archetypes[["respondent_id", archetype_col]],
        on="respondent_id",
        how="inner",
        validate="1:1",
    )
    tus_with_archetypes["respondent_id"] = tus_with_archetypes["respondent_id"].astype(str)
    tus_with_archetypes[archetype_col] = tus_with_archetypes[archetype_col].astype(int)

    records = synth_people.itertuples(index=False)
    if show_progress and tqdm is not None:
        records = tqdm(
            records,
            total=int(len(synth_people)),
            desc=progress_desc or "Assigning donors",
            position=progress_position,
            leave=True,
            dynamic_ncols=True,
        )

    donor_rows: list[dict] = []
    for rec in records:
        rec_series = pd.Series(rec._asdict())
        donor_pool = pd.DataFrame()
        match_level = -1
        used_cols: list[str] = []

        for level, cols in enumerate(match_sets):
            if not cols:
                continue
            pool = tus_with_archetypes.copy()
            for col in cols:
                if col not in pool.columns or col not in rec_series.index:
                    pool = pd.DataFrame()
                    break
                pool = pool.loc[pool[col].astype(str) == str(rec_series[col])]
            if not pool.empty:
                donor_pool = pool
                match_level = level
                used_cols = cols
                break

        if donor_pool.empty:
            donor_pool = tus_with_archetypes.copy()
            match_level = 999
            used_cols = []

        archetype_weights = (
            donor_pool.groupby(archetype_col)[weight_col]
            .sum(min_count=1)
            .fillna(0.0)
            .clip(lower=0.0)
        )
        available_archetypes = archetype_weights.index.astype(int).tolist()
        selected_archetype = int(
            _sample_weighted(
                [str(v) for v in available_archetypes],
                archetype_weights.tolist(),
                rng=rng,
            )
        )
        archetype_pool = donor_pool.loc[donor_pool[archetype_col].astype(int) == selected_archetype].copy()
        if archetype_pool.empty:
            archetype_pool = donor_pool.copy()
        donor_id = _sample_weighted_respondent(archetype_pool, weight_col=weight_col, rng=rng)

        donor_rows.append(
            {
                "person_id": str(rec_series["person_id"]),
                "respondent_id": donor_id,
                archetype_col: int(selected_archetype),
                "match_level": int(match_level),
                "match_cols": ",".join(used_cols),
                "donor_pool_size": int(len(donor_pool)),
                "archetype_pool_size": int(len(archetype_pool)),
                "n_available_archetypes": int(len(available_archetypes)),
            }
        )

    return pd.DataFrame(donor_rows)


def assign_tus_donors_by_archetype_draws(
    synth_people: pd.DataFrame,
    tus_people: pd.DataFrame,
    respondent_archetypes: pd.DataFrame,
    *,
    exact_match_cols: list[str],
    fallback_match_sets: list[list[str]] | None = None,
    weight_col: str = "person_weight",
    random_seed: int = 42,
    n_draws: int = 5,
    archetype_col: str = "archetype_id",
    draw_col: str = "draw_id",
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_position: int | None = None,
) -> pd.DataFrame:
    if int(n_draws) < 1:
        raise ValueError("n_draws must be at least 1.")

    draw_ids = range(int(n_draws))
    if show_progress and tqdm is not None:
        draw_ids = tqdm(
            draw_ids,
            total=int(n_draws),
            desc=progress_desc or "Assignment draws",
            position=progress_position,
            leave=True,
            dynamic_ncols=True,
        )

    draw_frames: list[pd.DataFrame] = []
    for draw_id in draw_ids:
        draw_df = assign_tus_donors_by_archetype(
            synth_people,
            tus_people,
            respondent_archetypes,
            exact_match_cols=exact_match_cols,
            fallback_match_sets=fallback_match_sets,
            weight_col=weight_col,
            random_seed=int(random_seed) + draw_id,
            archetype_col=archetype_col,
            show_progress=show_progress,
            progress_desc=f"{progress_desc or 'Assignment draws'} | draw {draw_id}",
            progress_position=None if progress_position is None else progress_position + 1,
        ).copy()
        draw_df[draw_col] = int(draw_id)
        draw_frames.append(draw_df)
    return pd.concat(draw_frames, ignore_index=True)


def summarize_profile_draw_uncertainty(
    profiles: pd.DataFrame,
    *,
    group_cols: list[str],
    draw_col: str = "draw_id",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    if draw_col not in profiles.columns:
        raise KeyError(f"Expected '{draw_col}' column in profiles.")

    if value_cols is None:
        excluded = set(group_cols) | {draw_col}
        value_cols = [
            col for col in profiles.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(profiles[col])
        ]
    if not value_cols:
        raise ValueError("No numeric value columns were available for uncertainty summaries.")

    rows: list[dict] = []
    for keys, sub in profiles.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n_draws"] = int(sub[draw_col].nunique())
        for col in value_cols:
            values = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_p05"] = np.nan
                row[f"{col}_p50"] = np.nan
                row[f"{col}_p95"] = np.nan
            else:
                row[f"{col}_mean"] = float(np.mean(values))
                row[f"{col}_std"] = float(np.std(values, ddof=0))
                row[f"{col}_p05"] = float(np.quantile(values, 0.05))
                row[f"{col}_p50"] = float(np.quantile(values, 0.50))
                row[f"{col}_p95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_weekday_weekend_stability(
    weekday_profiles: pd.DataFrame,
    weekend_profiles: pd.DataFrame,
    *,
    group_cols: list[str],
    value_cols: list[str] | None = None,
    weekday_label: str = "weekday",
    weekend_label: str = "weekend",
) -> pd.DataFrame:
    if value_cols is None:
        shared_cols = set(weekday_profiles.columns).intersection(weekend_profiles.columns)
        excluded = set(group_cols)
        value_cols = [
            col for col in shared_cols
            if col not in excluded and pd.api.types.is_numeric_dtype(weekday_profiles[col])
        ]
    if not value_cols:
        raise ValueError("No shared numeric value columns were available for stability summaries.")

    left = weekday_profiles[group_cols + value_cols].copy()
    right = weekend_profiles[group_cols + value_cols].copy()
    merged = left.merge(
        right,
        on=group_cols,
        how="inner",
        suffixes=(f"_{weekday_label}", f"_{weekend_label}"),
    )
    for col in value_cols:
        merged[f"{col}_delta"] = (
            pd.to_numeric(merged[f"{col}_{weekend_label}"], errors="coerce")
            - pd.to_numeric(merged[f"{col}_{weekday_label}"], errors="coerce")
        )
        merged[f"{col}_abs_delta"] = merged[f"{col}_delta"].abs()
    return merged
