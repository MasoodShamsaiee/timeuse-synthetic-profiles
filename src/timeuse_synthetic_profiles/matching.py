from __future__ import annotations

import random

import pandas as pd


def _sample_weighted_respondent(pool: pd.DataFrame, *, weight_col: str, rng: random.Random) -> str:
    if pool.empty:
        raise ValueError("Cannot sample from an empty donor pool.")
    weights = pd.to_numeric(pool[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) <= 0:
        return str(pool.iloc[rng.randrange(len(pool))]["respondent_id"])
    return str(rng.choices(pool["respondent_id"].astype(str).tolist(), weights=weights.tolist(), k=1)[0])


def assign_tus_donors(
    synth_people: pd.DataFrame,
    tus_people: pd.DataFrame,
    *,
    exact_match_cols: list[str],
    fallback_match_sets: list[list[str]] | None = None,
    weight_col: str = "person_weight",
    random_seed: int = 42,
) -> pd.DataFrame:
    if "respondent_id" not in tus_people.columns:
        raise KeyError("Expected 'respondent_id' column in TUS respondents.")
    if "person_id" not in synth_people.columns:
        raise KeyError("Expected 'person_id' column in synthetic people.")

    fallback_match_sets = [] if fallback_match_sets is None else fallback_match_sets
    match_sets = [list(exact_match_cols)] + [list(cols) for cols in fallback_match_sets]
    rng = random.Random(int(random_seed))

    donor_rows: list[dict] = []
    for rec in synth_people.itertuples(index=False):
        rec_series = pd.Series(rec._asdict())
        donor_pool = pd.DataFrame()
        match_level = -1
        used_cols: list[str] = []

        for level, cols in enumerate(match_sets):
            if not cols:
                continue
            pool = tus_people.copy()
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
            donor_pool = tus_people.copy()
            match_level = 999
            used_cols = []

        donor_id = _sample_weighted_respondent(donor_pool, weight_col=weight_col, rng=rng)
        donor_rows.append(
            {
                "person_id": str(rec_series["person_id"]),
                "respondent_id": donor_id,
                "match_level": int(match_level),
                "match_cols": ",".join(used_cols),
            }
        )

    return pd.DataFrame(donor_rows)


def attach_profiles_to_synthpop(
    synth_people: pd.DataFrame,
    donor_assignments: pd.DataFrame,
    respondent_profiles: pd.DataFrame,
) -> pd.DataFrame:
    merged = synth_people.merge(donor_assignments, on="person_id", how="left", validate="1:1")
    merged = merged.merge(respondent_profiles, on="respondent_id", how="left", validate="m:m")
    return merged
