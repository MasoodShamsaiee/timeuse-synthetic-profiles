from __future__ import annotations

import numpy as np
import pandas as pd


def _age_band(age: pd.Series) -> pd.Series:
    age_num = pd.to_numeric(age, errors="coerce")
    bins = [-np.inf, 5, 12, 17, 24, 34, 44, 54, 64, 74, np.inf]
    labels = [
        "0_5",
        "6_12",
        "13_17",
        "18_24",
        "25_34",
        "35_44",
        "45_54",
        "55_64",
        "65_74",
        "75_plus",
    ]
    return pd.cut(age_num, bins=bins, labels=labels, right=False).astype("string")


def _normalize_sex(x: pd.Series) -> pd.Series:
    out = x.astype("string").str.strip().str.lower()
    out = out.replace(
        {
            "0": "female",
            "1": "male",
            "f": "female",
            "m": "male",
        }
    )
    return out


def _derive_synth_economic_activity(work: pd.DataFrame) -> pd.Series:
    """
    Build a coarse labour-market attachment feature from synthpop's lfact mapping.

    In the upstream synthpop pipeline, lfact is collapsed to three coarse classes.
    We use a conservative two-way summary that can be approximated on the TUS side:
    - active: labour-market attached
    - inactive: not in the labour force / otherwise not labour-market attached
    """
    if "lfact_cat" not in work.columns:
        return pd.Series("unknown", index=work.index, dtype="string")

    lfact = work["lfact_cat"].astype("string")
    return lfact.map(
        {
            "1867": "active",
            "1868": "active",
            "1869": "inactive",
        }
    ).fillna("unknown").astype("string")


def _derive_tus_economic_activity(work: pd.DataFrame) -> pd.Series:
    """
    Build a coarse labour-market attachment feature from harmonized TUS activity fields.

    We intentionally keep this coarse so it aligns with the synthetic-side proxy.
    """
    out = pd.Series("unknown", index=work.index, dtype="string")

    if "employment_cat" in work.columns:
        employment = work["employment_cat"].astype("string").str.strip().str.lower()
        out.loc[employment == "employed"] = "active"
        out.loc[employment.isin(["student", "retired", "home_care", "other"])] = "inactive"

    if "is_worker" in work.columns:
        is_worker = pd.to_numeric(work["is_worker"], errors="coerce")
        out.loc[is_worker == 1] = "active"

    if "worked_last_week" in work.columns:
        worked_last_week = pd.to_numeric(work["worked_last_week"], errors="coerce")
        out.loc[worked_last_week == 1] = "active"

    return out.fillna("unknown").astype("string")


def _derive_household_comp_proxy(work: pd.DataFrame) -> pd.Series:
    """
    Build a shared, coarse household-composition feature available on both sides.

    Categories:
    - one_person
    - with_children
    - without_children
    """
    out = pd.Series("without_children", index=work.index, dtype="string")

    if "hhsize_actual" in work.columns:
        hhsize = pd.to_numeric(work["hhsize_actual"], errors="coerce")
    elif "household_size" in work.columns:
        hhsize = pd.to_numeric(work["household_size"], errors="coerce")
    elif "hhsize" in work.columns:
        hhsize = pd.to_numeric(work["hhsize"], errors="coerce")
    else:
        hhsize = pd.Series(np.nan, index=work.index)

    has_children = pd.to_numeric(work.get("has_children"), errors="coerce").fillna(0)
    out.loc[has_children > 0] = "with_children"
    out.loc[hhsize == 1] = "one_person"
    return out.astype("string")


def derive_synth_matching_features(
    syn_df: pd.DataFrame,
    *,
    district_col: str = "area",
    household_id_col: str = "household_id",
) -> pd.DataFrame:
    work = syn_df.copy()
    if district_col not in work.columns:
        raise KeyError(f"Expected district column '{district_col}' in synthetic population.")

    work["district_id"] = work[district_col].astype(str)
    work["sex_std"] = _normalize_sex(work["sex"])
    work["age_band"] = _age_band(work.get("age"))
    work["age_years"] = pd.to_numeric(work.get("age"), errors="coerce")

    if "hhsize" in work.columns:
        hhsize = pd.to_numeric(work["hhsize"], errors="coerce").fillna(1)
        work["hhsize_cat"] = hhsize.clip(lower=1, upper=5).astype(int).astype(str)
    else:
        work["hhsize_cat"] = "1"

    if household_id_col in work.columns:
        hh_id = work[household_id_col]
        hh_size_actual = work.groupby(hh_id, dropna=False)["person_id"].transform("size")
        work["hhsize_actual"] = hh_size_actual
        children = work.groupby(hh_id, dropna=False)["age_years"].transform(
            lambda s: float((pd.to_numeric(s, errors="coerce") < 18).sum())
        )
        work["has_children"] = (children > 0).astype(int)
    else:
        work["hhsize_actual"] = 1
        work["has_children"] = (pd.to_numeric(work["age_years"], errors="coerce") < 18).astype(int)

    if "hhtype" in work.columns:
        work["hhtype_cat"] = pd.to_numeric(work["hhtype"], errors="coerce").fillna(-1).astype(int).astype(str)
    else:
        work["hhtype_cat"] = "-1"

    for col in ["lfact", "hdgree", "totinc", "cfstat"]:
        if col in work.columns:
            work[f"{col}_cat"] = pd.to_numeric(work[col], errors="coerce").fillna(-1).astype(int).astype(str)
        else:
            work[f"{col}_cat"] = "-1"

    work["is_child"] = (pd.to_numeric(work["age_years"], errors="coerce") < 18).astype(int)
    work["is_senior"] = (pd.to_numeric(work["age_years"], errors="coerce") >= 65).astype(int)
    work["economic_activity_cat"] = _derive_synth_economic_activity(work)
    work["household_comp_proxy_cat"] = _derive_household_comp_proxy(work)
    return work


def derive_tus_matching_features(tus_df: pd.DataFrame) -> pd.DataFrame:
    work = tus_df.copy()
    work["sex_std"] = _normalize_sex(work["sex"])
    work["age_band"] = _age_band(work["age"])
    work["age_years"] = pd.to_numeric(work["age"], errors="coerce")

    default_zero = {
        "has_children": 0,
        "is_student": 0,
        "is_worker": 0,
        "works_from_home": 0,
    }
    for col, default in default_zero.items():
        if col not in work.columns:
            work[col] = default

    if "household_size" in work.columns:
        hhsize = pd.to_numeric(work["household_size"], errors="coerce").fillna(1)
        work["hhsize_cat"] = hhsize.clip(lower=1, upper=5).astype(int).astype(str)
    else:
        work["hhsize_cat"] = "1"

    for col in ["employment_cat", "student_cat", "commute_cat", "tenure_cat"]:
        if col not in work.columns:
            work[col] = "unknown"

    if "person_weight" not in work.columns:
        work["person_weight"] = 1.0

    work["economic_activity_cat"] = _derive_tus_economic_activity(work)
    work["household_comp_proxy_cat"] = _derive_household_comp_proxy(work)

    return work
