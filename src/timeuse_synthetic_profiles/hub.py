from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class HubTimeUsePackage:
    da_sociodemographic_profile: pd.DataFrame
    synthetic_population: pd.DataFrame
    area_behavioral_archetype: pd.DataFrame
    tus_activity_profile: pd.DataFrame
    activity_schedule_generator: pd.DataFrame
    manifest: dict[str, Any]


def build_da_sociodemographic_profile(syn_people: pd.DataFrame) -> pd.DataFrame:
    work = syn_people.copy()
    work["district_id"] = work["district_id"].astype(str)
    work["age_years"] = pd.to_numeric(work.get("age_years"), errors="coerce")
    work["has_children"] = pd.to_numeric(work.get("has_children"), errors="coerce").fillna(0)
    grouped = work.groupby("district_id", as_index=False).agg(
        n_people=("person_id", "size"),
        mean_age=("age_years", "mean"),
        share_children=("is_child", "mean"),
        share_seniors=("is_senior", "mean"),
        share_households_with_children=("has_children", "mean"),
    )

    age_band_mix = (
        work.assign(weight=1.0)
        .pivot_table(index="district_id", columns="age_band", values="weight", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    age_band_cols = [col for col in age_band_mix.columns if col != "district_id"]
    if age_band_cols:
        totals = age_band_mix[age_band_cols].sum(axis=1).replace(0, pd.NA)
        for col in age_band_cols:
            age_band_mix[f"share_age_band__{col}"] = age_band_mix[col] / totals
        age_band_mix = age_band_mix[["district_id", *[f"share_age_band__{col}" for col in age_band_cols]]]

    return grouped.merge(age_band_mix, on="district_id", how="left")


def build_area_behavioral_archetype(
    district_archetype_counts: pd.DataFrame,
    *,
    count_col: str = "expected_n_people",
) -> pd.DataFrame:
    work = district_archetype_counts.copy()
    work["district_id"] = work["district_id"].astype(str)
    work["archetype_id"] = pd.to_numeric(work["archetype_id"], errors="coerce").astype("Int64")
    work[count_col] = pd.to_numeric(work[count_col], errors="coerce").fillna(0.0)
    totals = work.groupby("district_id")[count_col].transform("sum").replace(0, pd.NA)
    work["archetype_share"] = work[count_col] / totals
    return work.sort_values(["district_id", "archetype_id"]).reset_index(drop=True)


def build_tus_activity_profile(centroid_profiles: pd.DataFrame) -> pd.DataFrame:
    work = centroid_profiles.copy()
    work["archetype_id"] = pd.to_numeric(work["archetype_id"], errors="coerce").astype("Int64")
    work["hour"] = pd.to_numeric(work["hour"], errors="coerce").astype("Int64")
    return work.sort_values(["archetype_id", "hour"]).reset_index(drop=True)


def build_activity_schedule_generator_output(
    district_profiles: pd.DataFrame,
    *,
    method: str,
    day_type: str,
) -> pd.DataFrame:
    work = district_profiles.copy()
    work["district_id"] = work["district_id"].astype(str)
    work["hour"] = pd.to_numeric(work["hour"], errors="coerce").astype("Int64")
    work["generator_method"] = method
    work["day_type"] = day_type
    return work.sort_values(["district_id", "hour"]).reset_index(drop=True)


def build_hub_manifest(
    *,
    method: str,
    day_type: str,
    output_tag: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "hub_entities": {
            "da_sociodemographic_profile": "DisseminationArea (DA) -> DA SocioDemographicProfile",
            "synthetic_population": "SyntheticPopulation",
            "area_behavioral_archetype": "AreaBehavioralArchetype",
            "tus_activity_profile": "TUSActivityProfile",
            "activity_schedule_generator": "ActivityScheduleGenerator",
        },
        "method": method,
        "day_type": day_type,
        "output_tag": output_tag,
        "config": config or {},
    }


def build_hub_timeuse_package(
    *,
    syn_people: pd.DataFrame,
    district_profiles: pd.DataFrame,
    centroid_profiles: pd.DataFrame,
    district_archetype_counts: pd.DataFrame,
    method: str,
    day_type: str,
    output_tag: str,
    config: dict[str, Any] | None = None,
) -> HubTimeUsePackage:
    manifest = build_hub_manifest(method=method, day_type=day_type, output_tag=output_tag, config=config)
    return HubTimeUsePackage(
        da_sociodemographic_profile=build_da_sociodemographic_profile(syn_people),
        synthetic_population=syn_people.copy(),
        area_behavioral_archetype=build_area_behavioral_archetype(district_archetype_counts),
        tus_activity_profile=build_tus_activity_profile(centroid_profiles),
        activity_schedule_generator=build_activity_schedule_generator_output(
            district_profiles,
            method=method,
            day_type=day_type,
        ),
        manifest=manifest,
    )


def export_hub_timeuse_package(
    package: HubTimeUsePackage,
    *,
    out_dir: str | Path,
    stem: str,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "da_sociodemographic_profile": out_dir / f"{stem}__da_sociodemographic_profile.parquet",
        "synthetic_population": out_dir / f"{stem}__synthetic_population.parquet",
        "area_behavioral_archetype": out_dir / f"{stem}__area_behavioral_archetype.parquet",
        "tus_activity_profile": out_dir / f"{stem}__tus_activity_profile.parquet",
        "activity_schedule_generator": out_dir / f"{stem}__activity_schedule_generator.parquet",
        "manifest": out_dir / f"{stem}__manifest.json",
    }

    package.da_sociodemographic_profile.to_parquet(outputs["da_sociodemographic_profile"], index=False)
    package.synthetic_population.to_parquet(outputs["synthetic_population"], index=False)
    package.area_behavioral_archetype.to_parquet(outputs["area_behavioral_archetype"], index=False)
    package.tus_activity_profile.to_parquet(outputs["tus_activity_profile"], index=False)
    package.activity_schedule_generator.to_parquet(outputs["activity_schedule_generator"], index=False)
    outputs["manifest"].write_text(json.dumps(package.manifest, indent=2), encoding="utf-8")
    return outputs
