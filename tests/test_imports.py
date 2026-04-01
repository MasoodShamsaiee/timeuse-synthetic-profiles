import pandas as pd


def test_package_imports():
    import timeuse_synthetic_profiles
    from timeuse_synthetic_profiles import derive_tus_matching_features, hhmm_to_minutes

    assert hasattr(timeuse_synthetic_profiles, "prepare_timeuse_data")
    assert callable(derive_tus_matching_features)
    assert hhmm_to_minutes("07:30") == 450


def test_matching_feature_smoke():
    from timeuse_synthetic_profiles import derive_tus_matching_features

    df = pd.DataFrame(
        {
            "respondent_id": ["1", "2"],
            "sex": ["male", "female"],
            "age": [28, 71],
            "person_weight": [1.2, 0.8],
            "household_size": [3, 1],
            "has_children": [1, 0],
            "employment_cat": ["employed", "retired"],
            "is_worker": [1, 0],
        }
    )

    out = derive_tus_matching_features(df)
    assert {"sex_std", "age_band", "hhsize_cat", "economic_activity_cat"}.issubset(out.columns)
    assert len(out) == 2
