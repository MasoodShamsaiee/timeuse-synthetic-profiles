import pandas as pd


def test_build_household_eval_id_prefers_existing_household_id():
    from timeuse_synthetic_profiles.workflows import build_household_eval_id

    df = pd.DataFrame({"area": ["A1", "A1"], "household_id": [10, 11]})
    out = build_household_eval_id(df)
    assert out["household_id_eval"].tolist() == ["A1::10", "A1::11"]


def test_filter_day_type_inputs_filters_respondents_and_episodes():
    from timeuse_synthetic_profiles.workflows import filter_day_type_inputs

    resp = pd.DataFrame(
        {"respondent_id": ["1", "2"], "day_type": ["weekday", "weekend"], "sex": ["male", "female"], "age": [30, 40]}
    )
    epi = pd.DataFrame(
        {
            "respondent_id": ["1", "2"],
            "start_minute": [0, 0],
            "end_minute": [60, 60],
            "activity_code": [101, 202],
        }
    )
    day_resp, day_ep = filter_day_type_inputs(resp, epi, day_type="weekday")
    assert day_resp["respondent_id"].tolist() == ["1"]
    assert day_ep["respondent_id"].tolist() == ["1"]
