# Data Contracts

## Purpose

This package connects harmonized TUS data to a synthetic population through several workflow stages. The main contracts are listed here so the package can evolve without hidden coupling.

## Harmonized respondent table

Required columns:

- `respondent_id`
- `sex`
- `age`

Common optional columns:

- `person_weight`
- `day_type`
- `household_size`
- `has_children`
- `employment_cat`
- `is_student`
- `is_worker`

## Harmonized episode table

Required columns:

- `respondent_id`
- `start_minute`
- `end_minute`
- `activity_code`

Common optional columns:

- `location_code`
- `episode_weight`

## Synthetic population input

Required columns:

- `area`
- `sex`

Preferred columns:

- `person_id`
- `household_id` or `HID`
- `age`

## District profile outputs

Common output identifiers:

- `district_id`
- `hour`
- one or more profile-share columns such as `share_home_awake`, `share_sleep`, `share_away`, `share_home_total`

## Archetype outputs

Common columns:

- `respondent_id`
- `archetype_id`

Centroid profile outputs additionally expect:

- `hour`
- profile share columns
