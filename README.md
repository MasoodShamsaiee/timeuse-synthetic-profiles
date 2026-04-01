# timeuse-synthetic-profiles

`timeuse-synthetic-profiles` is the extracted time-use behavior package from the larger research codebase. It contains TUS harmonization, diary/state expansion, donor matching, archetype fitting, classifier workflows, district aggregation, evaluation helpers, and hub-export packaging.

## What is included

- harmonization helpers for TUS respondent and episode files
- respondent profile construction from episode diaries
- synthetic/TUS matching feature engineering
- baseline donor assignment
- schedule archetype fitting and draw-based assignment
- classifier-based district archetype workflows
- district profile aggregation and evaluation summaries
- hub-oriented parquet/json export helpers
- supporting notebooks copied from the original research project

## Package layout

```text
src/timeuse_synthetic_profiles/
  harmonize.py
  io.py
  diaries.py
  features.py
  matching.py
  archetypes.py
  aggregation.py
  evaluation.py
  hub.py
  workflows.py
notebooks/
```

## Quick start

```powershell
conda run -n dsm_qc python -m pip install -e .[dev]
conda run -n dsm_qc python -c "import timeuse_synthetic_profiles; print('ok')"
```

## Notes

- this repo reads synthetic population files as inputs but does not import the `synthetic-population-qc` package directly
- see [docs/data_contracts.md](docs/data_contracts.md) for the expected respondent, episode, and synthetic-population table shapes
- default paths in the workflow module still reflect the original research data layout, so the next cleanup pass should externalize those paths more aggressively
