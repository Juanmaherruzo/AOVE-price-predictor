# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-11

### Changed
- **Relicensed to Apache 2.0.** The previous CC BY-NC 4.0 licence was not an
  open-source licence, its NonCommercial clause blocked the exact evaluation the
  project is published for, and GitHub could not detect it (the repository showed
  as unlicensed). The "for commercial licensing, get in touch" framing is gone.
- **The datasets and the trained checkpoint are now published.** Every source is
  freely redistributable with attribution, so withholding them served no purpose
  beyond making the results unverifiable. `aove-benchmark` now reproduces the
  published table from a plain clone with no arguments.

### Added
- `DATA.md`: per-column provenance, licences and attribution.

### Fixed
- **Three macro columns were undocumented proxies.** `diesel_price_eur`,
  `surface_delta_pct` and `stock_delta_pct` are hard-coded annual constants
  defined in `data_prepare.py`, not downloaded data — 16, 11 and 27 distinct
  values respectively across 850 weeks. The README presented all five macro
  inputs as official bulletin data. Both the README and `DATA.md` now state
  plainly which three are proxies, and note that a feature changing once a year
  cannot carry week-scale information — part of why the model loses to
  persistence.
- The price loader was named `PoolRedLoader` while its docstring credited a MAPA
  scraper; the actual source is the European Commission's weekly series. Renamed
  to `EUPriceLoader`, with the `--poolred-csv` flag renamed to `--price-csv` and
  the attribution corrected in the code, the dashboard and the docs.
- Removed `MODEL_NOT_INCLUDED.md`, which claimed the checkpoint was withheld to
  protect months of effort — an argument the benchmark had just disproved.

## [0.2.0] - 2026-08-11

### Fixed
- **Temporal alignment of the macro snapshot.** `aove_lag_price` is shifted one
  week in the ETL, and `TemporalSequenceBuilder` was *also* reading the macro row
  at `t-1`, so the model was fed a price two weeks stale while `aove_lag_weeks`
  and the docstrings both stated one week. Correcting it cut validation MAE from
  0.147 to 0.073 EUR/kg. Two regression tests now pin the alignment of both the
  macro snapshot and the climate window.
- `docker build` failed on a fresh clone: the Dockerfile copied `data/` and the
  checkpoint, neither of which is redistributed. They are mounted by
  `docker-compose.yml` at run time instead.
- Runtime paths resolved against the working directory, so the installed console
  scripts only worked from the repository root. They now resolve against the
  package location, overridable with `AOVE_ROOT`.
- Copyright year in `LICENSE` (2025 -> 2026).
- Citation block was not fenced and contained a nested Markdown link inside the
  BibTeX `url` field, so it could not be copied into a reference manager.

### Added
- `aove.benchmark` and the `aove-benchmark` console script: scores the model
  against a persistence baseline and a random walk with drift on the same
  chronological validation split, and reports a skill score.
- `/predict` now returns `mae_persistence_baseline` alongside `mae_reference`.

### Changed
- **The README now leads with a negative result.** Previously it reported
  MAE 0.204 EUR/kg and R² 0.943 with no baseline. Against persistence
  (MAE 0.057, R² 0.987) the model's skill score is -27.7%: it does not beat
  repeating last week's price. Weekly EVOO origin prices are close to a random
  walk, so an R² near 0.98 is what doing nothing achieves and cannot be read as
  forecasting skill.
- CORS no longer defaults to `*`; the allowed origins are read from
  `AOVE_CORS_ORIGINS` and default to the dashboard's own origin.
- Diagnostic plots regenerated from the corrected model.

## [0.1.0] - 2026-07-18

### Added
- Initial release: bimodal LSTM for weekly EVOO origin price prediction.
- `aove` package: single source of truth for the model, ETL, feature builder,
  inference, CLI, API, training and data-ingestion layers.
- Monte Carlo Dropout confidence intervals (Gal & Ghahramani, 2016).
- Console entry points: `aove-cli`, `aove-train`, `aove-prepare`.
- `pyproject.toml` packaging, CI pipeline (ruff, black, mypy, pytest) and tests.

### Changed
- Extracted the previously duplicated model/ETL code (present in five places)
  into the shared `aove` package.
- Converted the Jupyter notebooks into importable, tested Python modules.
