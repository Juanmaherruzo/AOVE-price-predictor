# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
