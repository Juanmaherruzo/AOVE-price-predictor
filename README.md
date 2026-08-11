# AOVE Oracle — Weekly EVOO Price Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-D22128)

![AOVE Oracle dashboard](docs/Use_example.png)

---

A weekly forecast system for Extra Virgin Olive Oil (EVOO) origin prices in Andalusia, Spain. The model ingests the current market price, spatially aggregated climate data and macroeconomic indicators, and returns a point prediction with an 80% confidence interval for the following week.

**Read the results section before the architecture section.** The headline is a negative result, reported deliberately: the model does not beat a naive baseline.

---

## Results

Every number below is measured on the same chronological hold-out — the last 20% of the series, 171 weeks from 2023-01-23 to 2026-04-27. Scalers are fitted on the training split only.

| Predictor | MAE (EUR/kg) | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Persistence** (repeat last week's price) | **0.057** | **0.130** | **0.95 %** | **0.987** |
| Random walk with drift | 0.058 | 0.130 | 0.96 % | 0.987 |
| Bimodal LSTM (this project) | 0.073 | 0.151 | 1.23 % | 0.982 |

**The LSTM does not beat persistence.** Its skill score — the fractional MAE reduction over the naive baseline — is **−27.7%**: the model is about a quarter *worse* than predicting that next week's price equals this week's.

### Why this is the headline

Weekly EVOO origin prices are close to a random walk. A model that is handed last week's price as an input feature — as this one is — will score a high R² by learning little more than "repeat the input". An R² of 0.98 on this task is what *doing nothing* achieves, so quoting it as a result would be meaningless. Reproduce the comparison yourself:

```bash
aove-benchmark --json benchmark.json
```

### What changed, and what it cost

An earlier version of this README reported MAE 0.204 EUR/kg and R² 0.943 with no baseline. Adding the baseline exposed two things:

1. **A temporal alignment bug.** `aove_lag_price` is shifted one week in the ETL, and the sequence builder was *also* reading the macro row at `t-1` — so the model saw a price two weeks stale while the config and docstrings both claimed one week. Fixing it (`features.py`) cut validation MAE from 0.147 to 0.073, roughly halving the error. Two regression tests now pin the alignment (`tests/test_features.py`).
2. **The comparison that was missing.** Even after the fix, persistence still wins. That is the honest state of the project.

### Where the model could still earn its place

The persistence baseline is unbeatable on price *level* but says nothing useful about price *change*, which is what a mill or a cooperative actually needs. The natural next step is to retarget the model on the weekly delta and score it against "delta = 0" and against directional accuracy vs. a coin flip. Until that is done, this repository is best read as a complete, tested ETL-to-API pipeline with an honest evaluation, not as a forecasting result.

### Diagnostic plots

Generated from the corrected model (`aove-train`), on the same validation split.

| Learning curve | Predicted vs Actual |
|---|---|
| ![learning curve](AOVE_model/01_learning_curve.png) | ![pred vs actual](AOVE_model/02_pred_vs_actual.png) |

| Residuals | Metrics dashboard |
|---|---|
| ![residuals](AOVE_model/03_residuals.png) | ![metrics](AOVE_model/04_metrics_dashboard.png) |

---

## Architecture

![Pipeline architecture](Pipeline_architecture_AOVE.svg)

### Model: Bimodal LSTM

The network has two input branches that are fused before the output head:

- **High-frequency branch (LSTM)** — a 2-layer LSTM with hidden size 128 processes a sliding window of 104 weeks (2 years) of climate features: rainfall, max temperature, water deficit and seasonal encoding (sin/cos of ISO week).
- **Macro branch (MLP)** — the latest week's macroeconomic snapshot (CPI, diesel price, olive stock delta, olive surface delta, lagged AOVE price) is concatenated with the LSTM's last hidden state and passed through a 3-layer fully connected head.

Monte Carlo Dropout (Gal & Ghahramani, 2016) provides the confidence interval: 200 stochastic forward passes with dropout active approximate the predictive posterior.

### Data pipeline (ETL)

- **Climate** — ERA5-Land weekly observations for 13 Andalusian olive-growing locations (Open-Meteo archive API), spatially aggregated by cultivated olive surface (weighted average).
- **EVOO origin price** — the weekly series published by the European Commission's agri-food data portal. This is the target, and its one-week lag is a model input.
- **CPI** — INE monthly series, forward-filled onto the weekly grid. A 15-day publication lag is modelled explicitly with `merge_asof`.
- **Diesel, olive stocks and planted surface** — ⚠️ **these three are hard-coded annual constants, not downloaded data.** They were stand-ins during the first build and were never replaced. Each changes once or twice a year, so none of them can carry week-scale information — which is part of why the model does not beat persistence.
- At inference the user supplies only the current week's price; everything else is pre-loaded.

Full provenance, licences and attribution: **[DATA.md](DATA.md)**.

---

## Repository layout

```
.
├── aove/                       # Installable package — single source of truth
│   ├── config.py               # Paths, feature columns, hyperparameters
│   ├── model.py                # Bimodal LSTM (AOVEPricePredictor)
│   ├── etl.py                  # Spatial climate aggregation + macro alignment
│   ├── features.py             # Temporal sequence builder (leakage-free split)
│   ├── inference.py            # Model loading + MC-Dropout sampling
│   ├── cli.py                  # Interactive CLI            -> aove-cli
│   ├── api.py                  # FastAPI server             -> aove.api:app
│   ├── training.py             # Trainer + fine-tuner       -> aove-train
│   ├── visualise.py            # Diagnostic plots
│   ├── data_prepare.py         # Data ingestion             -> aove-prepare
│   └── benchmark.py            # Baselines vs model         -> aove-benchmark
├── tests/                      # pytest suite (model, ETL, features, inference)
├── data/                       # Published datasets (climate, macro, raw price)
├── AOVE_model/                 # Trained checkpoint, diagnostics, benchmark.json
├── dashboard.html              # Single-page frontend (served by FastAPI)
├── pyproject.toml              # Packaging, dependencies, tool config
├── DATA.md                     # Data provenance, licences and attribution
├── Pipeline_architecture_AOVE.svg
├── docs/Use_example.png
└── Docker/                     # Dockerfile, docker-compose.yml, .dockerignore
```

> **Everything needed to reproduce the results is in the repository** — the datasets, the trained checkpoint and the benchmark script. Run `aove-benchmark` and check the numbers yourself. See [DATA.md](DATA.md) for where each column comes from.

---

## API

### Install

```bash
# with pip
pip install -e ".[dev]"

# or with uv (faster)
uv venv && uv pip install -e ".[dev]"
```

### Run locally

```bash
uvicorn aove.api:app --reload --port 8000
```

### Run with Docker

The image contains the code; `docker-compose.yml` mounts the checkpoint and the
datasets from your working copy, so a plain clone has everything it needs:

```bash
cd Docker
docker compose up --build
```

If either artefact is missing, the container still starts and `/health` reports
`degraded`, listing what it could not find.

The dashboard is available at `http://localhost:8000/dashboard`.  
Swagger UI at `http://localhost:8000/docs`.

CORS defaults to the dashboard's own origin; override with
`AOVE_CORS_ORIGINS=https://example.org` if you serve the frontend elsewhere.

### POST `/predict`

```json
{
  "current_price": 4.85,
  "mc_samples": 200
}
```

```json
{
  "status": "success",
  "predicted_price": 4.71,
  "confidence_interval": { "low_80": 4.52, "high_80": 4.89 },
  "trend_signal": "DOWN",
  "confidence_pct": 62,
  "prediction_week": "2025-11-17",
  "mae_reference": 0.073,
  "mae_persistence_baseline": 0.057
}
```

`mae_persistence_baseline` is returned on every prediction so a client can see
that the naive baseline is more accurate than the model on the validation split.

### Command-line tools

Installing the package exposes four console entry points. Paths resolve against
the repository root, so the commands work from any directory; set `AOVE_ROOT` to
point them elsewhere.

```bash
aove-cli                                         # interactive weekly prediction
aove-benchmark                                   # model vs naive baselines
aove-train --epochs 200                          # train the base model
aove-train --finetune                            # fine-tune the FC head
aove-prepare --price-csv ./data/precio_historico.csv   # rebuild datasets
```

---

## Live demo

> Deployment coming soon — the API will be hosted on [Hugging Face Spaces](https://huggingface.co/spaces), [Render](https://render.com) or [Railway](https://railway.app). A public URL will appear here once live.

---

## Tech stack

| Layer | Library |
|---|---|
| Deep learning | PyTorch 2.3 |
| Data processing | pandas, NumPy, scikit-learn |
| API | FastAPI + Uvicorn / Gunicorn |
| Containerisation | Docker + docker-compose |
| Quality | ruff, black, mypy (strict), pytest |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{herruzo2026aove,
  author = {Herruzo, Juan Manuel},
  title  = {AOVE Oracle: Weekly EVOO Price Predictor},
  year   = {2026},
  url    = {https://github.com/Juanmaherruzo/AOVE-price-predictor}
}
```

---

## Licence

Code and datasets released under the **Apache License 2.0** — see [LICENSE](LICENSE).
Use it, fork it, build on it, commercially or otherwise.

Third-party data carries its own attribution requirements, listed in
[DATA.md](DATA.md): ERA5-Land © Copernicus / ECMWF via Open-Meteo (CC BY 4.0),
CPI © INE, EVOO price series © European Commission (Decision 2011/833/EU).

---

## Contact

**Juan Manuel Herruzo**  
juanmherruzo@gmail.com

Questions, corrections and pull requests are welcome — particularly on the
retargeting described above, or if you have real MITECO diesel and AICA stock
series to replace the three proxy columns.
