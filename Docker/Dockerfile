# AOVE Oracle — Production Docker Image
# ============================================================
# Build (from repo root):  docker build -f Docker/Dockerfile -t aove-oracle .
# Run:                      docker run -p 8000:8000 aove-oracle
# Dashboard:                http://localhost:8000/dashboard

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package first (layer cache). Dependencies come from pyproject.toml.
COPY pyproject.toml ./
COPY aove/ ./aove/
RUN pip install --no-cache-dir .

# Application assets and runtime data (model + CSVs are mounted at run time).
COPY dashboard.html ./
COPY data/ ./data/
COPY AOVE_model/best_aove_model.pth ./AOVE_model/

EXPOSE 8000

RUN useradd -m appuser
USER appuser

# Gunicorn (multi-process) + Uvicorn workers, serving the package app.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", \
     "--bind", "0.0.0.0:8000", "aove.api:app"]
