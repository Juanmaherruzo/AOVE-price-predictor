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

# Application assets. The checkpoint and the CSV datasets are NOT baked into the
# image: they are not redistributed with this repository (see
# MODEL_NOT_INCLUDED.md) and docker-compose.yml mounts them from the host at run
# time. Copying them here would make `docker build` fail on a fresh clone.
COPY dashboard.html ./
RUN mkdir -p /app/data /app/AOVE_model

# Resolve data and checkpoint paths against /app rather than the caller's cwd.
ENV AOVE_ROOT=/app

EXPOSE 8000

RUN useradd -m appuser
USER appuser

# Gunicorn (multi-process) + Uvicorn workers, serving the package app.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", \
     "--bind", "0.0.0.0:8000", "aove.api:app"]
