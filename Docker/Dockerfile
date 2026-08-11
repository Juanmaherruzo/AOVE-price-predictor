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

# Application assets. The checkpoint and the CSV datasets are not baked into the
# image: docker-compose.yml mounts them from the host at run time so they can be
# swapped without a rebuild. Keeping the image free of data also keeps it small.
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
