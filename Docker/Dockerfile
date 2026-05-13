# AOVE Oracle — Production Docker Image
# ============================================================
# Build:   docker build -t aove-oracle .
# Run:     docker run -p 8000:8000 aove-oracle
# Dashboard: http://localhost:8000/dashboard

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py          .
COPY dashboard.html  .
COPY data/           ./data/
COPY AOVE_model/best_aove_model.pth ./AOVE_model/

# Expose port
EXPOSE 8000

# Non-root user
RUN useradd -m appuser
USER appuser

# Run with Gunicorn (multi-process) + Uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "--bind", "0.0.0.0:8000", "api:app"]
