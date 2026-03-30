# =============================================================
# Dockerfile — MLOps Vertex AI Training Container
# =============================================================

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

LABEL maintainer="your-email@example.com"
LABEL description="Purchase Price Prediction — Vertex AI Training Container"
LABEL version="1.0"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY src/     /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY data/    /app/data/

# ── Entry point ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: run training
ENTRYPOINT ["python", "src/models/train.py"]
