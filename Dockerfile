# Production Dockerfile for PJM East Probabilistic Energy Forecasting API
# Python 3.11-slim base for lightweight, security-hardened deployment

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    MODEL_PATH=models/pjm_production_forecaster_phase8.joblib \
    SCHEMA_PATH=models/production_feature_schema.json \
    PROB_MODEL_PATH=models/pjm_probabilistic_forecaster_phase9.joblib \
    CALIBRATION_PATH=models/conformal_calibration_params.json \
    ADAPTIVE_CALIBRATION_PATH=models/adaptive_calibration_params.json

WORKDIR /app

# Install system dependencies needed for curl health check and ECMWF ecCodes C libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libeccodes-dev \
    libeccodes-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, model artifacts, and evaluation outputs
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY outputs/improvement/ ./outputs/improvement/


# Create a non-privileged user and switch permissions
RUN useradd -u 1000 -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default HTTP port
EXPOSE 8000

# Docker native healthcheck polling the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch ASGI server via uvicorn
CMD ["sh", "-c", "uvicorn src.api.app:app --host ${HOST} --port ${PORT}"]
