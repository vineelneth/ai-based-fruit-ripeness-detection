FROM python:3.12-slim

WORKDIR /app

# Ensure Python finds the api/ package from /app regardless of uvicorn's sys.path behavior
ENV PYTHONPATH=/app

# libgomp1: required by TensorFlow for OpenMP threading
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying app code so this layer is cached
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Model artifacts — baked into image for zero-dependency startup
# If the model grows large or changes frequently, move to S3 + download-on-startup
COPY fruit_freshness_regression.keras .
COPY target_scaler.save .

# Application code
COPY api/ api/

# PORT is injected by Render at runtime; default 8000 for local runs
EXPOSE 8000

# 1 worker: each worker loads the full model into RAM (~400–500 MB)
# Scale horizontally (more containers) rather than vertically (more workers)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
