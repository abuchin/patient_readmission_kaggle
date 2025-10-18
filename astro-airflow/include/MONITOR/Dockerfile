# Auto-generated Dockerfile
# Use a Python version compatible with numpy>=2.3 (requires >=3.11)
FROM python:3.11-slim

# xgboost runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy exported MLflow model dir
COPY model /app/model

# Install deps from the model's own requirements captured by MLflow
RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ -f /app/model/requirements.txt ]; then \
      pip install --no-cache-dir -r /app/model/requirements.txt; \
    else \
      pip install --no-cache-dir mlflow xgboost scikit-learn pandas numpy scipy; \
    fi

# Avoid noisy over-threading in containers
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 5001
CMD ["mlflow", "models", "serve", "-m", "/app/model", "--env-manager", "local", "--host", "0.0.0.0", "--port", "5001"]
