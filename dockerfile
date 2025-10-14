# Dockerfile
FROM python:3.13.5

# Make Python/pip friendlier in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps often needed by scientific Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.lock.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Now copy your code
COPY . /app

# (Optional) create non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# ===== Choose ONE default command =====
# 1) If you have a FastAPI app (module:app), uncomment:
# EXPOSE 8000
# CMD ["uvicorn", "your_module:app", "--host", "0.0.0.0", "--port", "8000"]

# 2) If you want a Jupyter server instead, uncomment:
# EXPOSE 8888
# CMD ["python", "-m", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token="]

# 3) Or just run a script (replace with your entry point):
CMD ["python", "main.py"]
