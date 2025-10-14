#!/bin/bash
# Generate a .env file for Airflow docker-compose

cat > .env <<EOF
# Airflow admin credentials
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# Project root (used for mounting)
AIRFLOW_PROJ_DIR=$(pwd)

# MLflow tracking URI (local default)
MLFLOW_TRACKING_URI=file:/opt/airflow/project/RAY/mlruns

# Example secrets (edit as needed)
DOCKER_USERNAME=
DOCKER_PASSWORD=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
HF_TOKEN=
GITHUB_TOKEN=
EOF

echo ".env file created in airflow/.env. Edit this file to customize credentials and secrets."
