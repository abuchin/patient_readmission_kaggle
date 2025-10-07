#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_MLFLOW_URI = "file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns"
DEFAULT_EXP_NAME = "xgb_diabetic_readmission_hpo"
DEFAULT_OUT_DIR = "./model"
DEFAULT_IMAGE_TAG = "diabetic-xgb:serve"
DEFAULT_SERVE_PORT = 5001  # 5000 is in use

DOCKERFILE_TEMPLATE = """# Auto-generated Dockerfile
# Use a Python version compatible with numpy>=2.3 (requires >=3.11)
FROM python:3.11-slim

# xgboost runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy exported MLflow model dir
COPY model /app/model

# Install deps from the model's own requirements captured by MLflow
RUN python -m pip install --upgrade pip setuptools wheel \\
 && if [ -f /app/model/requirements.txt ]; then \\
      pip install --no-cache-dir -r /app/model/requirements.txt; \\
    else \\
      pip install --no-cache-dir mlflow xgboost scikit-learn pandas numpy scipy; \\
    fi

# Avoid noisy over-threading in containers
ENV OMP_NUM_THREADS=1 \\
    MKL_NUM_THREADS=1

EXPOSE {port}
CMD ["mlflow", "models", "serve", "-m", "/app/model", "--env-manager", "local", "--host", "0.0.0.0", "--port", "{port}"]
"""

def sh(cmd, cwd=None):
    print(f"+ {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=DEFAULT_MLFLOW_URI,
                    help="MLflow tracking URI used during training (file:/.../mlruns)")
    ap.add_argument("--experiment", default=DEFAULT_EXP_NAME,
                    help="MLflow experiment name")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Where to export the MLflow model directory (Docker build context expects ./model)")
    ap.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG,
                    help="Docker image tag to build")
    ap.add_argument("--serve-port", type=int, default=DEFAULT_SERVE_PORT,
                    help="Container port for mlflow models serve (default: 5001)")
    ap.add_argument("--no-build", action="store_true",
                    help="Only export model & write Dockerfile; skip docker build")
    args = ap.parse_args()

    # 1) MLflow store
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # 2) Find latest final run
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"Experiment not found: {args.experiment}")
        sys.exit(1)

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string='tags.mlflow.runName = "best_model_full_train"',
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        print("No run named 'best_model_full_train' found.")
        sys.exit(1)

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    print(f"Using model: {model_uri}")

    # 3) Export model locally (MLflow writes model/requirements.txt & python_env.yaml)
    os.makedirs(args.out_dir, exist_ok=True)
    ml_model = mlflow.sklearn.load_model(model_uri)
    mlflow.sklearn.save_model(sk_model=ml_model, path=args.out_dir)
    print(f"Exported model to: {os.path.abspath(args.out_dir)}")

    # 4) Write Dockerfile with chosen port
    dockerfile_path = os.path.join(os.getcwd(), "Dockerfile")
    with open(dockerfile_path, "w") as f:
        f.write(DOCKERFILE_TEMPLATE.format(port=args.serve_port))
    print(f"Wrote Dockerfile -> {dockerfile_path}")

    # 5) Build image (optional)
    if not args.no_build:
        sh(f"docker build -t {args.image_tag} .")
        print(f"\nBuilt image: {args.image_tag}")
        print("Run it with:")
        print(f"  docker run --rm -p {args.serve_port}:{args.serve_port} {args.image_tag}")
        print("\nPOST predictions to:")
        print(f"  curl -X POST http://localhost:{args.serve_port}/invocations \\")
        print('       -H "Content-Type: application/json" \\')
        print('       -d \'{"dataframe_records": [{"feature1": 1.2, "feature2": "A"}]}\' ')

if __name__ == "__main__":
    main()
