#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a Docker image for a *re-trained* model from MLflow.
Supports:
  - --model-uri runs:/<run_id>/model
  - --run-id <run_id>
  - --registered-model <name> --stage <Staging|Production|Archived>
  - (fallback) latest run in --experiment that has tag retrain=true or runName=retrained_best

Optionally wraps the model with a logging pyfunc (--with-logging) so each
prediction is written to Parquet (for drift monitoring).

Usage examples:
  # From a specific run id
  python build_retrained_image.py --run-id abc123 --image-tag diabetic-xgb:retrained

  # From a registered model stage
  python build_retrained_image.py --registered-model MediWatchXGB --stage Production --image-tag diabetic-xgb:prod

  # From explicit model uri
  python build_retrained_image.py --model-uri runs:/abc123/model --with-logging --image-tag diabetic-xgb:retrained-logged

  # CI example (pass run id as env var)
  python build_retrained_image.py --run-id $MLFLOW_RUN_ID --no-build   # just exports + dockerfile
"""

import argparse
import os
import sys
import shutil
import subprocess

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

# --- path setup so we can import MONITOR app wrapper -------------------------
THIS_DIR = os.path.abspath(os.path.dirname(__file__))         # .../code/DEPLOY
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))      # .../code
MONITOR_DIR = os.path.join(CODE_DIR, "MONITOR")               # .../code/MONITOR
if MONITOR_DIR not in sys.path:
    sys.path.insert(0, MONITOR_DIR)

# Lazy import; may not exist if user didn't create MONITOR yet
try:
    from app.logged_model import LoggedModel
except Exception:
    LoggedModel = None

DEFAULT_MLFLOW_URI = "file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns"
DEFAULT_EXPERIMENT = "xgb_diabetic_readmission_hpo_retrain"   # you can change it
DEFAULT_OUT_DIR = "./retrained_model"                         # exports here
DEFAULT_IMAGE_TAG = "diabetic-xgb:retrained"
DEFAULT_SERVE_PORT = 5002

DOCKERFILE_TEMPLATE = """# Auto-generated Dockerfile for retrained model
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY model /app/model

RUN python -m pip install --upgrade pip setuptools wheel \\
 && if [ -f /app/model/requirements.txt ]; then \\
      pip install --no-cache-dir -r /app/model/requirements.txt; \\
    else \\
      pip install --no-cache-dir mlflow xgboost scikit-learn pandas numpy scipy pyarrow; \\
    fi

ENV OMP_NUM_THREADS=1 \\
    MKL_NUM_THREADS=1 \\
    PRED_LOG_DIR=/logs
VOLUME ["/logs"]

EXPOSE {port}
CMD ["mlflow", "models", "serve", "-m", "/app/model", "--env-manager", "local", "--host", "0.0.0.0", "--port", "{port}"]
"""

def sh(cmd, cwd=None):
    print(f"+ {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        sys.exit(res.returncode)

def resolve_model_uri(
    client: MlflowClient,
    experiment: str | None,
    model_uri: str | None,
    run_id: str | None,
    registered_model: str | None,
    stage: str | None,
) -> str:
    """Return a canonical MLflow model URI."""
    if model_uri:
        return model_uri

    if run_id:
        return f"runs:/{run_id}/model"

    if registered_model and stage:
        try:
            mv = client.get_latest_versions(registered_model, stages=[stage])
            if not mv:
                print(f"No versions found for registered model '{registered_model}' in stage '{stage}'.")
                sys.exit(1)
            # pick most recent by last_updated_timestamp
            mv = sorted(mv, key=lambda m: m.last_updated_timestamp or 0, reverse=True)[0]
            return f"models:/{registered_model}/{mv.current_stage}"
        except RestException as e:
            print("Model registry error:", e)
            sys.exit(1)

    # Fallback: latest retrain run in experiment
    if experiment:
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            print(f"Experiment not found: {experiment}")
            sys.exit(1)
        # Try tags we expect from retraining
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string='tags.retrain = "true" OR tags.mlflow.runName = "retrained_best"',
            order_by=["start_time DESC"],
            max_results=1,
        )
        if len(runs) == 0:
            print("No retrain run found in experiment. Provide --run-id or --model-uri or --registered-model/--stage.")
            sys.exit(1)
        rid = runs.iloc[0]["run_id"]
        return f"runs:/{rid}/model"

    print("Could not resolve a model. Provide --model-uri or --run-id or --registered-model with --stage.")
    sys.exit(1)

def export_plain_model(model_uri: str, out_dir: str) -> str:
    """Export raw model in MLflow format into out_dir, return path used by Dockerfile COPY."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model = mlflow.sklearn.load_model(model_uri)
    mlflow.sklearn.save_model(sk_model=model, path=out_dir)
    print(f"[plain] Exported MLflow model to: {os.path.abspath(out_dir)}")
    return out_dir  # Docker expects COPY model/ → /app/model

def export_logged_pyfunc(model_uri: str, out_dir: str) -> str:
    """Wrap with LoggedModel and save to out_dir/model (pyfunc). Return path to that dir."""
    global LoggedModel
    if LoggedModel is None:
        from app.logged_model import LoggedModel

    base_model_dir = os.path.join(out_dir, "base_model")
    logged_model_dir = os.path.join(out_dir, "model")
    code_dir = os.path.join(out_dir, "code")

    for p in [base_model_dir, logged_model_dir, code_dir]:
        if os.path.exists(p):
            shutil.rmtree(p)
    os.makedirs(out_dir, exist_ok=True)

    base_model = mlflow.sklearn.load_model(model_uri)
    mlflow.sklearn.save_model(sk_model=base_model, path=base_model_dir)

    # Bundle MONITOR code into the model artifact
    shutil.copytree(os.path.join(MONITOR_DIR, "app"), os.path.join(code_dir, "app"))

    mlflow.pyfunc.save_model(
        path=logged_model_dir,
        python_model=LoggedModel(),
        artifacts={"base_model": base_model_dir},
        code_path=[code_dir],
        pip_requirements=["mlflow","pandas","numpy","scikit-learn","xgboost","pyarrow"],
    )
    print(f"[logged] Saved pyfunc model to: {os.path.abspath(logged_model_dir)}")
    return logged_model_dir

def write_dockerfile(port: int, dockerfile_path: str):
    with open(dockerfile_path, "w") as f:
        f.write(DOCKERFILE_TEMPLATE.format(port=port))
    print(f"Wrote Dockerfile -> {dockerfile_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=DEFAULT_MLFLOW_URI, help="MLflow tracking URI")
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="Fallback experiment for retrain runs")
    ap.add_argument("--model-uri", default=None, help="Direct MLflow model URI (e.g., runs:/<id>/model)")
    ap.add_argument("--run-id", default=None, help="Run ID to package (model path assumed at runs:/<id>/model)")
    ap.add_argument("--registered-model", default=None, help="Registered model name")
    ap.add_argument("--stage", default=None, help="Model registry stage (e.g., Staging, Production)")

    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Export directory (relative to DEPLOY)")
    ap.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG, help="Docker image tag")
    ap.add_argument("--serve-port", type=int, default=DEFAULT_SERVE_PORT, help="Container port")
    ap.add_argument("--no-build", action="store_true", help="Skip docker build (export + Dockerfile only)")
    ap.add_argument("--with-logging", action="store_true", help="Wrap with logging pyfunc (writes Parquet logs)")
    args = ap.parse_args()

    # MLflow client
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Resolve model URI
    model_uri = resolve_model_uri(
        client=client,
        experiment=args.experiment,
        model_uri=args.model_uri,
        run_id=args.run_id,
        registered_model=args.registered_model,
        stage=args.stage,
    )
    print(f"Using model: {model_uri}")

    # Export
    out_dir_abs = os.path.abspath(os.path.join(THIS_DIR, os.path.expanduser(args.out_dir)))
    if args.with_logging:
        docker_src_dir = export_logged_pyfunc(model_uri, out_dir_abs)  # out_dir_abs/model
    else:
        docker_src_dir = export_plain_model(model_uri, out_dir_abs)    # out_dir_abs

    # Ensure DEPLOY/model points to the right files for COPY
    target_model_dir = os.path.join(THIS_DIR, "model")
    if os.path.abspath(docker_src_dir) != os.path.abspath(target_model_dir):
        if os.path.islink(target_model_dir) or os.path.isfile(target_model_dir):
            os.unlink(target_model_dir)
        if os.path.isdir(target_model_dir):
            shutil.rmtree(target_model_dir)
        shutil.copytree(docker_src_dir, target_model_dir)
    else:
        if not os.path.isdir(target_model_dir):
            raise RuntimeError(f"Expected model dir missing: {target_model_dir}")

    # Dockerfile
    dockerfile_path = os.path.join(THIS_DIR, "Dockerfile")
    write_dockerfile(args.serve_port, dockerfile_path)

    # Build
    if not args.no_build:
        sh(f"docker build -t {args.image_tag} .", cwd=THIS_DIR)
        print(f"\nBuilt image: {args.image_tag}")
        print("Run it with:")
        if args.with_logging:
            print(f"  docker run --rm -e PRED_LOG_DIR=/logs -v $(pwd)/logs:/logs -p {args.serve_port}:{args.serve_port} {args.image_tag}\n")
        else:
            print(f"  docker run --rm -p {args.serve_port}:{args.serve_port} {args.image_tag}\n")
        print("POST predictions to:")
        print(f"  curl -X POST http://localhost:{args.serve_port}/invocations \\")
        print('       -H "Content-Type: application/json" \\')
        print('       -d \'{"dataframe_records": [{"feature1": 1.2, "feature2": "A"}]}\' ')

if __name__ == "__main__":
    main()
