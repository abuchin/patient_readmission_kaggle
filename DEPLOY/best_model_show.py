#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the best model from the Ray Tune experiment and save it as a Docker image.
"""

import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Compute MLflow URI relative to this script's location
# DEPLOY/ -> code/ -> RAY/mlruns
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
MLFLOW_URI = f"file:{CODE_DIR / 'RAY' / 'mlruns'}"

# Point to the same store you used during tuning
mlflow.set_tracking_uri(MLFLOW_URI)

# Experiment name
EXP_NAME = "xgb_diabetic_readmission_hpo"
client = MlflowClient()

# Get experiment
exp = client.get_experiment_by_name(EXP_NAME)
assert exp is not None, f"Experiment {EXP_NAME} not found"

# Find the latest full-train run and get its run_id
runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string='tags.mlflow.runName = "best_model_full_train"',
    order_by=["start_time DESC"],
    max_results=1,
)
assert len(runs) == 1, "No final best_model_full_train run found"
run_id = runs.iloc[0]["run_id"]

# Load the sklearn Pipeline you logged at artifact_path="model"
model_uri = f"runs:/{run_id}/model"
pipe = mlflow.sklearn.load_model(model_uri)

# Show the best pipeline
# y_proba = pipe.predict_proba(X_new)[:, 1]
print("Loaded model from:", model_uri)

# build the docker image