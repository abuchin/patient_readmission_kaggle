#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Large-scale HPO with Ray Tune + MLflow on the diabetic readmission dataset.

- Preprocess: Standardize numeric, OneHotEncode categorical (drop_first)
- Target: merge readmitted labels "<30" and ">30" -> "YES", keep "NO"
- Model: XGBClassifier
- HPO: Ray Tune + ASHA
- Tracking: MLflow (params, metrics, best model, preprocessing pipeline)
- Ray results directory: ./ray_exp

Usage:
    python run_hpo_xgb.py --data /home/ec2-user/projects/patient_selection/data/diabetic_data.csv \
                          --num-samples 50 --gpus-per-trial 0 --cpus-per-trial 4
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score

from xgboost import XGBClassifier

# Ray / Tune
import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

# MLflow
import mlflow
import mlflow.sklearn


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )


def compute_scale_pos_weight(y_binary: np.ndarray) -> float:
    pos = (y_binary == 1).sum()
    neg = (y_binary == 0).sum()
    return float(neg / max(pos, 1))


def trainable(config,
              X_train: pd.DataFrame,
              y_train_num: np.ndarray,
              preprocessor: ColumnTransformer,
              mlruns_uri: str):
    """One Ray Tune trial: 5-fold CV -> report mean metrics."""
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("xgb_diabetic_readmission_hpo")

    with mlflow.start_run(nested=True):
        mlflow.log_params(config)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs, aps, accs, f1s = [], [], [], []
        spw = compute_scale_pos_weight(y_train_num)

        for tr_idx, va_idx in skf.split(X_train, y_train_num):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train_num[tr_idx], y_train_num[va_idx]

            model = XGBClassifier(
                n_estimators=int(config["n_estimators"]),
                max_depth=int(config["max_depth"]),
                learning_rate=float(config["learning_rate"]),
                subsample=float(config["subsample"]),
                colsample_bytree=float(config["colsample_bytree"]),
                min_child_weight=float(config["min_child_weight"]),
                reg_alpha=float(config["reg_alpha"]),
                reg_lambda=float(config["reg_lambda"]),
                scale_pos_weight=spw,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )

            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
            pipe.fit(X_tr, y_tr)

            y_va_proba = pipe.predict_proba(X_va)[:, 1]
            y_va_pred = (y_va_proba >= 0.5).astype(int)

            aucs.append(roc_auc_score(y_va, y_va_proba))
            aps.append(average_precision_score(y_va, y_va_proba))
            accs.append(accuracy_score(y_va, y_va_pred))
            f1s.append(f1_score(y_va, y_va_pred))

        metrics = {
            "val_auc": float(np.mean(aucs)),
            "val_ap": float(np.mean(aps)),
            "val_acc": float(np.mean(accs)),
            "val_f1": float(np.mean(f1s)),
        }
        mlflow.log_metrics(metrics)
        session.report(metrics=metrics)


def fit_best_and_log(X_train, y_train_num, X_test, y_test_num, preprocessor, best_config, mlruns_uri: str):
    """Fit best pipeline on full TRAIN, eval on TEST, log to MLflow."""
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("xgb_diabetic_readmission_hpo")

    spw = compute_scale_pos_weight(y_train_num)

    best_model = XGBClassifier(
        n_estimators=int(best_config["n_estimators"]),
        max_depth=int(best_config["max_depth"]),
        learning_rate=float(best_config["learning_rate"]),
        subsample=float(best_config["subsample"]),
        colsample_bytree=float(best_config["colsample_bytree"]),
        min_child_weight=float(best_config["min_child_weight"]),
        reg_alpha=float(best_config["reg_alpha"]),
        reg_lambda=float(best_config["reg_lambda"]),
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)])

    with mlflow.start_run(run_name="best_model_full_train"):
        mlflow.log_params(best_config)
        pipe.fit(X_train, y_train_num)

        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        test_metrics = {
            "test_auc": roc_auc_score(y_test_num, y_proba),
            "test_ap": average_precision_score(y_test_num, y_proba),
            "test_acc": accuracy_score(y_test_num, y_pred),
            "test_f1": f1_score(y_test_num, y_pred),
        }
        mlflow.log_metrics({k: float(v) for k, v in test_metrics.items()})

        mlflow.sklearn.log_model(sk_model=pipe, artifact_path="model", registered_model_name=None)

        cfg_path = "best_config.json"
        with open(cfg_path, "w") as f:
            json.dump(best_config, f, indent=2)
        mlflow.log_artifact(cfg_path)

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to diabetic_data.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-samples", type=int, default=30, help="Number of Ray Tune trials")
    parser.add_argument("--cpus-per-trial", type=int, default=2)
    parser.add_argument("--gpus-per-trial", type=float, default=0.0)
    parser.add_argument("--ray-dir", type=str, default="ray_exp", help="Directory for Ray Tune outputs")
    parser.add_argument("--mlruns-dir", type=str,
                        default="/home/ec2-user/projects/patient_selection/code/RAY/mlruns",
                        help="Absolute path for MLflow backend store")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Central MLflow store (absolute URI) – make Ray workers inherit it
    mlruns_abs = os.path.abspath(args.mlruns_dir)
    os.makedirs(mlruns_abs, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{mlruns_abs}"
    mlflow.set_tracking_uri(f"file:{mlruns_abs}")

    # Data
    data = pd.read_csv(args.data)
    y = data["readmitted"].replace({">30": "YES", "<30": "YES", "NO": "NO"})
    X = data.drop("readmitted", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=args.test_size, random_state=args.seed
    )
    y_train_num = (y_train == "YES").astype(int).to_numpy()
    y_test_num = (y_test == "YES").astype(int).to_numpy()

    preprocessor = build_preprocessor(X_train)

    # Ray
    ray.init(ignore_reinit_error=True)

    search_space = {
        "n_estimators": tune.randint(200, 900),
        "max_depth": tune.randint(3, 11),
        "learning_rate": tune.loguniform(1e-3, 3e-1),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "min_child_weight": tune.loguniform(1e-1, 1e1),
        "reg_alpha": tune.loguniform(1e-8, 1e-1),
        "reg_lambda": tune.loguniform(1e-2, 1e1),
    }

    scheduler = ASHAScheduler(grace_period=1, max_t=1, reduction_factor=2)

    run_config = RunConfig(
        name="xgb_hpo",
        storage_path=os.path.abspath(args.ray_dir),
        verbose=1,
    )

    try:
        from ray.tune.search.optuna import OptunaSearch
        search_alg = OptunaSearch(metric="val_auc", mode="max")
    except Exception:
        search_alg = None

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                trainable,
                X_train=X_train,
                y_train_num=y_train_num,
                preprocessor=preprocessor,
                mlruns_uri=f"file:{mlruns_abs}",
            ),
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="val_auc",
            mode="max",
            scheduler=scheduler,
            num_samples=args.num_samples,
            search_alg=search_alg,
        ),
        run_config=run_config,
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_auc", mode="max")
    best_config = best_result.config

    os.makedirs(args.ray_dir, exist_ok=True)
    with open(os.path.join(args.ray_dir, "best_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)

    print("\nBest hyperparameters:\n", json.dumps(best_config, indent=2))

    test_metrics = fit_best_and_log(
        X_train=X_train,
        y_train_num=y_train_num,
        X_test=X_test,
        y_test_num=y_test_num,
        preprocessor=preprocessor,
        best_config=best_config,
        mlruns_uri=f"file:{mlruns_abs}",
    )

    print("\nTest metrics with best config:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nRay results saved under: {os.path.abspath(args.ray_dir)}")
    print(f"MLflow runs saved under: {mlruns_abs}")
    print(f"Launch UI: mlflow ui --backend-store-uri file:{mlruns_abs} --host 127.0.0.1 --port 5000")
    ray.shutdown()


if __name__ == "__main__":
    main()