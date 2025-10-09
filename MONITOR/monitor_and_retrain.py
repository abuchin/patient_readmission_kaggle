#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drift → Retrain controller (no Evidently).
- Computes PSI + KS on features (baseline vs current)
- Computes PSI on predictions (baseline preds vs current preds)
- If any gate trips, retrain on current CSV and (optionally) rebuild the serving image.

Examples:
  python MONITOR/monitor_and_retrain.py \
    --baseline data/baseline_val.csv \
    --current  /home/ec2-user/projects/patient_selection/data/diabetic_data_drift.csv \
    --endpoint http://localhost:5001/invocations \
    --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
    --experiment xgb_diabetic_readmission_hpo \
    --retrain-script ml/train.py \
    --build-script api/build_docker_image.py \
    --image-tag diabetic-xgb:serve \
    --pred-key pred_proba

  # If server returns class labels like ["YES","NO",...]
  python MONITOR/monitor_and_retrain.py ... --positive-class YES
"""

import argparse, os, json, sys, time, subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

np.seterr(all="ignore")


# ---------- Drift metrics ----------
def psi_numeric(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 20 or len(b) < 20:
        return 0.0
    q = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(a, q))
    if len(cuts) < 3:
        return 0.0
    ha, _ = np.histogram(a, bins=cuts)
    hb, _ = np.histogram(b, bins=cuts)
    pa = np.clip(ha / max(ha.sum(), 1), 1e-6, None)
    pb = np.clip(hb / max(hb.sum(), 1), 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def psi_categorical(a: pd.Series, b: pd.Series) -> float:
    va = a.astype("object").value_counts(normalize=True)
    vb = b.astype("object").value_counts(normalize=True)
    cats = sorted(set(va.index).union(vb.index))
    pa = np.array([va.get(c, 0.0) for c in cats]) + 1e-6
    pb = np.array([vb.get(c, 0.0) for c in cats]) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


# ---------- Prediction IO ----------
def post_csv(endpoint: str, csv_path: str, timeout: int = 60):
    with open(csv_path, "rb") as f:
        r = requests.post(
            endpoint,
            headers={"Content-Type": "text/csv"},
            data=f,
            timeout=timeout,
        )
    r.raise_for_status()
    # Return parsed JSON OR raw text (we’ll coerce later)
    try:
        return r.json()
    except Exception:
        return r.text


def _coerce_to_float_list(
    raw, pred_key: Optional[str] = None, positive_class: Optional[str] = None
) -> List[float]:
    """
    Accept various MLflow/pyfunc outputs and return a flat list[float].
    Supported:
      - [0.1, 0.8, ...]                          (list of numbers)
      - [[0.1], [0.8], ...]                      (list of single-element lists)
      - {"predictions": [...]}                   (mlflow wrapper)
      - [{"pred_proba": 0.1}, {"pred_proba": 0.8}, ...] (list of dicts)
      - ["YES", "NO", ...] / [1, 0, ...]        (class labels) -> map with positive_class or numeric cast
      - raw JSON string
    """
    # If text, try to json-load
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            raw = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"Server returned non-JSON text (first 200 chars): {raw[:200]}") from e

    # Unwrap {"predictions":[...]}
    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]

    # list of floats or ints
    if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], (int, float))):
        return [float(x) for x in raw]

    # list of single-element lists
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        # assume each row holds a single value
        return [float(row[0]) for row in raw]

    # list of dicts -> need key
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        key = pred_key
        if key is None:
            # Try common candidates automatically
            for cand in ("pred_proba", "probability", "score", "YES", "1"):
                if cand in raw[0]:
                    key = cand
                    break
        if key is None:
            raise ValueError(
                "Predictions are list of dicts; pass --pred-key to specify which key to extract (e.g. --pred-key pred_proba)."
            )
        vals = [row[key] for row in raw]
        # if they are lists inside dicts
        if len(vals) > 0 and isinstance(vals[0], list):
            vals = [v[0] for v in vals]
        try:
            return [float(v) for v in vals]
        except Exception:
            if positive_class is not None:
                return [1.0 if v == positive_class else 0.0 for v in vals]
            raise TypeError(
                f"Values under key '{key}' are not numeric; pass --positive-class to map labels to 0/1."
            )

    # list of strings (class labels)
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], str):
        if positive_class is None:
            raise ValueError(
                "Server returned string class labels; pass --positive-class to map positive label to 1.0."
            )
        return [1.0 if x == positive_class else 0.0 for x in raw]

    # list of ints (class ids)
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], int):
        # assume already 0/1
        return [float(x) for x in raw]

    raise TypeError(
        f"Unsupported prediction payload type: {type(raw)}; sample: {raw[:2] if isinstance(raw, list) else raw}"
    )


def score_and_attach(
    df: pd.DataFrame,
    tmp_csv: str,
    endpoint: str,
    pred_col: str,
    pred_key: Optional[str],
    positive_class: Optional[str],
) -> pd.DataFrame:
    df.to_csv(tmp_csv, index=False)
    raw = post_csv(endpoint, tmp_csv)
    preds = _coerce_to_float_list(raw, pred_key=pred_key, positive_class=positive_class)
    if len(preds) != len(df):
        raise ValueError(
            f"Prediction length {len(preds)} != number of rows {len(df)}. Check your payload/columns."
        )
    out = df.copy()
    out[pred_col] = np.asarray(preds, dtype=float)
    return out


# ---------- Drift decision ----------
def detect_drift(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    feature_psi_thresh: float,
    ks_p_thresh: float,
    drift_share_thresh: float,
    pred_psi_thresh: float,
    pred_col: str,
) -> Tuple[bool, dict]:
    common = [c for c in ref.columns if c in cur.columns]
    num_cols = [c for c in common if pd.api.types.is_numeric_dtype(ref[c])]
    cat_cols = [c for c in common if c not in num_cols and c != pred_col]

    results = []
    drifted = 0

    for c in num_cols:
        a = pd.to_numeric(ref[c], errors="coerce").values
        b = pd.to_numeric(cur[c], errors="coerce").values
        psi = psi_numeric(a, b, bins=10)
        ks_p = (
            ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)]).pvalue
            if np.isfinite(a).any() and np.isfinite(b).any()
            else 1.0
        )
        is_drift = (psi >= feature_psi_thresh) or (ks_p <= ks_p_thresh)
        drifted += int(is_drift)
        results.append(
            {"column": c, "type": "numeric", "psi": round(psi, 4), "ks_p": float(ks_p), "drift": bool(is_drift)}
        )

    for c in cat_cols:
        psi = psi_categorical(ref[c], cur[c])
        is_drift = psi >= feature_psi_thresh
        drifted += int(is_drift)
        results.append({"column": c, "type": "categorical", "psi": round(psi, 4), "drift": bool(is_drift)})

    n = max(len(results), 1)
    share = drifted / n

    # prediction distribution drift
    pred_psi = None
    pred_gate = False
    if pred_col in ref.columns and pred_col in cur.columns:
        pred_psi = psi_numeric(ref[pred_col].values.astype(float), cur[pred_col].values.astype(float), bins=10)
        pred_gate = pred_psi >= pred_psi_thresh

    trigger = (share >= drift_share_thresh) or pred_gate
    summary = {
        "share_drifted": round(share, 4),
        "n_features": n,
        "n_drifted": drifted,
        "pred_psi": None if pred_psi is None else round(pred_psi, 4),
        "pred_gate": bool(pred_gate),
        "feature_psi_thresh": feature_psi_thresh,
        "ks_p_thresh": ks_p_thresh,
        "drift_share_thresh": drift_share_thresh,
        "pred_psi_thresh": pred_psi_thresh,
        "details": results,
    }
    return trigger, summary


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--endpoint", required=True, help="MLflow /invocations URL")
    ap.add_argument("--tmp-dir", default="monitoring/tmp")
    ap.add_argument("--out-dir", default="monitoring/out")
    ap.add_argument("--target-col", default=None, help="optional; passed to retrain script if it needs it")
    ap.add_argument("--prediction-col", default="pred_proba")

    # Robust prediction parsing
    ap.add_argument("--pred-key", default=None, help="If server returns list of dicts, extract this key (e.g. 'pred_proba').")
    ap.add_argument("--positive-class", default=None, help="If server returns class labels, treat this label as positive (1).")

    # Drift thresholds
    ap.add_argument("--feature-psi-thresh", type=float, default=0.2)
    ap.add_argument("--ks-p-thresh", type=float, default=0.01)
    ap.add_argument("--drift-share-thresh", type=float, default=0.30)
    ap.add_argument("--pred-psi-thresh", type=float, default=0.2)

    # Retrain & deploy
    ap.add_argument("--retrain-script", required=True, help="e.g., ml/train.py")
    ap.add_argument("--tracking-uri", default="")
    ap.add_argument("--experiment", default="mediwatch_train")
    ap.add_argument("--build-script", default=None, help="optional: api/build_docker_image.py")
    ap.add_argument("--image-tag", default="diabetic-xgb:serve")
    args = ap.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    ref = pd.read_csv(args.baseline)
    cur = pd.read_csv(args.current)

    # Score and attach predictions
    ref_scored = score_and_attach(
        ref, os.path.join(args.tmp_dir, "ref.csv"), args.endpoint,
        args.prediction_col, args.pred_key, args.positive_class
    )
    cur_scored = score_and_attach(
        cur, os.path.join(args.tmp_dir, "cur.csv"), args.endpoint,
        args.prediction_col, args.pred_key, args.positive_class
    )

    # Drift decision
    trigger, summary = detect_drift(
        ref_scored, cur_scored,
        args.feature_psi_thresh, args.ks_p_thresh,
        args.drift_share_thresh, args.pred_psi_thresh,
        args.prediction_col,
    )

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_json = Path(args.out_dir) / f"drift_summary_{ts}.json"
    with open(out_json, "w") as f:
        json.dump({"trigger_retrain": trigger, **summary}, f, indent=2)
    print(json.dumps({"trigger_retrain": trigger, **summary}, indent=2))

    if not trigger:
        return

    # === RETRAIN ===
    env = os.environ.copy()
    if args.tracking_uri:
        env["MLFLOW_TRACKING_URI"] = args.tracking_uri

    retrain_cmd = [
        sys.executable, args.retrain_script,
        "--data", args.current,
        "--experiment", args.experiment
    ]
    if args.target_col:
        retrain_cmd += ["--target", args.target_col]

    print("Retraining:", " ".join(retrain_cmd))
    subprocess.check_call(retrain_cmd, env=env)

    # === (optional) REBUILD + REDEPLOY ===
    if args.build_script:
        build_cmd = [
            sys.executable, args.build_script,
            "--tracking-uri", args.tracking_uri,
            "--experiment", args.experiment,
            "--image-tag", args.image_tag,
            "--serve-port", "5001",
        ]
        print("Building image:", " ".join(build_cmd))
        subprocess.check_call(build_cmd, env=env)
        # Optionally: stop & start container here.


if __name__ == "__main__":
    main()
