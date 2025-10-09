#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drift → Retrain controller (no Evidently).
- Computes PSI + KS on features (baseline vs current)
- Computes PSI + KS on predictions (baseline vs current)
- If any gate trips, retrain on current CSV via your Ray script
- (Optionally) rebuild the serving image from MLflow

Works with: RAY/ray_tune_xgboost.py
"""

import argparse, os, json, sys, time, subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

np.seterr(all="ignore")

# ---------- Helpers ----------
def uri_to_local_path(uri: str) -> str:
    """Convert 'file:/abs/path' or 'file:///abs/path' -> '/abs/path' """
    if not uri:
        return ""
    if uri.startswith("file://"):
        # file:///abs/path  or  file://hostname/...
        # We only support local paths, so strip scheme and possible extra slash
        path = uri[len("file://"):]
        if path.startswith("/"):
            return path
        # fallback
        return "/" + path
    if uri.startswith("file:"):
        return uri[len("file:"):]
    return uri  # already a path

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
        r = requests.post(endpoint, headers={"Content-Type": "text/csv"}, data=f, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return r.text

def _coerce_to_float_list(raw, pred_key: Optional[str] = None, positive_class: Optional[str] = None) -> List[float]:
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            raw = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"Server returned non-JSON text (first 200 chars): {raw[:200]}") from e

    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]

    if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], (int, float))):
        return [float(x) for x in raw]

    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        return [float(row[0]) for row in raw]

    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        key = pred_key
        if key is None:
            for cand in ("pred_proba", "probability", "score", "YES", "1"):
                if cand in raw[0]:
                    key = cand
                    break
        if key is None:
            raise ValueError("Predictions are list of dicts; pass --pred-key (e.g. --pred-key pred_proba).")
        vals = [row[key] for row in raw]
        if len(vals) > 0 and isinstance(vals[0], list):
            vals = [v[0] for v in vals]
        try:
            return [float(v) for v in vals]
        except Exception:
            if positive_class is not None:
                return [1.0 if v == positive_class else 0.0 for v in vals]
            raise TypeError(f"Values under key '{key}' are not numeric; pass --positive-class to map labels to 0/1.")

    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], str):
        if positive_class is None:
            raise ValueError("Server returned string class labels; pass --positive-class to map positive label to 1.0.")
        return [1.0 if x == positive_class else 0.0 for x in raw]

    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], int):
        return [float(x) for x in raw]

    raise TypeError(f"Unsupported prediction payload type: {type(raw)}; sample: {raw[:2] if isinstance(raw, list) else raw}")

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
        raise ValueError(f"Prediction length {len(preds)} != number of rows {len(df)}. Check payload/columns.")
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
    *,
    ignore: Set[str] = frozenset(),
    any_feature_psi_thresh: float = 1.0,
    critical: Set[str] = frozenset(),
    critical_psi_thresh: float = 0.5,
    pred_ks_p_thresh: float = 0.0,
) -> Tuple[bool, dict]:
    common = [c for c in ref.columns if c in cur.columns and c not in ignore and c != pred_col]
    num_cols = [c for c in common if pd.api.types.is_numeric_dtype(ref[c])]
    cat_cols = [c for c in common if c not in num_cols]

    results = []
    drifted = 0
    max_psi = 0.0
    critical_hits = []

    for c in num_cols:
        a = pd.to_numeric(ref[c], errors="coerce").values
        b = pd.to_numeric(cur[c], errors="coerce").values
        psi = psi_numeric(a, b, bins=10)
        ks_p = ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)]).pvalue if np.isfinite(a).any() and np.isfinite(b).any() else 1.0
        max_psi = max(max_psi, psi)
        is_drift = (psi >= feature_psi_thresh) or (ks_p <= ks_p_thresh)
        if c in critical and psi >= critical_psi_thresh:
            is_drift = True
            critical_hits.append(c)
        drifted += int(is_drift)
        results.append({"column": c, "type": "numeric", "psi": round(psi, 4), "ks_p": float(ks_p), "drift": bool(is_drift)})

    for c in cat_cols:
        psi = psi_categorical(ref[c], cur[c])
        max_psi = max(max_psi, psi)
        is_drift = psi >= feature_psi_thresh
        if c in critical and psi >= critical_psi_thresh:
            is_drift = True
            critical_hits.append(c)
        drifted += int(is_drift)
        results.append({"column": c, "type": "categorical", "psi": round(psi, 4), "drift": bool(is_drift)})

    n = max(len(results), 1)
    share = drifted / n

    # prediction drift (PSI + optional KS)
    pred_psi = pred_ks_p = None
    pred_gate = False
    if pred_col in ref.columns and pred_col in cur.columns:
        r = ref[pred_col].values.astype(float)
        c = cur[pred_col].values.astype(float)
        pred_psi = psi_numeric(r, c, bins=10)
        pred_ks_p = ks_2samp(r, c).pvalue if len(r) and len(c) else 1.0
        pred_gate = (pred_psi >= pred_psi_thresh) or (pred_ks_p_thresh > 0 and pred_ks_p <= pred_ks_p_thresh)

    trigger = (
        (share >= drift_share_thresh)
        or (max_psi >= any_feature_psi_thresh)
        or (len(critical_hits) > 0)
        or pred_gate
    )

    summary = {
        "share_drifted": round(share, 4),
        "n_features": n,
        "n_drifted": drifted,
        "max_feature_psi": round(max_psi, 4),
        "any_feature_psi_thresh": any_feature_psi_thresh,
        "critical_hits": sorted(list(set(critical_hits))),
        "critical_psi_thresh": critical_psi_thresh,
        "pred_psi": None if pred_psi is None else round(pred_psi, 4),
        "pred_ks_p": None if pred_ks_p is None else float(pred_ks_p),
        "pred_gate": bool(pred_gate),
        "feature_psi_thresh": feature_psi_thresh,
        "ks_p_thresh": ks_p_thresh,
        "drift_share_thresh": drift_share_thresh,
        "pred_psi_thresh": pred_psi_thresh,
        "pred_ks_p_thresh": pred_ks_p_thresh,
        "details": results,
    }
    return trigger, summary

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # Inputs
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--endpoint", required=True, help="MLflow /invocations URL")
    ap.add_argument("--tmp-dir", default="monitoring/tmp")
    ap.add_argument("--out-dir", default="monitoring/out")
    ap.add_argument("--target-col", default=None, help="optional; passed to retrain script if it needs it")
    ap.add_argument("--prediction-col", default="pred_proba")

    # Prediction parsing
    ap.add_argument("--pred-key", default=None, help="If server returns list of dicts, extract this key (e.g. 'pred_proba').")
    ap.add_argument("--positive-class", default=None, help="If server returns class labels, treat this label as positive (1).")

    # Feature selection & gates
    ap.add_argument("--ignore-cols", default="", help="Comma-separated columns to skip from FEATURE drift (e.g. ids).")
    ap.add_argument("--any-feature-psi-thresh", type=float, default=1.0, help="Trip if ANY feature PSI ≥ this value.")
    ap.add_argument("--critical-cols", default="", help="Comma-separated columns with stricter PSI gate.")
    ap.add_argument("--critical-psi-thresh", type=float, default=0.5, help="PSI threshold for CRITICAL columns.")

    # Drift thresholds (share & predictions)
    ap.add_argument("--feature-psi-thresh", type=float, default=0.2)
    ap.add_argument("--ks-p-thresh", type=float, default=0.01)
    ap.add_argument("--drift-share-thresh", type=float, default=0.30)
    ap.add_argument("--pred-psi-thresh", type=float, default=0.2)
    ap.add_argument("--pred-ks-p-thresh", type=float, default=0.0, help="If >0, also trip when prediction KS p ≤ this.")

    # Retrain (Ray) & deploy
    ap.add_argument("--retrain-script", required=True, help="Path to RAY/ray_tune_xgboost.py")
    ap.add_argument("--tracking-uri", default="", help="MLflow tracking URI (e.g., file:/.../RAY/mlruns)")
    ap.add_argument("--experiment", default="xgb_diabetic_readmission_hpo", help="Experiment name used by builder")
    ap.add_argument("--build-script", default=None, help="optional: DEPLOY/build_docker_image.py")
    ap.add_argument("--image-tag", default="diabetic-xgb:serve")

    # Ray HPO knobs (forwarded to retrain script)
    ap.add_argument("--hpo-num-samples", type=int, default=15, help="Ray Tune num trials")
    ap.add_argument("--hpo-cpus", type=int, default=2)
    ap.add_argument("--hpo-gpus", type=float, default=0.0)
    ap.add_argument("--hpo-test-size", type=float, default=0.2)
    ap.add_argument("--hpo-ray-dir", default="ray_exp_retrain")
    ap.add_argument("--hpo-seed", type=int, default=123)

    # Manual override
    ap.add_argument("--force", action="store_true", help="Force retrain regardless of drift gates.")
    args = ap.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    ref = pd.read_csv(args.baseline)
    cur = pd.read_csv(args.current)

    # Score and attach predictions
    ref_scored = score_and_attach(ref, os.path.join(args.tmp_dir, "ref.csv"), args.endpoint, args.prediction_col, args.pred_key, args.positive_class)
    cur_scored = score_and_attach(cur, os.path.join(args.tmp_dir, "cur.csv"), args.endpoint, args.prediction_col, args.pred_key, args.positive_class)

    # Build ignore / critical sets; always ignore prediction column for FEATURE drift
    ignore = {s.strip() for s in (args.ignore_cols or "").split(",") if s.strip()}
    ignore.add(args.prediction_col)
    critical = {s.strip() for s in (args.critical_cols or "").split(",") if s.strip()}

    # Drift decision
    trigger, summary = detect_drift(
        ref_scored, cur_scored,
        args.feature_psi_thresh, args.ks_p_thresh,
        args.drift_share_thresh, args.pred_psi_thresh,
        args.prediction_col,
        ignore=ignore,
        any_feature_psi_thresh=args.any_feature_psi_thresh,
        critical=critical,
        critical_psi_thresh=args.critical_psi_thresh,
        pred_ks_p_thresh=args.pred_ks_p_thresh,
    )
    if args.force:
        trigger = True
        summary["forced"] = True

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_json = Path(args.out_dir) / f"drift_summary_{ts}.json"
    with open(out_json, "w") as f:
        json.dump({"trigger_retrain": trigger, **summary}, f, indent=2)
    print(json.dumps({"trigger_retrain": trigger, **summary}, indent=2))

    if not trigger:
        return

    # === RETRAIN via your Ray script ===
    env = os.environ.copy()
    if args.tracking_uri:
        env["MLFLOW_TRACKING_URI"] = args.tracking_uri
    mlruns_dir = uri_to_local_path(args.tracking_uri) if args.tracking_uri else ""

    retrain_cmd = [
        sys.executable, args.retrain_script,
        "--data", args.current,
        "--mlruns-dir", mlruns_dir,
        "--num-samples", str(args.hpo_num_samples),
        "--cpus-per-trial", str(args.hpo_cpus),
        "--gpus-per-trial", str(args.hpo_gpus),
        "--test-size", str(args.hpo_test_size),
        "--ray-dir", args.hpo_ray_dir,
        "--seed", str(args.hpo_seed),
    ]
    print("Retraining (Ray):", " ".join(retrain_cmd))
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
        # Optionally: stop/restart your container here.

if __name__ == "__main__":
    main()
