#!/usr/bin/env python3
import os, glob, json, sys, datetime as dt, numpy as np, pandas as pd, mlflow, yaml
from pathlib import Path
from argparse import ArgumentParser
from scipy import stats

# --- resolve paths robustly ---
THIS_FILE = Path(__file__).resolve()            # .../code/MONITOR/scripts/run_monitor_simple.py
MONITOR_DIR = THIS_FILE.parent.parent           # .../code/MONITOR
CODE_DIR = MONITOR_DIR.parent                   # .../code
REPO_ROOT = CODE_DIR.parent                     # .../

def load_cfg():
    ap = ArgumentParser()
    ap.add_argument("--config", default=os.getenv("MONITOR_CONFIG", ""), help="Path to monitoring.yaml")
    args = ap.parse_args()

    if args.config:
        cfg_path = Path(args.config).resolve()
    else:
        cfg_path = (MONITOR_DIR / "configs" / "monitoring.yaml").resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f), cfg_path

CFG, CFG_PATH = load_cfg()

def resolve_path(p: str) -> Path:
    pth = Path(p)
    return pth if pth.is_absolute() else (REPO_ROOT / pth).resolve()

MLFLOW_URI = CFG["mlflow_tracking_uri"]
MLFLOW_EXP = CFG["mlflow_experiment"]
REF_PATH   = resolve_path(CFG["paths"]["reference"])
LOG_DIR    = resolve_path(CFG["paths"]["logs_dir"])
OUT_DIR    = resolve_path(CFG["paths"]["reports_out"])
SHARE_THR  = float(CFG["drift_thresholds"]["share_of_drifted_columns"])
PVALUE_THR = float(CFG["drift_thresholds"]["p_value"])

OUT_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MLFLOW_EXP)
# ---- Helpers ----
def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"): return pd.read_parquet(path)
    if ext in (".feather", ".ft"): return pd.read_feather(path)
    if ext in (".csv", ".txt"):    return pd.read_csv(path)
    if ext in (".pkl", ".pickle"): return pd.read_pickle(path)
    raise ValueError(f"Unsupported reference format: {path}")

def latest_window_path() -> str:
    cands = sorted(glob.glob(os.path.join(LOG_DIR, "preds_*.parquet")))
    if not cands:
        raise FileNotFoundError(f"No production logs in {LOG_DIR}")
    return cands[-1]

def infer_types(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    # treat low-cardinality non-numeric as categorical
    cat = [c for c in df.columns if c not in num and not c.startswith("__")]
    num = [c for c in num if not c.startswith("__")]
    return num, cat

def ks_test(x_ref, x_cur):
    x_ref = x_ref.dropna()
    x_cur = x_cur.dropna()
    if len(x_ref) < 2 or len(x_cur) < 2:
        return np.nan, 1.0
    stat, p = stats.ks_2samp(x_ref, x_cur, alternative="two-sided", mode="auto")
    return float(stat), float(p)

def chi2_test(s_ref, s_cur):
    # align categories
    cats = sorted(set(s_ref.dropna().unique()) | set(s_cur.dropna().unique()))
    ref_counts = s_ref.value_counts().reindex(cats, fill_value=0).values
    cur_counts = s_cur.value_counts().reindex(cats, fill_value=0).values
    # need at least 2 categories with nonzero total
    if (ref_counts.sum() == 0) or (cur_counts.sum() == 0) or len(cats) < 2:
        return np.nan, 1.0
    table = np.vstack([ref_counts, cur_counts])
    stat, p, _, _ = stats.chi2_contingency(table)
    return float(stat), float(p)

def psi_numeric(x_ref, x_cur, bins=10):
    # PSI ~ population stability index
    xr = x_ref.replace([np.inf, -np.inf], np.nan).dropna()
    xc = x_cur.replace([np.inf, -np.inf], np.nan).dropna()
    if len(xr) == 0 or len(xc) == 0:
        return np.nan
    try:
        edges = np.histogram_bin_edges(xr, bins=bins)
        ref_hist, _ = np.histogram(xr, bins=edges)
        cur_hist, _ = np.histogram(xc, bins=edges)
        # add tiny epsilon to avoid div/0
        ref_pct = (ref_hist + 1e-9) / (ref_hist.sum() + 1e-9 * len(ref_hist))
        cur_pct = (cur_hist + 1e-9) / (cur_hist.sum() + 1e-9 * len(cur_hist))
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)
    except Exception:
        return np.nan

def main():
    # Load data
    reference = load_table(REF_PATH)
    current   = pd.read_parquet(latest_window_path())

    # tag/window
    if "__timestamp__" in current.columns and len(current["__timestamp__"]) > 0:
        tag_suffix = str(current["__timestamp__"].iloc[0])[:10]
    else:
        tag_suffix = dt.date.today().isoformat()
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Align columns: restrict to common feature set (ignore __meta__)
    ref_cols = [c for c in reference.columns if not c.startswith("__")]
    cur_cols = [c for c in current.columns   if not c.startswith("__")]
    common = [c for c in ref_cols if c in cur_cols]
    if not common:
        raise RuntimeError("No overlapping feature columns between reference and current logs.")
    R = reference[common].copy()
    C = current[common].copy()

    # Types
    num_cols, cat_cols = infer_types(R)

    # Compute tests
    rows = []
    for c in common:
        if c in num_cols:
            ks_stat, ks_p = ks_test(R[c], C[c])
            psi = psi_numeric(R[c], C[c])
            drifted = (not np.isnan(ks_p) and ks_p < PVALUE_THR) or (not np.isnan(psi) and psi >= 0.2)
            rows.append({
                "feature": c, "type": "numeric",
                "ks_stat": ks_stat, "ks_pvalue": ks_p,
                "psi": psi, "drifted": bool(drifted)
            })
        else:
            chi2_stat, chi2_p = chi2_test(R[c].astype(str), C[c].astype(str))
            drifted = (not np.isnan(chi2_p) and chi2_p < PVALUE_THR)
            rows.append({
                "feature": c, "type": "categorical",
                "chi2_stat": chi2_stat, "chi2_pvalue": chi2_p,
                "psi": np.nan, "drifted": bool(drifted)
            })

    df = pd.DataFrame(rows).sort_values(["drifted","feature"], ascending=[False, True])
    share = float(df["drifted"].mean()) if len(df) else 0.0
    any_p = df.filter(regex="pvalue").min(numeric_only=True).min()
    pval_summary = float(any_p) if pd.notnull(any_p) else 1.0

    alert = (share >= SHARE_THR) or (pval_summary < PVALUE_THR)

    # Save artifacts
    out_csv = os.path.join(OUT_DIR, f"drift_table_{tag_suffix}_{ts}.csv")
    out_json = os.path.join(OUT_DIR, f"drift_summary_{tag_suffix}_{ts}.json")
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "window": tag_suffix,
            "share_of_drifted_columns": share,
            "global_min_pvalue": pval_summary,
            "alert": alert,
            "thresholds": {"share": SHARE_THR, "pvalue": PVALUE_THR}
        }, f, indent=2)

    # Log to MLflow
    with mlflow.start_run(run_name=f"simple_monitor_{tag_suffix}_{ts}"):
        mlflow.log_artifact(out_csv)
        mlflow.log_artifact(out_json)
        mlflow.log_dict(CFG, os.path.join("artifacts", "monitoring_config.yaml"))
        mlflow.log_metric("share_of_drifted_columns", share)
        mlflow.log_metric("global_min_pvalue", pval_summary)
        mlflow.set_tag("monitoring.window", tag_suffix)

    print(f"[Drift] share={share:.3f} min_p={pval_summary:.3g} alert={alert}")
    sys.exit(2 if alert else 0)

if __name__ == "__main__":
    main()
