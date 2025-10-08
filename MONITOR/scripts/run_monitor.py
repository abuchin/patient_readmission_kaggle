import os, json, glob, sys, datetime as dt, yaml, pandas as pd, mlflow

# --- Evidently compatibility imports (works for 0.7.x and 0.4.x) ---
try:
    from evidently import Report
    from evidently.metrics import DatasetDriftMetric
    try:
        from evidently.presets import (
            DataDriftPreset,
            TargetDriftPreset,
            ClassificationPerformancePreset,
        )
    except ImportError:
        from evidently.metric_preset import (
            DataDriftPreset,
            TargetDriftPreset,
            ClassificationPerformancePreset,
        )
except ImportError:
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric
    from evidently.presets import (
        DataDriftPreset,
        TargetDriftPreset,
        ClassificationPerformancePreset,
    )

CFG = yaml.safe_load(open("code/MONITOR/configs/monitoring.yaml"))
REF_PATH   = CFG["paths"]["reference"]
LOG_DIR    = CFG["paths"]["logs_dir"]
OUT_DIR    = CFG["paths"]["reports_out"]
SHARE_THR  = CFG["drift_thresholds"]["share_of_drifted_columns"]
PVALUE_THR = CFG["drift_thresholds"]["p_value"]

mlflow.set_tracking_uri(CFG["mlflow_tracking_uri"])
mlflow.set_experiment(CFG["mlflow_experiment"])
os.makedirs(OUT_DIR, exist_ok=True)

def load_table(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]: return pd.read_parquet(path)
    if ext in [".feather", ".ft"]: return pd.read_feather(path)
    if ext in [".csv", ".txt"]:    return pd.read_csv(path)
    if ext in [".pkl", ".pickle"]: return pd.read_pickle(path)
    raise ValueError(f"Unsupported reference format: {path}")

def latest_window():
    cands = sorted(glob.glob(os.path.join(LOG_DIR, "preds_*.parquet")))
    if not cands:
        raise FileNotFoundError(f"No production logs in {LOG_DIR}")
    return cands[-1]

def has_labels(df):
    return "__target__" in df.columns and df["__target__"].notna().any()

def save_html_safe(rep: Report, path: str):
    try:
        rep.save_html(path)
    except Exception:
        # Fallback: at least persist JSON alongside
        with open(path.replace(".html", ".json"), "w") as f:
            json.dump(rep.as_dict(), f)

def main():
    reference = load_table(REF_PATH)
    current   = pd.read_parquet(latest_window())

    # robust tag (works whether __timestamp__ exists or not)
    if "__timestamp__" in current.columns and len(current["__timestamp__"]) > 0:
        tag_suffix = str(current["__timestamp__"].iloc[0])[:10]
    else:
        tag_suffix = dt.date.today().isoformat()

    suites = [DataDriftPreset()]
    if has_labels(reference) and has_labels(current):
        suites += [TargetDriftPreset(), ClassificationPerformancePreset()]

    report = Report(suites)
    report.run(reference_data=reference, current_data=current)

    feature_cols = [c for c in reference.columns if not c.startswith("__")]
    summary = Report([DatasetDriftMetric()])
    summary.run(reference_data=reference[feature_cols], current_data=current[feature_cols])

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    paths = {
        "report_html": f"{OUT_DIR}/evidently_report_{tag_suffix}_{ts}.html",
        "report_json": f"{OUT_DIR}/evidently_report_{tag_suffix}_{ts}.json",
        "sum_html":    f"{OUT_DIR}/evidently_summary_{tag_suffix}_{ts}.html",
        "sum_json":    f"{OUT_DIR}/evidently_summary_{tag_suffix}_{ts}.json",
    }

    # Save artifacts (tolerant to minor API differences)
    save_html_safe(report,  paths["report_html"])
    with open(paths["report_json"], "w") as f: json.dump(report.as_dict(), f)
    save_html_safe(summary, paths["sum_html"])
    with open(paths["sum_json"], "w") as f: json.dump(summary.as_dict(), f)

    with mlflow.start_run(run_name=f"monitoring_{tag_suffix}_{ts}"):
        for p in paths.values(): mlflow.log_artifact(p)
        mlflow.set_tag("monitoring.window", tag_suffix)

    # Parse thresholds (handle small schema differences across versions)
    sm = json.loads(open(paths["sum_json"]).read())
    try:
        block = next(m for m in sm["metrics"] if m.get("metric") == "DatasetDriftMetric")
        res   = block["result"]
        # prefer exact key, fall back to nested locations if needed
        share = float(res.get("share_of_drifted_columns") or 0.0)
        pval  = res.get("dataset_drift", {}).get("p_value")
        if pval is None: pval = res.get("p_value", 1.0)
        pval = float(pval)
    except Exception:
        share, pval = 0.0, 1.0

    alert = (share >= SHARE_THR) or (pval < PVALUE_THR)
    print(f"[Drift] share={share:.3f} p={pval:.3g} alert={alert}")

    flag = f"{OUT_DIR}/drift_alert_{tag_suffix}_{ts}.json"
    with open(flag, "w") as f:
        json.dump({"alert": alert, "share": share, "p_value": pval, "window": tag_suffix}, f)

    sys.exit(2 if alert else 0)

if __name__ == "__main__":
    main()
