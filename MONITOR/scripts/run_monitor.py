import os, json, glob, sys, datetime as dt, yaml, pandas as pd, mlflow
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from evidently.presets import DataDriftPreset, TargetDriftPreset, ClassificationPerformancePreset

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

def main():
    reference = load_table(REF_PATH)
    current   = pd.read_parquet(latest_window())

    tag_suffix = current.get("__timestamp__", pd.Series([dt.date.today().isoformat()])).iloc[0][:10]

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
    report.save_html(paths["report_html"])
    open(paths["report_json"], "w").write(json.dumps(report.as_dict()))
    summary.save_html(paths["sum_html"])
    open(paths["sum_json"], "w").write(json.dumps(summary.as_dict()))

    with mlflow.start_run(run_name=f"monitoring_{tag_suffix}_{ts}"):
        for p in paths.values(): mlflow.log_artifact(p)
        mlflow.set_tag("monitoring.window", tag_suffix)

    sm = json.loads(open(paths["sum_json"]).read())
    try:
        block = next(m for m in sm["metrics"] if m["metric"] == "DatasetDriftMetric")
        res   = block["result"]
        share = float(res["share_of_drifted_columns"])
        pval  = float(res.get("dataset_drift", {}).get("p_value") or res.get("p_value") or 1.0)
    except Exception:
        share, pval = 0.0, 1.0

    alert = (share >= SHARE_THR) or (pval < PVALUE_THR)
    print(f"[Drift] share={share:.3f} p={pval:.3g} alert={alert}")

    flag = f"{OUT_DIR}/drift_alert_{tag_suffix}_{ts}.json"
    open(flag, "w").write(json.dumps({"alert": alert, "share": share, "p_value": pval, "window": tag_suffix}))

    sys.exit(2 if alert else 0)

if __name__ == "__main__":
    main()
