# MONITOR - Model Monitoring & Drift Detection

## Overview

The MONITOR component provides comprehensive **drift detection** and **automated retraining** capabilities for the diabetic readmission prediction model. It continuously monitors production model performance and data quality, automatically triggering retraining when drift is detected.

This component is the **final stage** in the MLOps pipeline, closing the feedback loop from production back to training.

---

## 📋 Table of Contents

- [Components Overview](#components-overview)
- [Monitoring Approaches](#monitoring-approaches)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Drift Detection Logic](#drift-detection-logic)
- [Automated Retraining](#automated-retraining)
- [Integration with Pipeline](#integration-with-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Components Overview

```
MONITOR/
├── monitor_and_retrain.py      # ⭐ Main: Drift detection + auto-retraining (337 lines)
├── monitoring/
│   ├── out/                    # Drift detection outputs (JSON summaries)
│   └── tmp/                    # Temporary scoring files
└── reports/                    # Evidently HTML/JSON reports
```

### Component Descriptions

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **monitor_and_retrain.py** | End-to-end drift detection & retraining orchestration | Baseline CSV, Current CSV, Endpoint | Drift report, Retrained model |
| **run_monitor.py** | Evidently-based monitoring with presets | Reference data, Production logs | HTML/JSON reports, MLflow artifacts |
| **run_monitor_simple.py** | Statistical tests (KS, Chi², PSI) | Reference data, Production logs | CSV table, JSON summary |
| **logged_model.py** | Prediction logging wrapper | Model predictions | Parquet logs by date |
| **log_utils.py** | Utilities for appending predictions | Features + predictions | Daily Parquet files |

---

## Monitoring Approaches

The MONITOR component offers **three complementary approaches** to drift detection:

### 1. **Full Monitoring with Evidently** (`run_monitor.py`)

**Best for**: Comprehensive analysis with visualizations

**Features**:
- ✅ Data drift detection (distribution shifts)
- ✅ Target drift detection (label distribution changes)
- ✅ Model performance monitoring (when labels available)
- ✅ Interactive HTML reports with plots
- ✅ MLflow integration for historical tracking

**Use when**: You want detailed visual reports and comprehensive drift analysis

**Thresholds**:
- Share of drifted columns: 30% (configurable in `monitoring.yaml`)
- Dataset-level p-value: 0.05

### 2. **Simple Statistical Monitoring** (`run_monitor_simple.py`)

**Best for**: Lightweight, dependency-light monitoring

**Features**:
- ✅ KS test for numeric features (distribution comparison)
- ✅ Chi-squared test for categorical features
- ✅ PSI (Population Stability Index) for drift magnitude
- ✅ CSV output for programmatic processing
- ✅ No heavy dependencies (only scipy)

**Use when**: You need fast, lightweight monitoring or want to customize drift logic

**Thresholds**:
- PSI threshold: 0.2 (industry standard)
- p-value threshold: 0.05

### 3. **Automated Drift → Retrain Pipeline** (`monitor_and_retrain.py`) ⭐

**Best for**: Production automation with closed-loop retraining

**Features**:
- ✅ Real-time prediction scoring via API endpoint
- ✅ Feature drift detection (PSI + KS tests)
- ✅ Prediction distribution drift detection
- ✅ Automatic retraining trigger
- ✅ Optional Docker image rebuild
- ✅ Robust prediction parsing (handles various formats)

**Use when**: You want fully automated monitoring → retraining → deployment pipeline

**Thresholds** (all configurable):
- Feature PSI threshold: 0.2
- KS p-value threshold: 0.01
- Share of drifted features: 30%
- Prediction PSI threshold: 0.2

---

## Quick Start

### Option 1: Evidently Monitoring

```bash
cd code/MONITOR/
python scripts/run_monitor.py
```

**Prerequisites**:
- Configure `configs/monitoring.yaml` with correct paths
- Production prediction logs in Parquet format
- Reference data (training baseline)

**Output**:
- `reports/evidently_report_*.html` - Full interactive report
- `reports/evidently_summary_*.html` - Quick summary
- `reports/drift_alert_*.json` - Alert status
- MLflow experiment: "MediWatch-Monitoring"

### Option 2: Simple Statistical Monitoring

```bash
cd code/MONITOR/
python scripts/run_monitor_simple.py
```

**Output**:
- `reports/drift_table_*.csv` - Per-feature drift metrics
- `reports/drift_summary_*.json` - Overall drift summary
- Exit code 2 if drift detected (for automation)

### Option 3: Automated Drift → Retrain

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
  --experiment xgb_diabetic_readmission_hpo \
  --build-script code/DEPLOY/build_docker_image.py \
  --image-tag diabetic-xgb:serve
```

**What happens**:
1. Loads baseline and current datasets
2. Scores both through your deployed model endpoint
3. Computes drift metrics (PSI, KS test)
4. **If drift detected**:
   - Automatically triggers retraining script
   - Optionally rebuilds Docker image
   - Logs all metrics to MLflow

---

## Detailed Usage

### `monitor_and_retrain.py` - Main Monitoring Pipeline

#### Command Line Arguments

```bash
python monitor_and_retrain.py [OPTIONS]
```

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--baseline` | str | ✅ Yes | - | Baseline reference data (CSV) |
| `--current` | str | ✅ Yes | - | Current production data (CSV) |
| `--endpoint` | str | ✅ Yes | - | Model serving endpoint (e.g., http://localhost:5001/invocations) |
| `--retrain-script` | str | ✅ Yes | - | Path to training script (e.g., RAY/ray_tune_xgboost.py) |
| `--tmp-dir` | str | No | `monitoring/tmp` | Temporary directory for scoring |
| `--out-dir` | str | No | `monitoring/out` | Output directory for drift reports |
| `--prediction-col` | str | No | `pred_proba` | Name of prediction column |
| `--pred-key` | str | No | `None` | Extract this key from dict responses |
| `--positive-class` | str | No | `None` | Positive class label for categorical predictions |
| `--feature-psi-thresh` | float | No | `0.2` | PSI threshold for feature drift |
| `--ks-p-thresh` | float | No | `0.01` | KS test p-value threshold |
| `--drift-share-thresh` | float | No | `0.30` | Share of features that must drift to trigger |
| `--pred-psi-thresh` | float | No | `0.2` | PSI threshold for prediction drift |
| `--tracking-uri` | str | No | `""` | MLflow tracking URI for retraining |
| `--experiment` | str | No | `mediwatch_train` | MLflow experiment name |
| `--build-script` | str | No | `None` | Optional: Path to Docker build script |
| `--image-tag` | str | No | `diabetic-xgb:serve` | Docker image tag |
| `--target-col` | str | No | `None` | Target column name (passed to retrain script) |

#### Examples

**Basic drift detection (no retraining)**:
```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/X_test.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Full automated pipeline with Docker rebuild**:
```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
  --experiment xgb_diabetic_readmission_hpo \
  --build-script code/DEPLOY/build_docker_image.py \
  --image-tag diabetic-xgb:serve-v2
```

**Handling dict predictions with specific key**:
```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/baseline.csv \
  --current data/production_batch.csv \
  --endpoint http://prod-api:8080/predict \
  --pred-key probability \
  --retrain-script ml/train.py
```

**Handling categorical class predictions**:
```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/baseline.csv \
  --current data/production_batch.csv \
  --endpoint http://localhost:5001/invocations \
  --positive-class YES \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Custom thresholds (less sensitive)**:
```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --feature-psi-thresh 0.3 \
  --drift-share-thresh 0.50 \
  --pred-psi-thresh 0.3 \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

### `run_monitor.py` - Evidently Monitoring

#### Configuration

Edit `configs/monitoring.yaml`:

```yaml
mlflow_tracking_uri: "file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns"
mlflow_experiment: "MediWatch-Monitoring"

paths:
  reference: "/home/ec2-user/projects/patient_selection/data/X_train.csv"
  logs_dir: "/home/ec2-user/projects/patient_selection/code/DEPLOY/logs"
  reports_out: "/home/ec2-user/projects/patient_selection/code/MONITOR/reports"

drift_thresholds:
  share_of_drifted_columns: 0.30
  p_value: 0.05
```

#### Usage

```bash
cd code/MONITOR/
python scripts/run_monitor.py
```

**Exit codes**:
- `0`: No drift detected (normal operation)
- `2`: Drift alert triggered (requires attention)

#### Viewing Reports

Open the generated HTML reports in your browser:
```bash
# On EC2 instance
cd code/MONITOR/reports/
ls -lt evidently_report_*.html

# Copy to local machine via SCP
scp -i key.pem ec2-user@<instance>:~/projects/patient_selection/code/MONITOR/reports/*.html .

# Or open via SSH tunnel
ssh -i key.pem -L 8000:localhost:8000 ec2-user@<instance>
# Then on instance:
cd code/MONITOR/reports/
python -m http.server 8000
# Access at http://localhost:8000
```

### `run_monitor_simple.py` - Statistical Monitoring

#### Usage

```bash
cd code/MONITOR/
python scripts/run_monitor_simple.py
```

**With custom config**:
```bash
python scripts/run_monitor_simple.py --config /path/to/custom_monitoring.yaml
```

#### Output Format

**drift_table_*.csv**:
```csv
feature,type,ks_stat,ks_pvalue,psi,drifted
num_lab_procedures,numeric,0.0234,0.123,0.156,False
number_inpatient,numeric,0.0891,0.001,0.312,True
race,categorical,,,0.089,False
gender,categorical,,,0.023,False
```

**drift_summary_*.json**:
```json
{
  "window": "2025-10-09",
  "share_of_drifted_columns": 0.25,
  "global_min_pvalue": 0.001,
  "alert": false,
  "thresholds": {
    "share": 0.30,
    "pvalue": 0.05
  }
}
```

---

## Configuration

### Evidently Monitoring (`configs/monitoring.yaml`)

```yaml
mlflow_tracking_uri: "file:///absolute/path/to/mlruns"
mlflow_experiment: "MediWatch-Monitoring"

paths:
  # Baseline reference data (training set)
  reference: "/path/to/X_train.csv"
  
  # Directory where production predictions are logged
  logs_dir: "/path/to/logs"
  
  # Output directory for generated reports
  reports_out: "/path/to/reports"

drift_thresholds:
  # Alert if >= 30% of features drift
  share_of_drifted_columns: 0.30
  
  # Alert if dataset-level p-value < 0.05
  p_value: 0.05
```

### Drift Thresholds Explained

#### Feature PSI (Population Stability Index)

```
PSI < 0.1    : No significant change
PSI 0.1-0.2  : Small change (monitor)
PSI 0.2-0.25 : Moderate change (warning)
PSI > 0.25   : Significant change (action required)
```

**Default**: 0.2 (moderate threshold)

#### KS Test p-value

Statistical test comparing two distributions:
- **p < 0.01**: Distributions are significantly different (drift detected)
- **p >= 0.01**: No significant difference

**Default**: 0.01 (stringent threshold)

#### Share of Drifted Features

Percentage of features that must show drift to trigger retraining:
- **30%**: Conservative (allows some natural variation)
- **20%**: Moderate (more sensitive)
- **50%**: Lenient (only major shifts trigger)

**Default**: 0.30 (30%)

#### Prediction PSI

Drift in the **distribution of predictions** (not input features):
- High prediction drift may indicate model degradation
- Can detect drift even if individual features look normal

**Default**: 0.2

---

## Drift Detection Logic

### How Drift is Detected

The monitoring system uses a **multi-gate approach**:

```python
trigger_retrain = (
    (share_of_drifted_features >= drift_share_thresh) OR
    (prediction_psi >= pred_psi_thresh)
)
```

**A feature is considered "drifted" if**:
- **Numeric**: `(PSI >= 0.2) OR (KS p-value <= 0.01)`
- **Categorical**: `PSI >= 0.2`

### Drift Metrics

#### 1. PSI (Population Stability Index)

Measures the shift in a feature's distribution:

```
PSI = Σ (P_current - P_baseline) × ln(P_current / P_baseline)
```

Where P is the proportion in each bin.

**Interpretation**:
- Symmetric (treats shifts in either direction equally)
- Unbounded (can exceed 1.0 for large shifts)
- Industry standard for monitoring

#### 2. KS Test (Kolmogorov-Smirnov)

Statistical test for numeric features:
- Compares empirical CDFs (cumulative distribution functions)
- Returns a p-value (probability that distributions are the same)
- Non-parametric (no assumptions about distribution shape)

#### 3. Chi-squared Test

Statistical test for categorical features:
- Compares frequency distributions
- Tests independence of categorical variables
- Returns a p-value

---

## Automated Retraining

### Retraining Workflow

When drift is detected, `monitor_and_retrain.py` automatically:

1. **Saves drift report** to `monitoring/out/drift_summary_*.json`
2. **Invokes training script** with current data
3. **Waits for training completion**
4. **Optionally rebuilds Docker image** (if `--build-script` provided)
5. **Logs all metrics to MLflow**

### Retraining Script Requirements

Your training script must accept:
```bash
python your_train_script.py \
  --data /path/to/new_data.csv \
  --experiment your_experiment_name \
  [--target your_target_column]
```

**Example**: `code/RAY/ray_tune_xgboost.py` already follows this interface:
```python
if __name__ == "__main__":
    parser.add_argument("--data", required=True)
    parser.add_argument("--experiment", default="xgb_diabetic_readmission_hpo")
    # ... etc
```

### Build Script Requirements

Your build script must accept:
```bash
python your_build_script.py \
  --tracking-uri file:///path/to/mlruns \
  --experiment your_experiment_name \
  --image-tag your-image:tag \
  --serve-port 5001
```

**Example**: `code/DEPLOY/build_docker_image.py` already follows this interface.

### Environment Variables

The monitoring script passes `MLFLOW_TRACKING_URI` to subprocesses:
```python
env = os.environ.copy()
if args.tracking_uri:
    env["MLFLOW_TRACKING_URI"] = args.tracking_uri
subprocess.check_call(retrain_cmd, env=env)
```

---

## Integration with Pipeline

### Complete MLOps Loop

```
┌─────────────────────────────────────────────────────────────┐
│                     Production Serving                       │
│  Docker Container (port 5001) → Predictions via /invocations│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─→ [Optional] logged_model.py
                     │   └─→ log_utils.py
                     │       └─→ logs/preds_*.parquet
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    Drift Detection                           │
│  monitor_and_retrain.py OR run_monitor.py                   │
│  • Compares baseline vs current                              │
│  • Computes PSI, KS test, Chi²                              │
│  • Checks thresholds                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─→ monitoring/out/drift_summary.json
                     │
        ┌────────────┴──────────── Drift Detected?
        │                                        │
       No                                       Yes
        │                                        │
        └─→ Exit 0                               ↓
                              ┌─────────────────────────────────┐
                              │    Automated Retraining         │
                              │  RAY/ray_tune_xgboost.py        │
                              │  • HPO with new data            │
                              │  • Log to MLflow                │
                              └──────────┬──────────────────────┘
                                         │
                                         ↓
                              ┌─────────────────────────────────┐
                              │    Docker Image Rebuild         │
                              │  DEPLOY/build_docker_image.py   │
                              │  • Export new model             │
                              │  • Build container              │
                              └──────────┬──────────────────────┘
                                         │
                                         ↓
                              ┌─────────────────────────────────┐
                              │      Redeploy Container         │
                              │  docker stop old_container      │
                              │  docker run new_image           │
                              └─────────────────────────────────┘
```

### Integration Points

#### With EDA Component
- Uses training baseline (`X_train.csv`) as reference
- Reference data created during EDA phase

#### With RAY Component
- Triggers retraining via `ray_tune_xgboost.py`
- Logs all experiments to same MLflow tracking store
- Uses same preprocessing pipeline

#### With DEPLOY Component
- Automatically rebuilds Docker images after retraining
- Can trigger container restart (requires orchestration layer)
- Maintains same API interface

### Scheduling Options

#### Option 1: Cron Job
```bash
# Add to crontab (daily at 2 AM)
0 2 * * * cd /home/ec2-user/projects/patient_selection && \
  python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production_batch.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  >> logs/monitoring.log 2>&1
```

#### Option 2: Airflow DAG
```python
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG('drift_monitoring', schedule_interval='@daily') as dag:
    monitor_task = BashOperator(
        task_id='monitor_drift',
        bash_command='python code/MONITOR/monitor_and_retrain.py ...',
        env={'MLFLOW_TRACKING_URI': 'file:///...'}
    )
```

#### Option 3: Kubernetes CronJob
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: drift-monitor
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: monitor
            image: diabetic-monitor:latest
            command: ["python", "monitor_and_retrain.py", ...]
```

---

## Prediction Logging

### `logged_model.py` - Transparent Logging Wrapper

Wraps your MLflow model to **automatically log all predictions**:

```python
from app.logged_model import LoggedModel
import mlflow

# Wrap existing model
logged_model = LoggedModel()

# Use in MLflow serving
mlflow.pyfunc.save_model(
    path="logged_model_dir",
    python_model=logged_model,
    artifacts={"base_model": "path/to/base_model"}
)
```

**What it does**:
- Intercepts every prediction call
- Appends features + predictions to daily Parquet file
- Zero impact on prediction latency
- Automatic date-based partitioning

### `log_utils.py` - Logging Utilities

```python
from app.log_utils import append_prediction_log

# After making predictions
append_prediction_log(
    features_df=input_features,
    y_pred=predictions,
    req_meta={"user_id": "123", "timestamp": "..."}
)
```

**Output format**: `logs/preds_2025-10-09.parquet`

```
| feature1 | feature2 | ... | __prediction__ | __timestamp__ | __user_id__ |
|----------|----------|-----|----------------|---------------|-------------|
| 1.2      | "A"      | ... | 0.73           | 2025-10-09T10:30:00 | 123 |
```

---

## Troubleshooting

### Common Issues

#### 1. "No production logs in logs_dir"

**Error**: `FileNotFoundError: No production logs in /path/to/logs`

**Cause**: No prediction logs available for monitoring

**Solution**:
```bash
# Check if logs directory exists and has .parquet files
ls -la /path/to/logs/preds_*.parquet

# If using logged_model.py, ensure PRED_LOG_DIR is set
export PRED_LOG_DIR=/path/to/logs

# If serving with Docker, mount the logs volume
docker run -v $(pwd)/logs:/logs -e PRED_LOG_DIR=/logs ...
```

#### 2. Prediction Parsing Errors

**Error**: `Unsupported prediction payload type`

**Cause**: API returns unexpected format

**Solution**:
```bash
# Test endpoint manually
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"age": 50, ...}]}'

# If returns dict with key:
python monitor_and_retrain.py ... --pred-key pred_proba

# If returns class labels:
python monitor_and_retrain.py ... --positive-class YES
```

#### 3. MLflow Tracking URI Errors

**Error**: `No such file or directory: mlruns`

**Cause**: Relative path issue

**Solution**: Always use **absolute paths**:
```bash
# Wrong
--tracking-uri file:mlruns

# Correct
--tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns
```

#### 4. Permission Errors

**Error**: `PermissionError: [Errno 13] Permission denied: 'monitoring/out'`

**Cause**: Directory doesn't exist or lacks write permissions

**Solution**:
```bash
# Create directories
mkdir -p code/MONITOR/monitoring/out
mkdir -p code/MONITOR/monitoring/tmp
mkdir -p code/MONITOR/reports

# Fix permissions
chmod 755 code/MONITOR/monitoring/out
```

#### 5. Retraining Script Fails

**Error**: Training subprocess exits with non-zero code

**Cause**: Training script doesn't accept required arguments

**Solution**: Ensure your training script accepts:
```python
# Required arguments
parser.add_argument("--data", required=True)
parser.add_argument("--experiment", default="default")

# Optional (if using --target-col)
parser.add_argument("--target", default="readmitted")
```

#### 6. Docker Build Fails After Retraining

**Error**: `No run named 'best_model_full_train' found`

**Cause**: Retraining script didn't create expected run

**Solution**: Check your training script creates a run with:
```python
with mlflow.start_run(run_name="best_model_full_train"):
    # ... train and log model
    mlflow.sklearn.log_model(model, artifact_path="model")
```

#### 7. High False Positive Rate (Too Many Alerts)

**Symptom**: Drift detected too frequently

**Solution**: Adjust thresholds to be less sensitive:
```bash
python monitor_and_retrain.py \
  --feature-psi-thresh 0.3 \      # from 0.2
  --drift-share-thresh 0.50 \     # from 0.30
  --pred-psi-thresh 0.3           # from 0.2
```

#### 8. Evidently Import Errors

**Error**: `ModuleNotFoundError: No module named 'evidently'`

**Cause**: Evidently not installed

**Solution**:
```bash
pip install evidently==0.4.27

# Or use simple monitoring (no Evidently required)
python scripts/run_monitor_simple.py
```

---

## Performance Considerations

### Monitoring Overhead

| Component | Scoring Time | Report Time | Storage per Day |
|-----------|--------------|-------------|-----------------|
| `monitor_and_retrain.py` | ~1-2s per 1K rows | <1s | ~10KB JSON |
| `run_monitor.py` | N/A | ~5-10s | ~500KB HTML + 100KB JSON |
| `run_monitor_simple.py` | N/A | ~1-2s | ~50KB CSV + 10KB JSON |

### Optimization Tips

1. **Batch predictions**: Score in batches of 1000-10000 rows
2. **Parallel monitoring**: Run multiple monitors for different features
3. **Sampling**: For very large datasets, sample before monitoring
4. **Archive old reports**: Keep only last N reports
5. **Async retraining**: Use job queue (Celery, RQ) for retraining

---

## Best Practices

### 1. Reference Data Selection

✅ **DO**:
- Use validation set (not training set) as reference
- Ensure reference is representative of normal operation
- Update reference periodically (e.g., every 6 months)

❌ **DON'T**:
- Use entire training set (too large, slow)
- Use test set (not representative of production)
- Never update reference (model will "drift away")

### 2. Threshold Configuration

✅ **DO**:
- Start conservative (high thresholds)
- Monitor false positive rate
- Adjust based on business impact
- Document threshold changes

❌ **DON'T**:
- Use default thresholds blindly
- Set thresholds too low (alert fatigue)
- Change thresholds too frequently

### 3. Alerting Strategy

✅ **DO**:
- Log all drift checks (even when no drift)
- Send alerts to appropriate channels (Slack, PagerDuty)
- Include drift report in alert
- Have clear escalation path

❌ **DON'T**:
- Only log when drift detected
- Alert everyone for every drift
- Alert without context/reports

### 4. Retraining Automation

✅ **DO**:
- Test retraining pipeline thoroughly
- Have rollback mechanism
- Validate new model before deployment
- Keep multiple model versions

❌ **DON'T**:
- Deploy retrained models automatically without validation
- Overwrite only model version
- Skip model performance checks

---

## Testing

### Test Drift Detection

Use the provided drift simulation:

```bash
# Train baseline model on original data
python code/RAY/ray_tune_xgboost.py \
  --data data/diabetic_data.csv

# Deploy baseline model
python code/DEPLOY/build_docker_image.py
docker run -d -p 5001:5001 --name baseline diabetic-xgb:serve

# Test with drift data (has +10 on number_inpatient)
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/diabetic_data.csv \
  --current data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py

# Should detect drift and trigger retraining!
```

### Unit Testing

```python
# test_drift_detection.py
from monitor_and_retrain import psi_numeric, detect_drift
import numpy as np

def test_psi_no_drift():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(0, 1, 1000)
    psi = psi_numeric(a, b)
    assert psi < 0.1, "PSI should be low for same distribution"

def test_psi_with_drift():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(3, 1, 1000)  # Shifted mean
    psi = psi_numeric(a, b)
    assert psi > 0.2, "PSI should be high for different distribution"
```

---

## Future Enhancements

### Planned Features

- [ ] **Model performance monitoring**: Track accuracy, AUC over time
- [ ] **Concept drift detection**: Separate feature drift from label drift
- [ ] **Gradual retraining**: Retrain on blend of old + new data
- [ ] **Multi-model comparison**: A/B test old vs retrained model
- [ ] **Real-time streaming monitoring**: Monitor live prediction stream
- [ ] **Drift explanations**: SHAP values for drifted predictions
- [ ] **Alert routing**: Different alerts for different drift types
- [ ] **Dashboard**: Real-time monitoring dashboard (Streamlit/Dash)

---

## References

### Documentation
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [PSI (Population Stability Index)](https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html)

### Papers
- Gama et al. (2014). "A Survey on Concept Drift Adaptation"
- Žliobaitė (2010). "Learning under Concept Drift: An Overview"

### Related Components
- [RAY HPO README](../RAY/README.md) - Training pipeline
- [DEPLOY README](../DEPLOY/README.md) - Deployment pipeline
- [Main README](../README.md) - Project overview

---

## Support

### Getting Help

1. **Check logs**:
   ```bash
   # Monitoring logs
   cat code/MONITOR/monitoring/out/drift_summary_*.json
   
   # Retraining logs (if MLflow)
   mlflow ui --backend-store-uri file:///.../mlruns
   ```

2. **Verify configuration**:
   ```bash
   # Check paths exist
   ls -la $(grep reference configs/monitoring.yaml | awk '{print $2}' | tr -d '"')
   
   # Test endpoint
   curl -X GET http://localhost:5001/health
   ```

3. **Review thresholds**:
   ```bash
   # Check last drift report
   jq . code/MONITOR/monitoring/out/drift_summary_*.json | tail -1
   ```

4. **Debug prediction parsing**:
   ```bash
   # Test endpoint with sample data
   python -c "
   import requests, json
   with open('data/X_train.csv') as f:
       r = requests.post('http://localhost:5001/invocations',
                        headers={'Content-Type': 'text/csv'},
                        data=f.read())
   print(json.dumps(r.json(), indent=2))
   "
   ```

---

## Changelog

### Version 1.0 (Current)
- ✅ Complete drift detection pipeline
- ✅ Automated retraining orchestration
- ✅ Three monitoring approaches (Evidently, Simple, Integrated)
- ✅ Robust prediction parsing
- ✅ MLflow integration
- ✅ Docker rebuild automation

---

**Questions?** Open an issue or check the main project [README](../README.md).

