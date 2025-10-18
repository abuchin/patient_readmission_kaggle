# MONITOR - Automated Drift Detection & Model Retraining

## Overview

The MONITOR component provides **automated drift detection** and **intelligent retraining orchestration** for the diabetic readmission prediction model. When data drift is detected, it automatically triggers model retraining and optionally rebuilds the Docker deployment image.

This is the **final stage** of the MLOps pipeline, completing the feedback loop from production monitoring back to model development.

---

## 📋 Table of Contents

- [What This Does](#what-this-does)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Drift Detection Methods](#drift-detection-methods)
- [Command Line Arguments](#command-line-arguments)
- [Usage Examples](#usage-examples)
- [Configuration & Thresholds](#configuration--thresholds)
- [Output Files](#output-files)
- [Integration with Pipeline](#integration-with-pipeline)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## What This Does

`monitor_and_retrain.py` is a **complete drift monitoring and retraining pipeline** that:

1. ✅ **Scores both baseline and current data** through your deployed model endpoint
2. ✅ **Computes drift metrics** using statistical tests (PSI, KS test)
3. ✅ **Detects drift** in both features and predictions
4. ✅ **Triggers automated retraining** when drift thresholds are exceeded
5. ✅ **Rebuilds Docker images** after successful retraining (optional)
6. ✅ **Logs all results** to JSON files for tracking

**Key Innovation**: Unlike traditional monitoring that only alerts, this system **automatically fixes the problem** by retraining the model on fresh data.

---

## Architecture

### Folder Structure

```
MONITOR/
├── monitor_and_retrain.py    # Main script (350 lines)
├── monitoring/
│   ├── out/                  # Drift detection results (JSON)
│   │   └── drift_summary_YYYYMMDDTHHMMSSZ.json
│   └── tmp/                  # Temporary files for scoring
│       ├── ref.csv           # Baseline data with predictions
│       └── cur.csv           # Current data with predictions
└── README.md                 # This file
```

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Baseline   │────▶│   Score     │────▶│   Detect    │────▶│  Retrain    │
│   Data      │     │   via API   │     │   Drift     │     │  + Deploy   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                    │                    │                    │
    baseline.csv        tmp/ref.csv         PSI + KS test      Ray Tune HPO
       │                    │                    │                    │
  Current Data         tmp/cur.csv         drift_summary.json    Docker Image
```

---

## Quick Start

### Prerequisites

1. **Deployed model** serving predictions at an endpoint (e.g., `http://localhost:5001/invocations`)
2. **Baseline data** (CSV file with features)
3. **Current/production data** (CSV file with same features)
4. **Training script** (e.g., `RAY/ray_tune_xgboost.py`)

### Basic Usage

```bash
python monitor_and_retrain.py \
  --baseline /home/ec2-user/projects/patient_selection/data/diabetic_data.csv \
  --current /home/ec2-user/projects/patient_selection/data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script /home/ec2-user/projects/patient_selection/patient_readmission_kaggle/RAY/ray_tune_xgboost.py \
  --tracking-uri file:/home/ec2-user/projects/patient_selection/patient_readmission_kaggle/RAY/mlruns \
  --experiment xgb_diabetic_readmission_hpo
```

### What Happens

1. **Scoring**: Both datasets are scored through your API endpoint
2. **Drift Check**: Computes PSI and KS test for all features and predictions
3. **Decision**: If drift thresholds exceeded → triggers retraining
4. **Retraining**: Runs `ray_tune_xgboost.py` with the current data
5. **Output**: Saves drift report to `monitoring/out/drift_summary_*.json`

---

## How It Works

### Step-by-Step Process

#### 1. Data Loading
```python
ref = pd.read_csv(args.baseline)     # Your baseline/reference data
cur = pd.read_csv(args.current)      # New production data
```

#### 2. Prediction Scoring
```python
# Sends data to your deployed model endpoint
# POST http://localhost:5001/invocations with CSV payload
# Attaches predictions as new column: pred_proba
ref_scored = score_and_attach(ref, endpoint)
cur_scored = score_and_attach(cur, endpoint)
```

#### 3. Drift Calculation

For **each feature**:
- **Numeric**: Compute PSI (Population Stability Index) + KS test
- **Categorical**: Compute PSI based on category frequencies

For **predictions**:
- Compute PSI on prediction distribution
- Compute KS test on prediction values

#### 4. Drift Decision

**A feature drifts if**:
```python
(PSI >= 0.2) OR (KS p-value <= 0.01)
```

**Retraining is triggered if**:
```python
(share_of_drifted_features >= 30%) OR
(max_feature_psi >= 1.0) OR              # Any feature has extreme drift
(critical_feature_drifts) OR              # Important feature drifts
(prediction_psi >= 0.2)                   # Model output shifts
```

#### 5. Automated Retraining

If drift detected:
```bash
# Automatically runs your training script
python RAY/ray_tune_xgboost.py \
  --data current_data.csv \
  --mlruns-dir /path/to/mlruns \
  --num-samples 15 \
  --cpus-per-trial 2
```

#### 6. Optional Docker Rebuild

If `--build-script` provided:
```bash
# Automatically rebuilds deployment image
python DEPLOY/build_docker_image.py \
  --tracking-uri file:///path/to/mlruns \
  --experiment xgb_diabetic_readmission_hpo \
  --image-tag diabetic-xgb:serve
```

---

## Drift Detection Methods

### 1. PSI (Population Stability Index)

**For Numeric Features**:
```python
# Bins data into deciles
# Compares distribution between baseline and current
PSI = Σ (P_current - P_baseline) × ln(P_current / P_baseline)
```

**Interpretation**:
- `PSI < 0.1`: No significant change
- `PSI 0.1-0.2`: Small change (monitor closely)
- `PSI 0.2-0.25`: Moderate drift ⚠️ **Default threshold**
- `PSI > 0.25`: Significant drift (action required)

**For Categorical Features**:
```python
# Compares category frequencies
PSI = Σ (P_current - P_baseline) × ln(P_current / P_baseline)
```

### 2. KS Test (Kolmogorov-Smirnov)

**For Numeric Features**:
- Compares empirical cumulative distribution functions
- Returns p-value (probability distributions are the same)
- **Threshold**: p < 0.01 indicates significant difference

**For Predictions**:
- Optional secondary check for prediction drift
- Use `--pred-ks-p-thresh 0.01` to enable

### 3. Multi-Gate Logic

The system uses **multiple gates** for robust drift detection:

```python
trigger_retrain = (
    # Gate 1: Share threshold
    (share_of_drifted_features >= 0.30) OR
    
    # Gate 2: Extreme drift in any feature
    (max_feature_psi >= 1.0) OR
    
    # Gate 3: Critical feature drift
    (critical_feature_psi >= 0.5) OR
    
    # Gate 4: Prediction drift
    (prediction_psi >= 0.2)
)
```

---

## Command Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--baseline` | str | Path to baseline/reference data CSV |
| `--current` | str | Path to current/production data CSV |
| `--endpoint` | str | Model serving endpoint URL |
| `--retrain-script` | str | Path to training script (e.g., `RAY/ray_tune_xgboost.py`) |

### Data & Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tmp-dir` | str | `monitoring/tmp` | Temporary directory for scored CSVs |
| `--out-dir` | str | `monitoring/out` | Output directory for drift reports |
| `--prediction-col` | str | `pred_proba` | Name for prediction column |
| `--target-col` | str | `None` | Target column name (if needed by retrain script) |

### Prediction Parsing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pred-key` | str | `None` | Key to extract from dict responses (e.g., `pred_proba`) |
| `--positive-class` | str | `None` | Positive class label for categorical predictions |

### Feature Selection & Gates

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ignore-cols` | str | `""` | Comma-separated columns to ignore (e.g., IDs) |
| `--critical-cols` | str | `""` | Comma-separated critical feature names |
| `--critical-psi-thresh` | float | `0.5` | PSI threshold for critical features |
| `--any-feature-psi-thresh` | float | `1.0` | Trigger if ANY feature PSI exceeds this |

### Drift Thresholds

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--feature-psi-thresh` | float | `0.2` | PSI threshold for regular features |
| `--ks-p-thresh` | float | `0.01` | KS test p-value threshold |
| `--drift-share-thresh` | float | `0.30` | Share of features that must drift (30%) |
| `--pred-psi-thresh` | float | `0.2` | PSI threshold for predictions |
| `--pred-ks-p-thresh` | float | `0.0` | KS p-value for predictions (0=disabled) |

### MLflow & Deployment

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tracking-uri` | str | `""` | MLflow tracking URI (e.g., `file:///path/to/mlruns`) |
| `--experiment` | str | `xgb_diabetic_readmission_hpo` | MLflow experiment name |
| `--build-script` | str | `None` | Path to Docker build script (optional) |
| `--image-tag` | str | `diabetic-xgb:serve` | Docker image tag |

### Ray Tune HPO Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hpo-num-samples` | int | `15` | Number of Ray Tune trials |
| `--hpo-cpus` | int | `2` | CPUs per trial |
| `--hpo-gpus` | float | `0.0` | GPUs per trial |
| `--hpo-test-size` | float | `0.2` | Test set proportion |
| `--hpo-ray-dir` | str | `ray_exp_retrain` | Ray results directory |
| `--hpo-seed` | int | `123` | Random seed for reproducibility |

### Special Flags

| Argument | Type | Description |
|----------|------|-------------|
| `--force` | flag | Force retraining regardless of drift |

---

## Usage Examples

### Example 1: Basic Drift Detection (No Retraining)

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/X_test.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Result**: Checks for drift, saves report, but only retrains if drift detected.

### Example 2: Fully automated pipeline

```bash
python MONITOR/monitor_and_retrain.py \
  --baseline /home/ec2-user/projects/patient_selection/data/diabetic_data.csv \
  --current  /home/ec2-user/projects/patient_selection/data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations \
  --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
  --retrain-script /home/ec2-user/projects/patient_selection/code/RAY/ray_tune_xgboost.py \
  --build-script /home/ec2-user/projects/patient_selection/code/DEPLOY/build_docker_image.py \
  --image-tag diabetic-xgb:serve \
  --ignore-cols encounter_id,patient_nbr \
  --any-feature-psi-thresh 1.0 \
  --critical-cols number_inpatient \
  --critical-psi-thresh 0.5 \
  --pred-ks-p-thresh 1e-9 \
  --hpo-num-samples 10 \
  --hpo-cpus 4 \
  --hpo-gpus 0 \
  --hpo-test-size 0.2 \
  --hpo-ray-dir ray_exp_retrain \
  --hpo-seed 2025

```


**Result**: Detects drift → Retrains → Rebuilds Docker image

### Example 3: Handling Dict Predictions

If your API returns:
```json
[{"pred_proba": 0.73, "label": "YES"}, {"pred_proba": 0.28, "label": "NO"}]
```

Use:
```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/baseline.csv \
  --current data/production.csv \
  --endpoint http://api.example.com/predict \
  --pred-key pred_proba \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

### Example 4: Handling Class Label Predictions

If your API returns:
```json
["YES", "NO", "YES", "NO"]
```

Use:
```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/baseline.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --positive-class YES \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

### Example 5: Ignoring ID Columns

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --ignore-cols "encounter_id,patient_nbr" \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

### Example 6: Critical Features with Stricter Thresholds

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --critical-cols "number_inpatient,time_in_hospital,num_medications" \
  --critical-psi-thresh 0.3 \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Result**: If critical features drift > 0.3 PSI, immediate retraining triggered

### Example 7: More Sensitive Drift Detection

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --feature-psi-thresh 0.15 \
  --drift-share-thresh 0.20 \
  --pred-psi-thresh 0.15 \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

### Example 8: Less Sensitive (Production-Stable)

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

### Example 9: Force Retraining (No Drift Check)

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/latest_batch.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  --force
```

**Result**: Skips drift detection, immediately retrains

### Example 10: Fast Retraining (Fewer HPO Trials)

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  --hpo-num-samples 5 \
  --hpo-cpus 4
```

**Result**: Quick retraining with only 5 HPO trials instead of default 15

---

## Configuration & Thresholds

### Recommended Settings by Environment

#### Development/Testing
```bash
--feature-psi-thresh 0.15       # More sensitive
--drift-share-thresh 0.20       # Lower threshold
--hpo-num-samples 5             # Faster retraining
```

#### Staging
```bash
--feature-psi-thresh 0.2        # Moderate sensitivity
--drift-share-thresh 0.30       # Default
--hpo-num-samples 15            # Balanced quality/speed
```

#### Production
```bash
--feature-psi-thresh 0.25       # Less sensitive (avoid false alarms)
--drift-share-thresh 0.40       # Higher threshold
--hpo-num-samples 30            # Better model quality
--critical-cols "key_features"  # Focus on important features
```

### Threshold Tuning Guide

**Too Many False Alarms?**
- ⬆️ Increase `--feature-psi-thresh` from 0.2 → 0.3
- ⬆️ Increase `--drift-share-thresh` from 0.30 → 0.50
- ⬆️ Increase `--pred-psi-thresh` from 0.2 → 0.3

**Missing Real Drift?**
- ⬇️ Decrease `--feature-psi-thresh` from 0.2 → 0.15
- ⬇️ Decrease `--drift-share-thresh` from 0.30 → 0.20
- ➕ Add `--critical-cols` for important features
- ➕ Enable `--pred-ks-p-thresh 0.01`

**Focusing on Key Features?**
- Use `--critical-cols "feat1,feat2,feat3"`
- Set `--critical-psi-thresh 0.3` (stricter than regular)
- Use `--any-feature-psi-thresh 0.5` (trigger on any extreme drift)

---

## Output Files

### Drift Summary JSON

**Location**: `monitoring/out/drift_summary_20251009T213138Z.json`

**Structure**:
```json
{
  "trigger_retrain": false,
  "share_drifted": 0.0392,
  "n_features": 51,
  "n_drifted": 2,
  "max_feature_psi": 0.0000,
  "any_feature_psi_thresh": 1.0,
  "critical_hits": [],
  "critical_psi_thresh": 0.5,
  "pred_psi": 0.0,
  "pred_ks_p": 1.0,
  "pred_gate": false,
  "feature_psi_thresh": 0.2,
  "ks_p_thresh": 0.01,
  "drift_share_thresh": 0.3,
  "pred_psi_thresh": 0.2,
  "pred_ks_p_thresh": 0.0,
  "details": [
    {
      "column": "time_in_hospital",
      "type": "numeric",
      "psi": 0.0,
      "ks_p": 1.0,
      "drift": false
    },
    {
      "column": "num_medications",
      "type": "numeric",
      "psi": 0.1234,
      "ks_p": 0.05,
      "drift": false
    },
    ...
  ]
}
```

### Temporary Scoring Files

**Location**: `monitoring/tmp/`

- `ref.csv`: Baseline data with `pred_proba` column
- `cur.csv`: Current data with `pred_proba` column

These are automatically created during the scoring step and retained for debugging.

---

## Integration with Pipeline

### Complete MLOps Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                1. Model Training (RAY)                       │
│   ray_tune_xgboost.py → Best model → MLflow tracking        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                2. Model Deployment (DEPLOY)                  │
│   build_docker_image.py → Docker container → Port 5001      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                3. Production Serving                         │
│   POST /invocations → Predictions → Business value          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         4. Drift Monitoring (MONITOR) ← YOU ARE HERE         │
│   monitor_and_retrain.py                                    │
│   • Score baseline + current data                           │
│   • Compute drift metrics (PSI, KS)                         │
│   • Check thresholds                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴─────────── Drift Detected?
         │                                     │
        No                                    Yes
         │                                     │
         └─→ Continue Serving                  ↓
                              ┌─────────────────────────────────┐
                              │  5. Auto-Retrain (back to RAY)  │
                              │  • Trigger ray_tune_xgboost.py  │
                              │  • Train on current data        │
                              │  • Save to MLflow               │
                              └──────────┬──────────────────────┘
                                         ↓
                              ┌─────────────────────────────────┐
                              │  6. Auto-Redeploy (DEPLOY)      │
                              │  • build_docker_image.py        │
                              │  • New container                │
                              └──────────┬──────────────────────┘
                                         │
                                         └─→ Back to Production Serving
```

### Scheduling Options

#### Option 1: Cron (Simple)

```bash
# Run daily at 2 AM
0 2 * * * cd /home/ec2-user/projects/patient_selection && \
  python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current /tmp/daily_production_data.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
  >> logs/monitoring.log 2>&1
```

#### Option 2: Systemd Timer

Create `/etc/systemd/system/drift-monitor.service`:
```ini
[Unit]
Description=Drift Monitoring and Retraining
After=network.target

[Service]
Type=oneshot
User=ec2-user
WorkingDirectory=/home/ec2-user/projects/patient_selection
ExecStart=/home/ec2-user/projects/patient_selection/code/patient_env/bin/python \
  code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current /tmp/production_data.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

Create `/etc/systemd/system/drift-monitor.timer`:
```ini
[Unit]
Description=Run drift monitoring daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable drift-monitor.timer
sudo systemctl start drift-monitor.timer
```

#### Option 3: Airflow DAG

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'drift_monitoring_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
)

monitor_task = BashOperator(
    task_id='monitor_and_retrain',
    bash_command='''
    cd /home/ec2-user/projects/patient_selection && \
    python code/MONITOR/monitor_and_retrain.py \
      --baseline data/X_train.csv \
      --current /data/production/{{ ds }}.csv \
      --endpoint http://model-api:5001/invocations \
      --retrain-script code/RAY/ray_tune_xgboost.py \
      --tracking-uri file:///home/ec2-user/projects/patient_selection/code/RAY/mlruns
    ''',
    dag=dag,
)
```

---

## Advanced Features

### 1. Critical Features Monitoring

Monitor **key clinical features** more strictly:

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --critical-cols "number_inpatient,time_in_hospital,num_medications,number_diagnoses" \
  --critical-psi-thresh 0.3 \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Use case**: Hospital protocols change, affecting specific features

### 2. Extreme Drift Gate

Trigger retraining if **any single feature** shows extreme drift:

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --any-feature-psi-thresh 0.5 \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Use case**: Rapid changes in patient demographics or medical practices

### 3. Prediction Drift with KS Test

Double-check prediction drift using both PSI and KS test:

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --pred-psi-thresh 0.2 \
  --pred-ks-p-thresh 0.01 \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Use case**: Model output distribution shifts (concept drift)

### 4. Ignoring Noisy Columns

Exclude columns that change frequently but aren't meaningful:

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --ignore-cols "encounter_id,patient_nbr,admission_source_id" \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Use case**: ID columns, timestamps, or administrative fields

### 5. Fast Retraining for Emergencies

Quick retraining with minimal HPO trials:

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/emergency_data.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  --hpo-num-samples 3 \
  --hpo-cpus 8 \
  --force
```

**Use case**: Critical system failure, need immediate model update

---

## Troubleshooting

### Common Issues

#### 1. Endpoint Connection Errors

**Error**: `requests.exceptions.ConnectionError: Connection refused`

**Cause**: Model endpoint not running

**Solution**:
```bash
# Check if container is running
docker ps | grep diabetic

# Start if needed
docker run -d -p 5001:5001 --name model diabetic-xgb:serve

# Test endpoint
curl -X GET http://localhost:5001/health
```

#### 2. Prediction Parsing Errors

**Error**: `Unsupported prediction payload type`

**Cause**: API returns unexpected format

**Solution**:
```bash
# Test endpoint manually
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: text/csv" \
  --data-binary @data/X_train.csv | head

# If returns dict with key:
--pred-key pred_proba

# If returns class labels:
--positive-class YES
```

#### 3. Column Mismatch

**Error**: `KeyError: 'some_column'`

**Cause**: Current data missing columns from baseline

**Solution**:
```bash
# Check columns
python -c "
import pandas as pd
baseline = pd.read_csv('data/X_train.csv')
current = pd.read_csv('data/production.csv')
print('Baseline columns:', set(baseline.columns))
print('Current columns:', set(current.columns))
print('Missing in current:', set(baseline.columns) - set(current.columns))
"

# Add missing columns or use --ignore-cols
```

#### 4. Retraining Script Fails

**Error**: `subprocess.CalledProcessError: Command ... returned non-zero exit status 1`

**Cause**: Training script error

**Solution**:
```bash
# Run training manually to see full error
python code/RAY/ray_tune_xgboost.py \
  --data data/diabetic_data_drift.csv \
  --mlruns-dir /home/ec2-user/projects/patient_selection/code/RAY/mlruns \
  --num-samples 5

# Check logs
cat ~/ray_results/*/logs/*.out
```

#### 5. MLflow Tracking URI Issues

**Error**: `No such file or directory: 'mlruns'`

**Cause**: Relative path issue

**Solution**: Always use absolute paths:
```bash
--tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns
# NOT: file:mlruns or file:./RAY/mlruns
```

#### 6. No Drift Detected (But Visually Obvious)

**Symptom**: Data clearly changed but no drift triggered

**Solution**: Make thresholds more sensitive:
```bash
--feature-psi-thresh 0.15       # from 0.2
--drift-share-thresh 0.20       # from 0.30
--ks-p-thresh 0.05              # from 0.01
```

Or check specific columns:
```bash
# View drift details
cat monitoring/out/drift_summary_*.json | jq '.details[] | select(.drift == true)'
```

#### 7. Too Many Retrainings (False Alarms)

**Symptom**: Retraining triggered too frequently

**Solution**: Make thresholds less sensitive:
```bash
--feature-psi-thresh 0.3        # from 0.2
--drift-share-thresh 0.50       # from 0.30
```

Or focus on critical features only:
```bash
--critical-cols "key_feature1,key_feature2"
--any-feature-psi-thresh 0.6    # only extreme drift
```

#### 8. Permission Errors

**Error**: `PermissionError: [Errno 13] Permission denied: 'monitoring/out'`

**Solution**:
```bash
# Create directories with correct permissions
mkdir -p code/MONITOR/monitoring/out
mkdir -p code/MONITOR/monitoring/tmp
chmod 755 code/MONITOR/monitoring/out
chmod 755 code/MONITOR/monitoring/tmp
```

---

## Best Practices

### 1. Baseline Selection

✅ **DO**:
- Use a **held-out validation set** (not training set)
- Ensure baseline is **representative** of normal operation
- **Update baseline** periodically (e.g., quarterly)
- **Version baselines** (`baseline_2025Q1.csv`, `baseline_2025Q2.csv`)

❌ **DON'T**:
- Use entire training set (too large, slow)
- Use test set (not production-representative)
- Never update baseline (model will "drift away")

### 2. Threshold Configuration

✅ **DO**:
- **Start conservative** (higher thresholds)
- **Monitor false positive rate** over 1-2 weeks
- **Adjust gradually** based on production behavior
- **Document changes** in a changelog

❌ **DON'T**:
- Copy default thresholds blindly
- Change thresholds too frequently
- Set too-low thresholds (alert fatigue)

### 3. Critical Features

✅ **DO**:
- Identify **domain-critical features** (clinical judgment)
- Use **stricter thresholds** for critical features
- **Document rationale** for critical feature selection

❌ **DON'T**:
- Mark all features as critical
- Use same threshold for all features

### 4. Scheduling

✅ **DO**:
- Run monitoring **daily** or **weekly**
- Run during **low-traffic periods** (2-4 AM)
- **Log all runs** (even when no drift)
- **Alert on failures** (not just drift)

❌ **DON'T**:
- Run too frequently (hourly)
- Run during peak hours
- Only log drift events

### 5. Retraining Strategy

✅ **DO**:
- **Validate** new model before deployment
- **Keep old model** for rollback
- **A/B test** new vs old model
- **Document** each retraining event

❌ **DON'T**:
- Auto-deploy without validation
- Delete old models immediately
- Retrain on every tiny drift

### 6. Production Deployment

✅ **DO**:
- Use **blue-green deployment** (two containers)
- Implement **health checks**
- Have **rollback procedure**
- Test **with sample data** before full cutover

❌ **DON'T**:
- Replace running container immediately
- Deploy without testing
- Have no rollback plan

---

## Testing

### Test with Simulated Drift

The project includes drift-simulated data:

```bash
# 1. Start baseline model
docker run -d -p 5001:5001 --name baseline diabetic-xgb:serve

# 2. Run monitoring with drift data (has +10 on number_inpatient)
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/diabetic_data.csv \
  --current data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
  --hpo-num-samples 3

# Should detect drift in number_inpatient and trigger retraining!
```

### Unit Testing Examples

```python
# test_drift_detection.py
from monitor_and_retrain import psi_numeric, psi_categorical, detect_drift
import numpy as np
import pandas as pd

def test_psi_no_drift():
    """Same distribution → low PSI"""
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(0, 1, 1000)
    psi = psi_numeric(a, b)
    assert psi < 0.1, f"Expected low PSI, got {psi}"

def test_psi_with_drift():
    """Shifted distribution → high PSI"""
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(3, 1, 1000)  # Mean shifted by 3
    psi = psi_numeric(a, b)
    assert psi > 0.2, f"Expected high PSI, got {psi}"

def test_categorical_drift():
    """Changed category frequencies → drift"""
    ref = pd.Series(['A']*700 + ['B']*300)
    cur = pd.Series(['A']*300 + ['B']*700)  # Flipped
    psi = psi_categorical(ref, cur)
    assert psi > 0.2, f"Expected drift, got PSI={psi}"

def test_drift_decision():
    """30% features drift → trigger"""
    ref = pd.DataFrame({
        'f1': np.random.normal(0, 1, 100),
        'f2': np.random.normal(0, 1, 100),
        'f3': np.random.normal(0, 1, 100),
    })
    cur = pd.DataFrame({
        'f1': np.random.normal(3, 1, 100),  # Drifted
        'f2': np.random.normal(0, 1, 100),  # Same
        'f3': np.random.normal(0, 1, 100),  # Same
    })
    
    trigger, summary = detect_drift(
        ref, cur,
        feature_psi_thresh=0.2,
        ks_p_thresh=0.01,
        drift_share_thresh=0.30,
        pred_psi_thresh=0.2,
        pred_col='pred_proba'
    )
    
    # 1/3 = 33% features drifted → should trigger
    assert trigger, "Should trigger with 33% drift"
    assert summary['n_drifted'] >= 1
```

Run tests:
```bash
python -m pytest test_drift_detection.py -v
```

---

## Future Enhancements

### Planned Features

- [ ] **Model performance tracking**: Monitor actual accuracy/AUC in production
- [ ] **Concept drift detection**: Separate feature drift from label shift
- [ ] **Gradual retraining**: Blend old + new data with weights
- [ ] **Multi-model comparison**: A/B test old vs retrained model
- [ ] **Real-time streaming**: Monitor live prediction stream
- [ ] **SHAP-based drift**: Explain which features contribute most to drift
- [ ] **Alert routing**: Slack/email notifications on drift
- [ ] **Dashboard**: Streamlit/Grafana dashboard for monitoring
- [ ] **Rollback automation**: Auto-rollback if new model performs poorly
- [ ] **Cost tracking**: Log retraining costs and model performance gains

---

## References

### Statistical Methods
- [Population Stability Index (PSI)](https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Chi-squared Test](https://en.wikipedia.org/wiki/Chi-squared_test)

### MLOps Best Practices
- [Continuous Training in Production](https://ml-ops.org/content/continuous-training)
- [Model Monitoring](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)

### Related Components
- [RAY HPO README](../RAY/README.md) - Hyperparameter optimization
- [DEPLOY README](../DEPLOY/README.md) - Model deployment
- [Main README](../README.md) - Project overview

---

## Support

### Debugging Commands

```bash
# 1. Check drift summary
cat code/MONITOR/monitoring/out/drift_summary_*.json | jq .

# 2. View drifted features only
cat code/MONITOR/monitoring/out/drift_summary_*.json | \
  jq '.details[] | select(.drift == true)'

# 3. Check prediction scores
head -20 code/MONITOR/monitoring/tmp/ref.csv
head -20 code/MONITOR/monitoring/tmp/cur.csv

# 4. Test endpoint manually
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: text/csv" \
  --data-binary @data/X_train.csv | head -20

# 5. Check MLflow experiments
mlflow ui --backend-store-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns

# 6. View recent drift reports
ls -lht code/MONITOR/monitoring/out/

# 7. Check if retraining succeeded
ls -lht code/RAY/mlruns/*/meta.yaml
```

---

## Quick Reference Card

```bash
# Standard Usage
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/X_train.csv \
  --current data/production.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py

# With Docker rebuild
python code/MONITOR/monitor_and_retrain.py \
  ... (above args) ... \
  --tracking-uri file://.../RAY/mlruns \
  --build-script code/DEPLOY/build_docker_image.py

# Custom thresholds
python code/MONITOR/monitor_and_retrain.py \
  ... (above args) ... \
  --feature-psi-thresh 0.25 \
  --drift-share-thresh 0.40

# Critical features
python code/MONITOR/monitor_and_retrain.py \
  ... (above args) ... \
  --critical-cols "feat1,feat2" \
  --critical-psi-thresh 0.3

# Force retrain
python code/MONITOR/monitor_and_retrain.py \
  ... (above args) ... \
  --force
```

---

**Questions?** Check the [main project README](../README.md) or open an issue.

**Last Updated**: October 2025
