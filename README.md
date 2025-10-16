# Patient Readmission Prediction Project

## Overview

Machine learning pipeline for predicting hospital readmissions using diabetic patient data. Features hyperparameter optimization with Ray Tune, MLflow tracking, Docker deployment, automated drift detection with retraining, and Airflow orchestration.

**Detailed description**: https://docs.google.com/document/d/1WmBA18F_3HDC5_bm-lsKuL9ZWlZT92M2uA5EJ-tV2s4/edit?usp=sharing

## Project Structure

```
patient_selection/
├── code/
│   ├── EDA/               # Exploratory Data Analysis
│   ├── RAY/               # Hyperparameter Optimization (Ray Tune + MLflow)
│   ├── DEPLOY/            # Model Deployment (Docker + REST API)
│   ├── MONITOR/           # Drift Detection & Auto-Retraining
│   ├── airflow/           # Pipeline Orchestration
│   └── requirements.txt   # Python dependencies
└── data/
    └── diabetic_data.csv  # Dataset from Kaggle
```

## Dataset

**Source**: [Diabetes 130-US hospitals for years 1999-2008](https://www.kaggle.com/datasets/brandao/diabetes)  
**File**: `diabetic_data.csv`

**Features**:
- Demographics: gender, race, age
- Medical: lab procedures, medications, diagnoses, inpatient visits
- Target: readmission status (NO, <30 days, >30 days)

**Key Characteristics**:
- ~100,000 patient records with 50+ features
- Imbalanced classes (more non-readmitted patients)
- Mixed distributions (some normal, some skewed → suggests non-linear methods)
- Missing data handled as "Unknown/Invalid"

## Pipeline Overview

```
1. EDA (Exploration)     → Understand data, baseline model
2. RAY (Optimization)    → Find best hyperparameters (XGBoost)
3. DEPLOY (Production)   → Package as Docker container with REST API
4. MONITOR (Observability) → Detect drift, trigger retraining
5. AIRFLOW (Orchestration) → Automate deployment + monitoring workflows
```

**Continuous Feedback Loop**:
```
Production Data → MONITOR (drift detection)
      ↓ (if drift detected)
    RAY (retrain) → DEPLOY (rebuild) → Production
           ↑
    AIRFLOW automates this loop
```

---

## Architecture

### System Overview

The system follows a modular architecture with five main components that work together to create a complete ML lifecycle:

```
┌────────────────────────────────────────────────────────────────────────┐
│                         AIRFLOW ORCHESTRATION                          │
│                    (Scheduler, DAGs, Task Manager)                     │
│                                                                        │
│  ┌─────────────┐           ┌──────────────────────────┐              │
│  │ Deploy DAG  │  (@once)  │   Monitor & Retrain DAG  │ (daily 02:00)│
│  └─────────────┘           └──────────────────────────┘              │
└────────────────────────────────────────────────────────────────────────┘
           │                              │
           │ triggers                     │ triggers
           ↓                              ↓
    ┌──────────────┐            ┌─────────────────┐
    │   DEPLOY     │            │    MONITOR      │
    │  Component   │            │   Component     │
    └──────────────┘            └─────────────────┘
           ↑                              │
           │                              │ (if drift)
           │                              ↓
           │                    ┌─────────────────┐
           │                    │      RAY        │
           └────────────────────│   Component     │
                 (rebuild)      └─────────────────┘
                                         ↑
                                         │ (training data)
                                         │
                                ┌─────────────────┐
                                │      EDA        │
                                │   Component     │
                                └─────────────────┘

                    ┌────────────────────────────┐
                    │   MLflow Tracking Store    │
                    │  (Shared Experiment Data)  │
                    └────────────────────────────┘
                              ↑
                    (all components log here)
```

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

1. TRAINING PHASE
   ────────────────

   diabetic_data.csv
         │
         ↓
   ┌───────────────────┐
   │   EDA Component   │  • Data exploration
   │   (Notebook)      │  • Feature analysis
   └───────────────────┘  • Baseline modeling
         │
         │ insights & preprocessed data
         ↓
   ┌───────────────────┐
   │   RAY Component   │  • Hyperparameter search (Ray Tune)
   │  (ray_tune_xgb.py)│  • 5-fold cross-validation
   └───────────────────┘  • Multiple trials (30-50)
         │
         │ best hyperparameters + trained model
         ↓
   ┌───────────────────┐
   │  MLflow Tracking  │  • Experiment logs
   │      Store        │  • Model artifacts
   └───────────────────┘  • Hyperparameters & metrics
         │
         │ best model retrieval
         ↓

2. DEPLOYMENT PHASE
   ────────────────

   ┌───────────────────┐
   │ DEPLOY Component  │  • Export best model
   │(build_docker.py)  │  • Generate Dockerfile
   └───────────────────┘  • Build Docker image
         │
         │ Docker image with model + dependencies
         ↓
   ┌───────────────────┐
   │  Docker Container │  • MLflow model serving
   │   (Port 5001)     │  • REST API endpoints
   └───────────────────┘  • /invocations, /health
         │
         │ prediction requests (JSON/CSV)
         ↓
   ┌───────────────────┐
   │  Client/User      │  • Send patient data
   │  Application      │  • Receive predictions
   └───────────────────┘

3. MONITORING PHASE
   ─────────────────

   production_data.csv + baseline_data.csv
         │
         ↓
   ┌───────────────────┐
   │ MONITOR Component │  • Score data via API
   │(monitor_retrain.py│  • Compute PSI & KS tests
   └───────────────────┘  • Detect feature/prediction drift
         │
         ├─ NO DRIFT → Continue monitoring
         │
         └─ DRIFT DETECTED
               │
               ↓
         ┌───────────────────┐
         │  Trigger Retrain  │  • Call RAY component
         │   (subprocess)    │  • New HPO with fresh data
         └───────────────────┘
               │
               │ new best model
               ↓
         ┌───────────────────┐
         │ Rebuild & Deploy  │  • Call DEPLOY component
         │   (subprocess)    │  • New Docker image
         └───────────────────┘
               │
               ↓
         Updated Production Model
```

### Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT COMMUNICATION                          │
└─────────────────────────────────────────────────────────────────────────┘

EDA Component
  └─→ Outputs: Cleaned data understanding, feature insights
      └─→ Consumed by: RAY (informs preprocessing strategy)

RAY Component
  ├─→ Inputs: diabetic_data.csv, preprocessing config
  ├─→ Process: Ray Tune (parallel HPO), XGBoost training, CV evaluation
  ├─→ Outputs: Best model + config → MLflow
  └─→ Consumed by: DEPLOY (retrieves best model)

MLflow Tracking Store (Shared Knowledge Base)
  ├─→ Written by: RAY (experiments), MONITOR (drift logs)
  ├─→ Read by: DEPLOY (best model), MONITOR (model endpoint)
  └─→ Storage: File system (mlruns/) or Remote Server

DEPLOY Component
  ├─→ Inputs: MLflow tracking URI, experiment name
  ├─→ Process: Export model, generate Dockerfile, build image
  ├─→ Outputs: Docker container serving REST API
  └─→ Consumed by: MONITOR (prediction endpoint), Clients (predictions)

MONITOR Component
  ├─→ Inputs: Baseline data, current data, model endpoint
  ├─→ Process: Score data, compute drift (PSI, KS), evaluate thresholds
  ├─→ Decision Logic:
  │     IF drift_detected:
  │        └─→ Call RAY (retrain)
  │        └─→ Call DEPLOY (rebuild)
  │     ELSE:
  │        └─→ Log "no drift" and continue
  └─→ Outputs: Drift reports (JSON), retraining trigger

AIRFLOW Orchestration
  ├─→ Deploy DAG (@once):
  │     └─→ Task: Run DEPLOY/build_docker_image.py
  │
  └─→ Monitor DAG (daily 02:00):
        └─→ Task: Run MONITOR/monitor_and_retrain.py
              ├─→ Internally calls RAY if drift
              └─→ Internally calls DEPLOY if retraining succeeds
```

### Data Transformations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA TRANSFORMATION PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────┘

Raw CSV Data
  │
  │ [diabetic_data.csv: ~100K rows × 50+ columns]
  │
  ↓
┌─────────────────────────┐
│ EDA: Data Understanding │
└─────────────────────────┘
  │ • Handle missing values → "Unknown/Invalid"
  │ • Analyze distributions
  │ • Identify feature types (numeric vs categorical)
  │
  ↓
Preprocessed Data Insights
  │
  ↓
┌─────────────────────────┐
│  RAY: Feature Pipeline  │
└─────────────────────────┘
  │ • Target encoding: (<30, >30) → YES, (NO) → NO
  │ • Numeric features → StandardScaler (mean=0, std=1)
  │ • Categorical features → OneHotEncoder (drop_first=True)
  │ • Train/test split: 80/20 stratified
  │
  ↓
Transformed Features
  │ [Scaled numerics + One-hot categoricals]
  │ [Shape: ~80K × 100+ features (after encoding)]
  │
  ↓
┌─────────────────────────┐
│ RAY: XGBoost Training   │
└─────────────────────────┘
  │ • 5-fold cross-validation
  │ • Hyperparameter tuning (30-50 trials)
  │ • Select best model by ROC-AUC
  │
  ↓
Trained Model Pipeline
  │ [ColumnTransformer + XGBClassifier]
  │ (saved as MLflow artifact)
  │
  ↓
┌─────────────────────────┐
│  DEPLOY: Model Export   │
└─────────────────────────┘
  │ • Serialize model + preprocessor
  │ • Package with MLflow dependencies
  │ • Containerize with Docker
  │
  ↓
Production Model Endpoint
  │
  ↓
┌─────────────────────────┐
│ Prediction Request      │
└─────────────────────────┘
  │ Input: {"age": 55, "time_in_hospital": 3, ...}
  │
  ↓
  │ • Automatic preprocessing (scaler + encoder)
  │ • XGBoost inference
  │
  ↓
  │ Output: [0.7234] (probability of readmission)
  │
  ↓
┌─────────────────────────┐
│   MONITOR: Drift Check  │
└─────────────────────────┘
  │ • Collect predictions + features
  │ • Compare distributions (baseline vs current)
  │ • Compute PSI per feature
  │ • Compute KS test p-values
  │
  ↓
Drift Detected? → Retrain (back to RAY)
No Drift? → Continue monitoring
```

### Execution Flows

#### Initial Deployment Flow

```
1. User runs EDA notebook
   └─→ Understand data characteristics

2. User runs RAY component
   └─→ python RAY/ray_tune_xgboost.py --data diabetic_data.csv
       └─→ Ray Tune spawns 30-50 trials
       └─→ Each trial: 5-fold CV on XGBoost
       └─→ Best model logged to MLflow

3. User runs DEPLOY component
   └─→ python DEPLOY/build_docker_image.py
       └─→ Query MLflow for best model
       └─→ Export model to ./model
       └─→ Generate Dockerfile
       └─→ Build Docker image
       └─→ docker run -p 5001:5001 diabetic-xgb:serve

4. Model is now live and serving predictions
```

#### Automated Monitoring Flow (via Airflow)

```
Daily at 02:00 UTC:
  │
  ├─→ Airflow Scheduler triggers "monitor_and_retrain" DAG
  │
  ↓
  ├─→ MONITOR component executes
  │     ├─→ Load baseline data (training set)
  │     ├─→ Load current data (recent production logs)
  │     ├─→ Score both datasets via API (http://localhost:5001/invocations)
  │     ├─→ Compute drift metrics:
  │     │     • PSI for each feature
  │     │     • KS test p-values
  │     │     • Prediction distribution PSI
  │     │
  │     ├─→ Evaluate multi-gate logic:
  │     │     • 30%+ features drifted?
  │     │     • Any feature PSI > 1.0?
  │     │     • Critical features drifted?
  │     │     • Prediction PSI > 0.2?
  │     │
  │     └─→ Decision:
  │           │
  │           ├─ NO DRIFT:
  │           │   └─→ Log "no drift" to JSON
  │           │   └─→ Exit (200 status)
  │           │
  │           └─ DRIFT DETECTED:
  │               └─→ Log "drift detected" to JSON
  │               └─→ Trigger subprocess: RAY/ray_tune_xgboost.py
  │                     └─→ Retrain with current data
  │                     └─→ New best model to MLflow
  │               └─→ Trigger subprocess: DEPLOY/build_docker_image.py
  │                     └─→ Export new model
  │                     └─→ Rebuild Docker image
  │               └─→ Optionally restart container
  │               └─→ Log "retraining complete" to JSON
  │
  ↓
Airflow logs task success/failure
  └─→ Next run: tomorrow at 02:00 UTC
```

#### Manual Retrain Flow (Without Drift)

```
User decides to retrain manually:
  │
  ├─→ python MONITOR/monitor_and_retrain.py \
  │     --force-retrain \
  │     --baseline data/diabetic_data.csv \
  │     --current data/new_data.csv
  │
  ↓
  ├─→ Skip drift detection
  ├─→ Directly trigger RAY component
  ├─→ Wait for retraining to complete
  ├─→ Trigger DEPLOY component
  └─→ New model deployed
```

### Storage & Artifacts

```
File System Layout:
───────────────────

RAY/
├── mlruns/                          # MLflow tracking store
│   └── <experiment_id>/
│       └── <run_id>/
│           ├── artifacts/
│           │   └── model/           # Serialized model + preprocessor
│           │       ├── MLmodel
│           │       ├── model.pkl
│           │       ├── conda.yaml
│           │       └── requirements.txt
│           ├── metrics/             # ROC-AUC, F1, etc.
│           ├── params/              # Hyperparameters
│           └── tags/                # Metadata
└── ray_exp/                         # Ray Tune results
    └── xgb_hpo/
        ├── best_config.json         # Best hyperparameters
        └── trainable_*/             # Individual trial results

DEPLOY/
├── model/                           # Exported model (from MLflow)
│   ├── MLmodel
│   ├── model.pkl
│   └── requirements.txt
└── Dockerfile                       # Generated Dockerfile

MONITOR/
└── monitoring/
    ├── out/
    │   └── drift_summary_*.json     # Drift detection results
    └── tmp/
        ├── ref_scored.csv           # Baseline scored data
        └── cur_scored.csv           # Current scored data
```

---

## Component Details

### 1. EDA - Exploratory Data Analysis

**Location**: `code/EDA/`

**Purpose**: Understand dataset characteristics and build baseline model.

**Key Activities**:
- Data loading, inspection, quality assessment
- Visualization of feature distributions and correlations
- Baseline Random Forest classifier (200 estimators)
- Feature importance analysis

**Key Insights**:
- Age distribution balanced, race distribution shows Caucasian majority
- Class imbalance exists (more non-readmitted)
- Top features: time_in_hospital, num_medications, num_lab_procedures

**Tools**: pandas, numpy, matplotlib, seaborn, scikit-learn

---

### 2. RAY - Hyperparameter Optimization

**Location**: `code/RAY/` | **Documentation**: [RAY/README.md](RAY/README.md)

**Purpose**: Large-scale hyperparameter optimization for XGBoost using Ray Tune.

**Features**:
- **Preprocessing**: StandardScaler (numeric) + OneHotEncoder (categorical), automatic class balancing
- **Search**: OptunaSearch (TPE algorithm) + ASHA scheduler (early stopping)
- **Hyperparameters**: 8 parameters optimized (n_estimators, max_depth, learning_rate, subsample, etc.)
- **Evaluation**: 5-fold stratified CV with multiple metrics (ROC-AUC, AP, Accuracy, F1)
- **Tracking**: MLflow integration with nested runs, artifacts, and best model

**Usage**:
```bash
python RAY/ray_tune_xgboost.py \
    --data /path/to/diabetic_data.csv \
    --num-samples 50 \
    --cpus-per-trial 4 \
    --test-size 0.2
```

**Docker Usage**:
```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  -v /home/ec2-user/projects/patient_selection/data:/data:ro \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py --data /data/diabetic_data.csv
```

**Output**:
- Best hyperparameters saved to `ray_exp/best_config.json`
- MLflow experiments in `mlruns/`
- Typical performance: ROC-AUC 0.70-0.75, F1 0.45-0.55

**Tools**: ray[tune], xgboost, scikit-learn, mlflow, optuna

---

### 3. DEPLOY - Model Deployment

**Location**: `code/DEPLOY/` | **Documentation**: [DEPLOY/README.md](DEPLOY/README.md)

**Purpose**: Package and deploy best model as production-ready Docker container with REST API.

**Key Components**:
- **`build_docker_image.py`**: Exports best model from MLflow, generates Dockerfile, builds container
- **Container**: Python 3.11-slim + MLflow serving framework + XGBoost 3.0+
- **API Endpoints**: `/invocations` (POST predictions), `/health`, `/version`

**Workflow**:
```bash
# Build image
python DEPLOY/build_docker_image.py \
    --tracking-uri file:///.../RAY/mlruns \
    --experiment xgb_diabetic_readmission_hpo \
    --image-tag diabetic-xgb:serve

# Run container
docker run --rm -p 5001:5001 diabetic-xgb:serve

# Make predictions
curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{"dataframe_records": [{"age": 55, "time_in_hospital": 3, ...}]}'
```

**Performance**: <100ms inference, 5-10s startup

**Tools**: docker, mlflow, gunicorn

---

### 4. MONITOR - Drift Detection & Auto-Retraining

**Location**: `code/MONITOR/` | **Documentation**: [MONITOR/README.md](MONITOR/README.md)

**Purpose**: Automated drift detection and intelligent retraining. **Unlike traditional monitoring, this system automatically fixes problems by retraining on fresh data.**

**Main Script**: `monitor_and_retrain.py`

**Workflow**:
1. Score baseline and current data through deployed model endpoint
2. Compute drift metrics (PSI, KS test) for features and predictions
3. Detect drift using multi-gate logic
4. Trigger automated retraining when thresholds exceeded
5. Rebuild Docker images after successful retraining
6. Log all results to JSON files

**Drift Detection**:

**PSI (Population Stability Index)**:
- < 0.1: No significant change
- 0.1-0.2: Small change (monitor)
- 0.2-0.25: Moderate drift ⚠️
- \> 0.25: Significant drift (action required)

**KS Test**: Compares distributions, p < 0.01 indicates significant difference

**Multi-Gate Trigger Logic**:
```python
trigger_retrain = (
    (share_of_drifted_features >= 30%) OR
    (max_feature_psi >= 1.0) OR           # Extreme drift
    (critical_feature_drifts) OR          # Important features
    (prediction_psi >= 0.2)               # Model output shifts
)
```

**Usage**:
```bash
python MONITOR/monitor_and_retrain.py \
  --baseline data/diabetic_data.csv \
  --current data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script RAY/ray_tune_xgboost.py \
  --tracking-uri file://.../RAY/mlruns \
  --build-script DEPLOY/build_docker_image.py \
  --ignore-cols "encounter_id,patient_nbr" \
  --critical-cols "number_inpatient" \
  --hpo-num-samples 10
```

**Output**: Drift summary JSON (`monitoring/out/drift_summary_*.json`) with detailed metrics

**Scheduling**: Use cron (daily at 2 AM) or Airflow DAG

**Tools**: scipy, pandas, numpy, requests

---

### 5. AIRFLOW - Pipeline Orchestration

**Location**: `code/airflow/` | **Documentation**: [airflow/README.md](airflow/README.md)

**Purpose**: Automate and orchestrate the entire ML pipeline with scheduled execution, dependency management, and monitoring.

**Components**:
- **DAG 1 - Deploy on Start** (`@once`): Initial model deployment
- **DAG 2 - Monitor and Retrain** (daily 02:00 UTC): Drift detection + retraining
- **Docker Compose**: PostgreSQL + Airflow services (scheduler, webserver, triggerer)

**Setup**:
```bash
# Generate environment file
cd code/airflow
bash set_airflow_env.sh

# Start Airflow
cd code
AIRFLOW_PROJ_DIR=$(pwd) docker-compose --env-file airflow/.env -f airflow/docker-compose.yaml up -d

# Access UI
# http://localhost:8080 (default: airflow/airflow)
```

**Advantages**:
- ✅ Automated scheduling (no manual cron)
- ✅ Dependency management
- ✅ Failure handling with retries
- ✅ Centralized logging
- ✅ Web UI for monitoring

**Resource Requirements**: Min 4GB RAM, 2 CPUs | Recommended 8GB RAM, 4+ CPUs

**Tools**: apache-airflow 3.0.1, postgresql, docker-compose

---

## Quick Start

### Complete Workflow

```bash
# 1. EDA
cd code/EDA/
jupyter notebook EDA.ipynb

# 2. Hyperparameter Optimization
cd ../RAY/
python ray_tune_xgboost.py \
    --data /home/ec2-user/projects/patient_selection/data/diabetic_data.csv \
    --num-samples 50 \
    --cpus-per-trial 4

# 3. Deploy Model
cd ../DEPLOY/
python build_docker_image.py \
    --tracking-uri file://.../RAY/mlruns \
    --experiment xgb_diabetic_readmission_hpo
docker run --rm -p 5001:5001 diabetic-xgb:serve

# 4. Monitor (in another terminal)
cd ../MONITOR/
python monitor_and_retrain.py \
  --baseline ../../data/diabetic_data.csv \
  --current ../../data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations

# 5. Orchestrate with Airflow
cd ../airflow/
bash set_airflow_env.sh
cd ..
AIRFLOW_PROJ_DIR=$(pwd) docker-compose --env-file airflow/.env -f airflow/docker-compose.yaml up -d
```

### MLflow UI

```bash
# On instance
mlflow ui --backend-store-uri file://.../code/RAY/mlruns --host 127.0.0.1 --port 5000

# From local machine (SSH tunnel)
ssh -i /path/to/key.pem -N -L 5001:127.0.0.1:5000 ec2-user@<EC2-DNS>
```

---

## Running with Docker

### Pull Image
```bash
docker pull abuchin/patient-env:1
```

### Usage Patterns

**Option 1: Code + External Data (Recommended)**
```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  -v /path/to/data:/data:ro \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py --data /data/diabetic_data.csv
```

**Option 2: Self-Contained**
```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py --data /work/data/diabetic_data.csv
```

**Jupyter Notebook**
```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  -p 8888:8888 \
  abuchin/patient-env:1 \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Docker Best Practices

1. **Use read-only mounts for data**: `-v /path/to/data:/data:ro`
2. **Mount code as editable**: `-v "$PWD":/work`
3. **Preserve outputs in mounted volumes**: `--ray-dir /work/ray_exp`
4. **Fix permissions if needed**: `sudo chown -R $USER:$USER ray_exp mlruns`

---

## Requirements

### Core Dependencies

```bash
# EDA
pandas>=1.3.0, numpy>=1.21.0, seaborn>=0.11.0, matplotlib>=3.3.0, scikit-learn>=1.0.0

# RAY
ray[tune]>=2.0.0, xgboost>=1.7.0, mlflow>=1.20.0, optuna>=3.0.0

# DEPLOY
docker, mlflow>=1.20.0, xgboost>=3.0.0, numpy>=2.3.0

# MONITOR
scipy>=1.16.0, pandas>=1.3.0, requests>=2.25.0

# AIRFLOW
apache-airflow==3.0.1, postgresql
```

### Installation

```bash
# Virtual environment
python3.13 -m venv patient_env
source patient_env/bin/activate

# Install all
pip install -r requirements.txt

# Or by component
pip install pandas numpy seaborn matplotlib scikit-learn jupyter  # EDA
pip install ray[tune] xgboost mlflow optuna  # RAY
pip install scipy requests  # MONITOR
```

---

## Expected Performance

### EDA Phase
- Dataset: ~100,000 records, 50+ features
- Baseline Random Forest: ~65-70% accuracy
- Top features: time_in_hospital, num_medications, num_lab_procedures

### RAY HPO Phase
- Trials: 30-50 configurations
- Duration: ~30-60 minutes (varies by CPUs)
- Best model: ROC-AUC 0.70-0.75, AP 0.30-0.40, F1 0.45-0.55
- Improvement: 5-10% over baseline

### DEPLOY Phase
- Docker image: ~500MB-1GB
- Build time: 2-5 minutes
- Inference: <100ms per prediction

### MONITOR Phase
- Drift detection: ~5-10 seconds per run
- Daily prediction logs in Parquet format
- Configurable thresholds (30% feature drift, p<0.05)

---

## Troubleshooting

### Common Issues

**RAY**: Out of memory → Reduce `num_samples` or `cpus_per_trial`  
**DEPLOY**: Port conflicts → Use different port with `--serve-port`  
**MONITOR**: No prediction logs → Ensure model is serving and predictions are made  
**AIRFLOW**: DAGs not appearing → Check mount paths and import errors

### Check Installation
```bash
python -c "import ray; import mlflow; import xgboost; print('All packages installed')"
```

### Check Logs
```bash
# Ray logs
cat ~/ray_results/*/progress.csv

# MLflow logs
mlflow runs list --experiment-name xgb_diabetic_readmission_hpo

# Docker logs
docker logs <container-id>

# Monitoring logs
cat code/MONITOR/monitoring/out/drift_summary_*.json
```

---

## Project Milestones

- [x] **Phase 1**: EDA (feature analysis, baseline model)
- [x] **Phase 2**: Hyperparameter Optimization (Ray Tune + MLflow)
- [x] **Phase 3**: Model Deployment (Docker + REST API)
- [x] **Phase 4**: Drift Detection & Auto-Retraining (PSI, KS test, multi-gate logic)
- [x] **Phase 5**: Pipeline Orchestration (Airflow DAGs, Docker Compose)
- [ ] **Phase 6**: Production Enhancement (real-time dashboard, A/B testing)
- [ ] **Phase 7**: User Interface (web app, LLM chatbot)

---

## Future Work

- Real-time monitoring dashboard (Streamlit/Grafana)
- HuggingFace deployment
- LLM-powered chatbot for dataset queries
- A/B testing framework
- Feature store integration
- SHAP-based drift explanation

---

## References

- **Dataset**: [Kaggle - Diabetes 130-US hospitals](https://www.kaggle.com/datasets/brandao/diabetes)
- **Ray Tune**: [docs.ray.io/tune](https://docs.ray.io/en/latest/tune/index.html)
- **MLflow**: [mlflow.org/docs](https://mlflow.org/docs/latest/index.html)
- **XGBoost**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)

---

## Contributing

1. **Code Style**: Follow PEP 8
2. **Documentation**: Update READMEs when adding features
3. **Testing**: Validate with sample data
4. **MLflow**: Log all experiments with descriptive names
5. **Docker**: Test containers locally before pushing

## License

This project is part of a patient readmission prediction system for healthcare applications.
