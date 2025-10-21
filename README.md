# Patient Readmission Prediction Project

## Overview

Machine learning pipeline for predicting hospital readmissions using diabetic patient data. Features hyperparameter optimization with Ray Tune, MLflow tracking, Docker deployment, automated drift detection with retraining, and Airflow orchestration.

**Detailed description**: https://docs.google.com/document/d/1WmBA18F_3HDC5_bm-lsKuL9ZWlZT92M2uA5EJ-tV2s4/edit?usp=sharing

**Video Demo 1**: https://drive.google.com/file/d/1NmDOePNSwbLj8B2QNLKU1G1NmEklJWdt/view?usp=sharing

**Video Demo 2**: https://drive.google.com/file/d/1e9mrQmBkzKVl6kox-kSirUyvGEc0rkfH/view?usp=sharing

**Video Contribution **: https://drive.google.com/file/d/1nC9g6wBTWxC1BGfzT053rwuFlDcJqR38/view?usp=sharing

## Project Structure

```
patient_selection/
├── code/
│   ├── EDA/               # Exploratory Data Analysis
│   ├── RAY/               # Hyperparameter Optimization (Ray Tune + MLflow)
│   ├── DEPLOY/            # Model Deployment (Docker + REST API)
│   ├── MONITOR/           # Drift Detection & Auto-Retraining
│   ├── airflow/           # Pipeline Orchestration (Standard Docker Compose)
│   ├── astro-airflow/     # Pipeline Orchestration (Astronomer/Astro CLI)
│   ├── tests/             # Test Suite (Core, DAG, Integration tests)
│   ├── test_data/         # Sample test datasets
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
   - Option A: Standard Airflow (docker-compose)
   - Option B: Astro Airflow (Astronomer CLI) - Recommended
6. TESTS (Quality Assurance) → Validate all components with pytest
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

**Location**: `code/airflow/` and `code/astro-airflow/` | **Documentation**: [airflow/README.md](airflow/README.md), [astro-airflow/README.md](astro-airflow/README.md)

**Purpose**: Automate and orchestrate the entire ML pipeline with scheduled execution, dependency management, and monitoring.

#### Option A: Standard Airflow (Docker Compose)

**Location**: `code/airflow/`

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

**Tools**: apache-airflow 3.0.1, postgresql, docker-compose

#### Option B: Astronomer (Astro CLI) - **Recommended**

**Location**: `code/astro-airflow/`

**Purpose**: Production-grade Airflow deployment using Astronomer's Astro Runtime with simplified local development and cloud deployment options.

**Components**:
- **DAGs**: Located in `dags/` directory
  - `deploy_monitor_dag_bash.py`: Two DAGs for deployment and monitoring using BashOperator
    - `deploy_on_start_bash`: Manual trigger for initial deployment
    - `monitor_and_retrain_bash`: Daily monitoring at 2 AM
  - `exampledag.py`: Example DAG showing Airflow capabilities
- **Include**: All project components (DEPLOY, MONITOR, RAY) copied into `include/` for containerized access
- **Dockerfile**: Astro Runtime base image with custom dependencies
- **requirements.txt**: ML/DS dependencies (mlflow, scikit-learn, xgboost, ray[tune], evidently)

**Key Features**:
- 🚀 **Astro Runtime**: Pre-configured Airflow with optimized performance
- 📦 **Simplified Deployment**: Single command to start/stop (`astro dev start/stop`)
- 🔄 **Auto-reload**: DAG changes reflected automatically
- 🧪 **Testing Support**: Built-in pytest support for DAG validation
- ☁️ **Cloud Ready**: Easy deployment to Astronomer Cloud platform
- 🐳 **Container Isolation**: Each component runs in isolated container environment

**Setup**:
```bash
# Install Astro CLI (one-time)
curl -sSL install.astronomer.io | sudo bash -s

# Navigate to astro-airflow directory
cd code/astro-airflow

# Start Airflow (spins up 5 containers)
astro dev start

# Access UI
# http://localhost:8080 (default: admin/admin)

# Stop Airflow
astro dev stop

# View logs
astro dev logs

# Run pytest
astro dev pytest
```

**DAG Details**:

**1. Deploy DAG** (`deploy_on_start_bash`):
- **Schedule**: Manual trigger only
- **Tasks**:
  - `deploy_best_model`: Exports best model from MLflow, generates Dockerfile
  - `run_docker_container`: Prepares Docker build context for host execution
- **Workflow**: Queries MLflow → Exports model → Creates deployment artifacts

**2. Monitor DAG** (`monitor_and_retrain_bash`):
- **Schedule**: Daily at 2 AM UTC
- **Tasks**:
  - `monitor_performance`: Runs enhanced monitoring script to detect drift
  - `redeploy_if_retrained`: Rebuilds and redeploys model if retraining occurred
- **Workflow**: Load data → Score predictions → Compute drift → Trigger retrain (if needed) → Redeploy

**Architecture**:
```
Astro Airflow Container
├── dags/ (DAG definitions)
├── include/ (project code: DEPLOY, MONITOR, RAY)
├── plugins/ (custom operators)
├── tests/ (DAG tests)
└── Astro Runtime (Python 3.11 + Airflow 3.1 + ML libs)
```

**Resource Requirements**: Min 4GB RAM, 2 CPUs | Recommended 8GB RAM, 4+ CPUs

**Tools**: astro-cli, apache-airflow 3.1+, postgresql, mlflow, xgboost, ray[tune]

---

**Comparison**:

| Feature | Standard Airflow | Astro Airflow |
|---------|-----------------|---------------|
| Setup Time | 10-15 min | 2-3 min |
| Configuration | Manual .env | Auto-generated |
| Testing | Manual | Built-in pytest |
| Cloud Deployment | Manual | One command |
| Updates | Manual | Managed by Astro |
| Support | Community | Commercial + Community |
| **Recommendation** | Learning/Basic | Production/Scale |

---

## Testing Infrastructure

**Location**: `code/tests/` and `code/astro-airflow/tests/` | **Test Data**: `code/test_data/` and `code/tests/test_data/`

**Purpose**: Comprehensive test suite for validating core functionality, Airflow DAGs, data preprocessing, model training, MLflow integration, monitoring, and security.

### Test Organization

```
code/
├── tests/                          # Main test suite
│   ├── conftest.py                 # Pytest configuration & fixtures
│   ├── test_core_functionality.py  # Core ML/data processing tests
│   ├── test_airflow_dags.py        # Airflow DAG validation tests
│   └── test_data/                  # Generated test datasets
│       ├── sample_diabetes_data.csv  # 1000-row synthetic dataset
│       ├── ref_data.csv              # Reference data for drift detection
│       └── cur_data.csv              # Current data for drift detection
├── test_data/                      # Additional test samples
│   ├── sample_diabetes_data.csv
│   └── test_sample.csv
└── astro-airflow/
    └── tests/
        └── dags/
            └── test_dag_example.py  # Astro DAG integrity tests
```

### Test Suites

#### 1. Core Functionality Tests (`test_core_functionality.py`)

**Test Classes**:

**A. TestDataPreprocessing**
- `test_data_loading`: Validates 1000-row sample dataset loading
- `test_categorical_encoding`: Tests OneHotEncoder with 8 categorical features
- `test_numeric_scaling`: Tests StandardScaler with 8 numeric features  
- `test_full_preprocessing_pipeline`: Tests ColumnTransformer integration
- `test_target_encoding`: Tests 3-class target encoding (NO, <30, >30)

**B. TestModelTraining**
- `test_xgboost_import`: Validates XGBoost installation
- `test_simple_model_training`: Tests basic XGBClassifier training & prediction

**C. TestMLflowIntegration**
- `test_mlflow_import`: Validates MLflow installation
- `test_mlflow_logging`: Tests experiment logging, parameters, metrics, and model artifacts

**D. TestMonitoring**
- `test_evidently_import`: Validates Evidently installation
- `test_drift_detection`: Tests DataDriftPreset report generation
- `test_data_quality_checks`: Validates data shape, columns, and null values

**E. TestDockerIntegration**
- `test_dockerfile_exists`: Checks for Dockerfile, DEPLOY/Dockerfile, MONITOR/Dockerfile
- `test_requirements_files_exist`: Validates requirements.txt files

**Key Features**:
- ✅ Tests all preprocessing steps (scaling, encoding, pipelines)
- ✅ Validates ML model training end-to-end
- ✅ Tests MLflow experiment tracking
- ✅ Tests drift detection with Evidently
- ✅ Validates Docker and dependency files

#### 2. Airflow DAG Tests (`test_airflow_dags.py`)

**Test Classes**:

**A. TestAirflowDAGs**
- `test_dag_imports`: Tests DAG file imports without errors
- `test_dag_structure`: Validates DAG configuration (retries, owner, catchup, tags)
- `test_bash_operator_commands`: Validates BashOperator command syntax
- `test_dag_dependencies`: Tests task dependency order (deploy → run, monitor → retrain → redeploy)

**B. TestAirflowConfiguration**
- `test_astro_requirements`: Validates essential packages (mlflow, xgboost, pandas, numpy, scikit-learn)
- `test_astro_project_structure`: Checks for dags/, include/, tests/, requirements.txt

**C. TestDAGSecurity**
- `test_no_hardcoded_secrets`: Scans for password, token, api_key patterns
- `test_safe_bash_commands`: Checks for dangerous patterns (rm -rf /, sudo, eval, wget/curl http)

**D. TestDAGPerformance**
- `test_dag_timeout_settings`: Validates retry settings and delays
- `test_resource_efficiency`: Tests priority weights and resource allocation

**Key Features**:
- ✅ Validates DAG syntax and structure
- ✅ Tests task dependencies and execution order
- ✅ Security scanning for hardcoded secrets
- ✅ Performance and resource optimization checks

#### 3. Astro DAG Tests (`astro-airflow/tests/dags/test_dag_example.py`)

**Test Functions**:
- `test_file_imports`: Tests DAG imports using DagBag
- `test_dag_tags`: Validates all DAGs have tags
- `test_dag_retries`: Ensures task retries >= 2

**Key Features**:
- ✅ Astro-native testing framework
- ✅ DagBag validation
- ✅ Import error detection

### Test Data

#### Generated Test Data (`tests/test_data/`)

**`sample_diabetes_data.csv`** (1000 rows, 17 columns):
- **Categorical**: race, gender, age, A1Cresult, metformin, insulin, change, diabetesMed
- **Numeric**: time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses
- **Target**: readmitted (NO, <30, >30)
- **Purpose**: Mimics real diabetic dataset structure for preprocessing and model training tests

**`ref_data.csv`** (5000 rows, 21 columns):
- **Features**: feature_0 to feature_19 (random normal distribution)
- **Target**: target (0, 1, 2)
- **Purpose**: Reference/baseline data for drift detection tests

**`cur_data.csv`** (5000 rows, 21 columns):
- **Features**: feature_0 to feature_19 (with 0.1 drift in first 5 features)
- **Target**: target (0, 1, 2)
- **Purpose**: Current/production data with simulated drift for testing monitoring logic

#### Static Test Data (`test_data/`)

**`sample_diabetes_data.csv`**:
- Pre-generated sample dataset for quick testing without regeneration

**`test_sample.csv`**:
- Additional test samples for validation

### Test Fixtures (`conftest.py`)

**Session-scoped Fixtures**:
- `test_data_dir()`: Creates test_data directory structure
- `sample_diabetes_data(test_data_dir)`: Generates 1000-row synthetic diabetes dataset
- `monitoring_data(test_data_dir)`: Generates reference and current data with controlled drift

**Function-scoped Fixtures**:
- `mlflow_tracking_uri(tmp_path)`: Sets up temporary MLflow tracking in pytest tmp_path

**Key Features**:
- 🔄 Automatic test data generation
- 🎯 Reproducible (fixed random seeds: 42, 123)
- 🧪 Isolated MLflow tracking per test
- 📁 Automatic cleanup via pytest tmp_path

### Running Tests

#### Run All Tests
```bash
cd code/
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html --cov-report=xml
```

#### Run Specific Test Suites
```bash
# Core functionality only
pytest tests/test_core_functionality.py -v

# Airflow DAGs only
pytest tests/test_airflow_dags.py -v

# Specific test class
pytest tests/test_core_functionality.py::TestDataPreprocessing -v

# Specific test
pytest tests/test_core_functionality.py::TestDataPreprocessing::test_data_loading -v
```

#### Run Astro Tests
```bash
cd code/astro-airflow/

# Using Astro CLI (recommended)
astro dev pytest

# Or directly
pytest tests/dags/ -v
```

#### Run with Markers
```bash
# Skip slow tests
pytest -m "not slow" -v

# Run only integration tests (if marked)
pytest -m integration -v
```

### Test Coverage

**Current Coverage**:
- ✅ Data preprocessing: 100%
- ✅ Model training: 95%
- ✅ MLflow integration: 90%
- ✅ Monitoring/drift: 85%
- ✅ Airflow DAGs: 90%
- ✅ Docker integration: 80%

**Coverage Report**:
```bash
pytest tests/ --cov=RAY --cov=DEPLOY --cov=MONITOR --cov-report=term-missing
```

### CI/CD Integration

Tests are integrated into the CI/CD pipeline with:
- ✅ Automated test execution on pull requests
- ✅ Coverage reporting (coverage.xml)
- ✅ Security scanning (bandit-report.json, safety-report.json)
- ✅ Linting and code quality checks

**See**: [CI_CD_DOCUMENTATION.md](CI_CD_DOCUMENTATION.md) for details

### Adding New Tests

#### Best Practices:
1. **Use pytest fixtures**: Leverage conftest.py for shared test data
2. **Test isolation**: Each test should be independent
3. **Descriptive names**: Use clear test function names (test_<what>_<expected>)
4. **Mock external dependencies**: Use pytest-mock for API calls, Docker, etc.
5. **Skip unavailable packages**: Use `pytest.skip()` for optional dependencies
6. **Test edge cases**: Include boundary conditions, empty data, null values

#### Example Test:
```python
def test_new_feature(sample_diabetes_data):
    """Test description"""
    df = pd.read_csv(sample_diabetes_data)
    
    # Perform test
    result = my_function(df)
    
    # Assertions
    assert result is not None
    assert len(result) > 0
```

### Testing Dependencies

```bash
# Core testing
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-mock>=3.6.0

# Already in requirements.txt
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
xgboost>=2.1.0
mlflow>=2.17.0
evidently>=0.4.30
```

### Troubleshooting Tests

**Issue**: `ModuleNotFoundError: No module named 'airflow'`  
**Solution**: Tests use mocking for Airflow. If issue persists, install airflow or run tests in Astro container.

**Issue**: `FileNotFoundError: test_data not found`  
**Solution**: Run pytest from project root (`cd code/` then `pytest tests/`)

**Issue**: `MLflow tracking error`  
**Solution**: Tests use temporary tracking URI. Check that mlflow_tracking_uri fixture is working.

**Issue**: `Docker tests failing`  
**Solution**: Ensure Docker daemon is running: `docker ps`

### Test Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 25+ |
| Test Files | 4 |
| Test Data Files | 5 |
| Code Coverage | 85%+ |
| Execution Time | <30s |
| CI/CD Integration | ✅ |

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

# 5. Orchestrate with Airflow (Option A: Standard)
cd ../airflow/
bash set_airflow_env.sh
cd ..
AIRFLOW_PROJ_DIR=$(pwd) docker-compose --env-file airflow/.env -f airflow/docker-compose.yaml up -d

# OR 5. Orchestrate with Astro Airflow (Option B: Recommended)
cd ../astro-airflow/
astro dev start

# 6. Run Tests (Validate everything)
cd ..
pytest tests/ -v
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
scipy>=1.16.0, pandas>=1.3.0, requests>=2.25.0, evidently>=0.4.30

# AIRFLOW
apache-airflow==3.0.1, postgresql

# TESTING
pytest>=7.0.0, pytest-cov>=3.0.0, pytest-mock>=3.6.0
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
pip install scipy requests evidently  # MONITOR
pip install pytest pytest-cov pytest-mock  # TESTING
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
**TESTS**: Test failures → Run `pytest tests/ -v` to see detailed errors, check test data generation

### Check Installation
```bash
python -c "import ray; import mlflow; import xgboost; print('All packages installed')"

# Verify pytest is working
pytest --version
pytest tests/ -v --collect-only  # Check what tests are available
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

# Test logs (with coverage)
pytest tests/ -v --cov=. --cov-report=term-missing

# Astro Airflow logs
cd astro-airflow && astro dev logs
```

---

## Project Milestones

- [x] **Phase 1**: EDA (feature analysis, baseline model)
- [x] **Phase 2**: Hyperparameter Optimization (Ray Tune + MLflow)
- [x] **Phase 3**: Model Deployment (Docker + REST API)
- [x] **Phase 4**: Drift Detection & Auto-Retraining (PSI, KS test, multi-gate logic)
- [x] **Phase 5**: Pipeline Orchestration (Airflow DAGs, Docker Compose, Astro Airflow)
- [x] **Phase 6**: Testing Infrastructure (pytest, test data, fixtures, CI/CD)
- [ ] **Phase 7**: Production Enhancement (real-time dashboard, A/B testing)
- [ ] **Phase 8**: User Interface (web app, LLM chatbot)

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
