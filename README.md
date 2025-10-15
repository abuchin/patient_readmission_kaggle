# Patient Readmission Prediction Project

## Overview

This project aims to create a machine learning model that can predict whether a given patient will be readmitted to the hospital based on collected patient data. The model will be tuned using Ray, orchestrated using Airflow, and deployed on HuggingFace. Additionally, it will be connected to a website where users can ask questions about the dataset and get answers from an LLM.

## Detailed description
https://docs.google.com/document/d/1WmBA18F_3HDC5_bm-lsKuL9ZWlZT92M2uA5EJ-tV2s4/edit?usp=sharing

## Project Structure

```
patient_selection/
├── code/
│   ├── EDA/               # Exploratory Data Analysis
│   │   ├── EDA.ipynb      # Interactive analysis notebook
│   │   └── README.md      # EDA documentation
│   ├── RAY/               # Hyperparameter Optimization
│   │   ├── ray_tune_xgboost.py  # Ray Tune HPO script
│   │   ├── mlruns/        # MLflow tracking store
│   │   ├── ray_exp/       # Ray Tune experiment results
│   │   └── README.md      # RAY component documentation
│   ├── DEPLOY/            # Model Deployment
│   │   ├── build_docker_image.py  # Docker build automation
│   │   ├── best_model_show.py     # Model inspection utility
│   │   ├── Dockerfile     # Container definition
│   │   ├── model/         # Exported MLflow model
│   │   └── README.md      # Deployment documentation
│   ├── MONITOR/           # Automated Drift Detection & Retraining
│   │   ├── monitor_and_retrain.py # Main drift detection + auto-retrain script
│   │   ├── monitoring/    # Monitoring data
│   │   │   ├── out/       # Drift summary JSON reports
│   │   │   └── tmp/       # Temporary scoring files
│   │   └── README.md      # MONITOR component documentation
│   ├── airflow/           # Pipeline Orchestration
│   │   ├── dags/          # Airflow DAG definitions
│   │   │   └── deploy_monitor_dag.py  # Deploy + Monitor DAGs
│   │   ├── docker-compose.yaml  # Airflow services setup
│   │   ├── Dockerfile     # Custom Airflow image
│   │   ├── set_airflow_env.sh   # Environment configuration
│   │   └── README.md      # Airflow setup guide
│   ├── requirements.txt   # Python dependencies
│   └── README.md          # This file
├── data/
│   ├── diabetic_data.csv  # Main dataset
│   └── readmission.zip    # Additional data files
└── README.md              # Project overview
```

## Dataset

The project uses a diabetic patient dataset (`diabetic_data.csv`) that contains various patient attributes and outcomes. The dataset includes:

- **Demographic information**: Gender, race, age
- **Medical procedures**: Number of lab procedures, medications
- **Diagnostic information**: Number of diagnoses, inpatient visits
- **Target variable**: Readmission status

The dataset could be found on kaggle. The file name diabetic_data.csv is what we want and would reuse for the whole project.
https://www.kaggle.com/datasets/brandao/diabetes

### Data Characteristics

Based on the exploratory data analysis:

1. **Missing Data Handling**: Features with missing values are replaced with "Unknown/Invalid" values
2. **Distribution Patterns**:
   - Some numeric features (num_lab_procedures, num_medications) are close to normally distributed
   - Other features (number_diagnoses, number_inpatient) are not normally distributed
   - This suggests that non-linear methods might be beneficial for pattern recognition

3. **Categorical Features**:
   - Most patients are Caucasian, with smaller representation of other race groups
   - Age distribution is relatively equal across different groups
   - Drug usage distribution is uneven, with some patients having taken drugs while others haven't

## Methodology

### Data Preprocessing

1. **Target Variable**: The `readmitted` column serves as the target variable
2. **Feature Engineering**: Categorical features are converted using one-hot encoding
3. **Data Splitting**: 
   - 80% training, 20% testing
   - Stratified splitting to maintain equal representation of readmitted/not-readmitted patients
   - Training data is further split into two halves for potential model retraining in case of data drift

### Model Development

The project implements a **Random Forest Classifier** with the following configuration:
- 200 estimators (trees)
- No maximum depth limit (trees expand fully)
- Balanced class weights to handle class imbalance
- Multi-core processing enabled

### Model Evaluation

The model performance is evaluated using:
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Detailed metrics including precision, recall, and F1-score
- **Confusion Matrix**: Visual representation of prediction vs. actual outcomes
- **Feature Importance**: Analysis of which features contribute most to predictions

## Key Findings

1. **Feature Importance**: The model identifies the top 20 most important features for predicting readmission
2. **Model Performance**: The Random Forest model provides baseline performance metrics for patient readmission prediction
3. **Data Quality**: The dataset contains various data quality considerations that need to be addressed during preprocessing

## Project Components

### 1. EDA - Exploratory Data Analysis

**Location**: `code/EDA/`

**Purpose**: Comprehensive exploration and initial modeling of the diabetic readmission dataset.

**Key Activities**:
- **Data Loading & Inspection**: Load diabetic_data.csv and examine structure, data types, and basic statistics
- **Data Quality Assessment**: Identify missing values, outliers, and data distribution patterns
- **Visualization**: Create plots to understand feature distributions, correlations, and relationships with target variable
- **Feature Analysis**:
  - Numeric features: num_lab_procedures, num_medications, number_diagnoses, number_inpatient
  - Categorical features: race, gender, age groups, medication usage
  - Distribution analysis: Normal vs non-normal distributions
- **Baseline Modeling**: Initial Random Forest classifier with 200 estimators
- **Performance Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Feature Importance**: Identify top predictive features for readmission

**Key Insights**:
- Missing data is handled by replacing with "Unknown/Invalid" values
- Some features are normally distributed, others are skewed (suggesting non-linear methods)
- Class imbalance exists between readmitted and non-readmitted patients
- Age distribution is relatively balanced, but race distribution shows Caucasian majority
- Feature importance analysis reveals which patient attributes are most predictive

**Output**: 
- Cleaned understanding of dataset characteristics
- Baseline model performance metrics
- Feature importance rankings
- Data preprocessing strategy for production pipeline

**Tools Used**: pandas, numpy, matplotlib, seaborn, scikit-learn

---

### 2. RAY - Hyperparameter Optimization

**Location**: `code/RAY/`

**Purpose**: Large-scale hyperparameter optimization using Ray Tune to find the best XGBoost model configuration.

**Key Features**:

#### Data Processing
- **Target Encoding**: Merges readmission labels (`<30` and `>30` → `YES`, keeps `NO`) for binary classification
- **Feature Preprocessing**:
  - StandardScaler for numeric features (age, lab procedures, medications, etc.)
  - OneHotEncoder for categorical features (race, gender, medication types) with drop_first=True
  - Unknown category handling for robust inference
- **Class Balancing**: Automatically computes `scale_pos_weight` to handle class imbalance

#### Hyperparameter Search
- **Model**: XGBoost Classifier (gradient boosting framework)
- **Search Space**: 8 hyperparameters optimized simultaneously
  - n_estimators: 200-900 trees
  - max_depth: 3-10 levels
  - learning_rate: 0.001-0.3 (log scale)
  - subsample: 0.6-1.0 (row sampling)
  - colsample_bytree: 0.6-1.0 (column sampling)
  - min_child_weight: 0.1-10 (log scale)
  - reg_alpha: 1e-8 to 1e-1 (L1 regularization)
  - reg_lambda: 1e-2 to 1e1 (L2 regularization)
- **Search Algorithm**: OptunaSearch with Tree-structured Parzen Estimator (TPE) for intelligent sampling
- **Scheduler**: ASHA (Asynchronous Successive Halving) for early stopping of poor trials

#### Evaluation Strategy
- **Cross-Validation**: 5-fold stratified CV to ensure robust performance estimates
- **Metrics Tracked**:
  - ROC-AUC: Area under ROC curve (primary metric)
  - Average Precision: Area under precision-recall curve
  - Accuracy: Overall prediction correctness
  - F1-Score: Harmonic mean of precision and recall
- **Trials**: Configurable (default 30, recommended 50-100 for production)

#### Experiment Tracking
- **MLflow Integration**: Complete experiment tracking with nested runs
  - Every trial logs hyperparameters and validation metrics
  - Best model logs test metrics and full pipeline
  - Model artifacts saved with preprocessing pipeline
- **Ray Results**: Checkpoints and trial histories saved to `ray_exp/`
- **Best Configuration**: Saved as JSON for reproducibility

#### Resource Management
- **Parallelization**: Multiple trials run simultaneously across available CPUs
- **Resource Allocation**: Configurable CPUs/GPUs per trial
- **Storage**: Efficient result storage with automatic cleanup

**Workflow**:
```bash
python ray_tune_xgboost.py \
    --data /path/to/diabetic_data.csv \
    --num-samples 50 \
    --cpus-per-trial 4 \
    --test-size 0.2
```

**Output**:
- Best hyperparameter configuration (saved to `ray_exp/best_config.json`)
- Complete MLflow experiment with all trials (`mlruns/`)
- Best model trained on full training set, evaluated on test set
- Comprehensive metrics: test_auc, test_ap, test_acc, test_f1

**Tools Used**: ray[tune], xgboost, scikit-learn, mlflow, optuna

**Documentation**: See [RAY/README.md](RAY/README.md) for detailed usage instructions.

---

### 3. DEPLOY - Model Deployment with Docker

**Location**: `code/DEPLOY/`

**Purpose**: Package and deploy the best model as a production-ready Docker container with REST API serving.

**Key Components**:

#### Build Automation (`build_docker_image.py`)
- **Model Export**: Automatically finds and exports the best model from MLflow tracking store
- **Dockerfile Generation**: Creates optimized Dockerfile with proper dependencies
- **Docker Build**: Builds container image with all required runtime dependencies
- **Configuration**: Flexible options for ports, image tags, and MLflow URIs

#### Container Specification
- **Base Image**: Python 3.11-slim (required for numpy>=2.3 compatibility)
- **System Dependencies**: libgomp1 for XGBoost parallel processing
- **Python Environment**: 
  - MLflow (model serving framework)
  - XGBoost 3.0+ (model inference)
  - scikit-learn (preprocessing pipeline)
  - pandas, numpy, scipy (data handling)
- **Optimizations**:
  - Environment variables to prevent over-threading (OMP_NUM_THREADS=1)
  - No-cache pip installs for smaller image size
  - Minimal system packages for security and efficiency

#### Model Serving
- **Framework**: MLflow Models Serve (built-in REST API)
- **Host**: 0.0.0.0 (accessible from outside container)
- **Port**: 5001 (configurable, default to avoid conflicts with MLflow UI on 5000)
- **Endpoints**:
  - `/invocations` - POST predictions (primary endpoint)
  - `/health` - GET health check
  - `/version` - GET model version info

#### API Interface
**Request Format**:
```json
{
  "dataframe_records": [
    {
      "age": 55,
      "time_in_hospital": 3,
      "num_lab_procedures": 45,
      "num_medications": 15,
      "race": "Caucasian",
      "gender": "Female"
      // ... all required features
    }
  ]
}
```

**Response Format**:
```json
[0.7234]  // Probability of readmission (YES class)
```

#### Deployment Workflow

**Step 1: Build Docker Image**
```bash
python build_docker_image.py \
    --tracking-uri file://.../RAY/mlruns \
    --experiment xgb_diabetic_readmission_hpo \
    --image-tag diabetic-xgb:serve \
    --serve-port 5001
```

**Step 2: Run Container Locally**
```bash
docker run --rm -p 5001:5001 diabetic-xgb:serve
```

**Step 3: Make Predictions**
```bash
curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{"dataframe_records": [{"feature1": value, ...}]}'
```

#### Production Deployment Options
- **Docker Hub**: Push to public/private registry
- **AWS ECR**: Deploy to Amazon Elastic Container Registry
- **EC2 Instance**: Run container on cloud VM with SSH tunnel access
- **Kubernetes**: Scale with container orchestration (future)
- **Docker Compose**: Multi-container orchestration

#### Monitoring & Management
- **Logging**: Docker logs capture all requests and errors
- **Health Checks**: Built-in /health endpoint for monitoring
- **Resource Limits**: Configurable memory and CPU constraints
- **Background Running**: Daemon mode with restart policies

**Tools Used**: docker, mlflow, gunicorn (optional), nginx (reverse proxy, optional)

**Documentation**: See [DEPLOY/README.md](DEPLOY/README.md) for comprehensive deployment guide.

---

### 4. MONITOR - Automated Drift Detection & Model Retraining

**Location**: `code/MONITOR/`

**Purpose**: Automated drift detection and intelligent retraining orchestration. When data drift is detected, automatically triggers model retraining and optionally rebuilds deployment images.

**Key Innovation**: Unlike traditional monitoring that only alerts, this system **automatically fixes the problem** by retraining the model on fresh data.

**Main Script**: `monitor_and_retrain.py` (350 lines)

#### What It Does

The monitoring pipeline:
1. ✅ **Scores both baseline and current data** through your deployed model endpoint
2. ✅ **Computes drift metrics** using statistical tests (PSI, KS test)
3. ✅ **Detects drift** in both features and predictions
4. ✅ **Triggers automated retraining** when drift thresholds are exceeded
5. ✅ **Rebuilds Docker images** after successful retraining (optional)
6. ✅ **Logs all results** to JSON files for tracking

#### Drift Detection Methods

**PSI (Population Stability Index)**:
```
PSI = Σ (P_current - P_baseline) × ln(P_current / P_baseline)

Interpretation:
- PSI < 0.1  : No significant change
- PSI 0.1-0.2: Small change (monitor)
- PSI 0.2-0.25: Moderate drift ⚠️
- PSI > 0.25 : Significant drift (action required)
```

**KS Test (Kolmogorov-Smirnov)**:
- Compares empirical cumulative distribution functions
- Returns p-value (probability distributions are the same)
- Threshold: p < 0.01 indicates significant difference

**Multi-Gate Logic**:
```python
trigger_retrain = (
    (share_of_drifted_features >= 30%) OR
    (max_feature_psi >= 1.0) OR            # Any feature extreme drift
    (critical_feature_drifts) OR            # Important features
    (prediction_psi >= 0.2)                 # Model output shifts
)
```

#### Key Features

**Critical Features Monitoring**: Monitor key clinical features more strictly
```bash
--critical-cols "number_inpatient,time_in_hospital"
--critical-psi-thresh 0.3
```

**Extreme Drift Gate**: Trigger if ANY single feature shows extreme drift
```bash
--any-feature-psi-thresh 0.5
```

**Prediction Drift**: Double-check prediction distribution shifts
```bash
--pred-psi-thresh 0.2
--pred-ks-p-thresh 0.01
```

**Flexible Thresholds**: All thresholds are configurable
- `--feature-psi-thresh 0.2` (feature drift)
- `--drift-share-thresh 0.30` (30% features must drift)
- `--ks-p-thresh 0.01` (KS test significance)

#### Usage Example

```bash
python code/MONITOR/monitor_and_retrain.py \
  --baseline data/diabetic_data.csv \
  --current data/diabetic_data_drift.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py \
  --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
  --build-script code/DEPLOY/build_docker_image.py \
  --ignore-cols "encounter_id,patient_nbr" \
  --critical-cols "number_inpatient" \
  --hpo-num-samples 10
```

#### Workflow

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

#### Output Files

**Drift Summary JSON** (`monitoring/out/drift_summary_*.json`):
```json
{
  "trigger_retrain": false,
  "share_drifted": 0.0392,
  "n_features": 51,
  "n_drifted": 2,
  "max_feature_psi": 0.0,
  "pred_psi": 0.0,
  "pred_gate": false,
  "feature_psi_thresh": 0.2,
  "ks_p_thresh": 0.01,
  "drift_share_thresh": 0.3,
  "details": [
    {"column": "time_in_hospital", "type": "numeric", "psi": 0.0, "ks_p": 1.0, "drift": false},
    {"column": "num_medications", "type": "numeric", "psi": 0.1234, "ks_p": 0.05, "drift": false},
    ...
  ]
}
```

#### Integration with Pipeline

**With EDA**: Uses training baseline as reference data  
**With RAY**: Automatically triggers `ray_tune_xgboost.py` when drift detected  
**With DEPLOY**: Rebuilds Docker images after successful retraining  
**With MLflow**: Shares same tracking store for experiment continuity  

#### Scheduling Options

**Cron (Daily at 2 AM)**:
```bash
0 2 * * * cd /home/ec2-user/projects/patient_selection && \
  python code/MONITOR/monitor_and_retrain.py \
  --baseline data/diabetic_data.csv \
  --current /tmp/production_data.csv \
  --endpoint http://localhost:5001/invocations \
  --retrain-script code/RAY/ray_tune_xgboost.py
```

**Airflow DAG**:
```python
monitor_task = BashOperator(
    task_id='monitor_and_retrain',
    bash_command='python code/MONITOR/monitor_and_retrain.py ...'
)
```

#### Advanced Features

- **Critical features** with stricter thresholds
- **Extreme drift** detection (any feature PSI > threshold)
- **Prediction drift** with dual tests (PSI + KS)
- **Column ignoring** (skip IDs, timestamps)
- **Force retraining** mode (bypass drift checks)
- **Fast retraining** (fewer HPO trials for emergencies)
- **Robust prediction parsing** (handles various API formats)

**Tools Used**: scipy (statistical tests), pandas, numpy, requests, subprocess

**Documentation**: See [MONITOR/README.md](MONITOR/README.md) for complete usage guide (1,200+ lines).

---

### 5. AIRFLOW - Pipeline Orchestration & Automation

**Location**: `code/airflow/`

**Purpose**: Automate and orchestrate the entire ML pipeline using Apache Airflow. Provides scheduled execution of deployment and monitoring tasks with dependency management, logging, and failure handling.

**Key Innovation**: Transforms manual pipeline steps into automated, scheduled workflows with proper dependency management and monitoring.

#### Components

```
airflow/
├── dags/
│   └── deploy_monitor_dag.py    # Airflow DAG definitions (95 lines)
├── config/
│   └── airflow.cfg               # Airflow configuration
├── docker-compose.yaml           # Multi-container Airflow setup (295 lines)
├── Dockerfile                    # Custom Airflow image with project deps
├── set_airflow_env.sh            # Environment setup script
├── requirements.txt              # Airflow dependencies
└── README.md                     # Airflow setup instructions
```

#### DAGs (Directed Acyclic Graphs)

**1. Deploy on Start** (`deploy_on_start`)
- **Schedule**: `@once` (runs once when Airflow starts)
- **Purpose**: Initial model deployment
- **Task**: Executes `DEPLOY/build_docker_image.py`
- **Output**: Exports best model from MLflow and optionally builds Docker image

**2. Monitor and Retrain** (`monitor_and_retrain`)
- **Schedule**: Daily at 02:00 UTC (`0 2 * * *`)
- **Purpose**: Automated drift detection and retraining
- **Task**: Executes `MONITOR/monitor_and_retrain.py`
- **Features**:
  - Checks for data drift
  - Automatically retrains if drift detected
  - Rebuilds Docker images after retraining
  - Prevents concurrent runs (`max_active_runs=1`)

#### Docker Compose Architecture

**Services**:
- **postgres**: PostgreSQL database for Airflow metadata
- **airflow-apiserver**: REST API server (port 8080)
- **airflow-scheduler**: Task scheduler and executor
- **airflow-dag-processor**: DAG file processor
- **airflow-triggerer**: Event-triggered task handler
- **flower** (optional): Celery monitoring UI (port 5555)

**Executor**: LocalExecutor (suitable for single-machine deployment)

#### Setup & Usage

**1. Generate Environment File**:
```bash
cd code/airflow
bash set_airflow_env.sh
```

This creates `.env` with:
- Admin credentials (username/password)
- MLflow tracking URI
- Project root path
- Optional secrets (AWS, Docker, HuggingFace, GitHub)

**2. Start Airflow**:
```bash
cd code
AIRFLOW_PROJ_DIR=$(pwd) docker-compose --env-file airflow/.env -f airflow/docker-compose.yaml up -d
```

**3. Access Airflow UI**:
- URL: http://localhost:8080
- Default credentials: `airflow` / `airflow`

**4. Stop Airflow**:
```bash
docker-compose -f airflow/docker-compose.yaml down
```

#### Volume Mounts

The Docker Compose setup mounts:
- `./dags` → `/opt/airflow/dags` (DAG definitions)
- `./logs` → `/opt/airflow/logs` (Airflow logs)
- `./config` → `/opt/airflow/config` (Configuration files)
- `./` → `/opt/airflow/project` (Entire project for script access)

#### DAG Configuration

**Deploy DAG** (`deploy_monitor_dag.py` lines 41-61):
```python
with DAG(
    dag_id="deploy_on_start",
    schedule="@once",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["deploy"],
) as deploy_dag:
    # Runs DEPLOY/build_docker_image.py
    deploy_task = PythonOperator(...)
```

**Monitor DAG** (`deploy_monitor_dag.py` lines 64-94):
```python
with DAG(
    dag_id="monitor_and_retrain",
    schedule="0 2 * * *",  # Daily at 02:00 UTC
    start_date=datetime(2025, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["monitor"],
) as monitor_dag:
    # Runs MONITOR/monitor_and_retrain.py
    monitor_task = PythonOperator(...)
```

#### Environment Variables

Set in `.env` file (created by `set_airflow_env.sh`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `_AIRFLOW_WWW_USER_USERNAME` | `airflow` | Web UI admin username |
| `_AIRFLOW_WWW_USER_PASSWORD` | `airflow` | Web UI admin password |
| `AIRFLOW_PROJ_DIR` | `$(pwd)` | Project root for mounting |
| `MLFLOW_TRACKING_URI` | `file:/opt/airflow/project/RAY/mlruns` | MLflow backend |
| `MONITOR_ENDPOINT` | `http://localhost:5001/invocations` | Model serving endpoint |
| `PROJECT_ROOT` | `/opt/airflow/project` | Mounted project location |

#### Integration with Pipeline

**With RAY**: 
- Monitor DAG triggers `ray_tune_xgboost.py` when drift detected
- Inherits MLflow tracking URI from environment

**With DEPLOY**: 
- Deploy DAG runs `build_docker_image.py` on startup
- Monitor DAG rebuilds images after retraining
- Uses same MLflow tracking store

**With MONITOR**: 
- Executes `monitor_and_retrain.py` on daily schedule
- Passes baseline/current data paths
- Configures drift thresholds via DAG parameters

#### Customization

Edit `dags/deploy_monitor_dag.py` to:
- Change schedules (e.g., hourly, weekly)
- Add email alerts on failure
- Add Slack notifications
- Chain multiple tasks
- Add custom operators

**Example**: Email on failure
```python
DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "retries": 1,
}
```

#### Monitoring Airflow

**Check DAG Status**:
```bash
# View running containers
docker-compose -f airflow/docker-compose.yaml ps

# Check scheduler logs
docker-compose -f airflow/docker-compose.yaml logs airflow-scheduler

# Check webserver logs
docker-compose -f airflow/docker-compose.yaml logs airflow-apiserver
```

**CLI Commands** (inside container):
```bash
# List DAGs
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler airflow dags list

# Trigger DAG manually
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler \
  airflow dags trigger monitor_and_retrain

# Check DAG run status
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler \
  airflow dags list-runs -d monitor_and_retrain
```

#### Advantages of Airflow Orchestration

✅ **Automated Scheduling**: No manual cron jobs  
✅ **Dependency Management**: Tasks run in correct order  
✅ **Failure Handling**: Automatic retries and alerts  
✅ **Logging**: Centralized logs for all tasks  
✅ **Monitoring**: Web UI for pipeline visualization  
✅ **Scalability**: Can distribute tasks across workers  
✅ **Version Control**: DAGs are Python code in git  

#### Resource Requirements

**Minimum**:
- 4GB RAM
- 2 CPUs
- 10GB disk space

**Recommended**:
- 8GB RAM
- 4+ CPUs
- 50GB disk space (for logs and MLflow artifacts)

#### Troubleshooting

**DAGs not appearing**:
```bash
# Check DAG files are mounted
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler ls -la /opt/airflow/dags

# Check for Python errors
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler \
  airflow dags list-import-errors
```

**Project scripts not found**:
```bash
# Verify project is mounted
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler ls -la /opt/airflow/project

# Check PROJECT_ROOT environment variable
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler env | grep PROJECT_ROOT
```

**Database connection errors**:
```bash
# Restart postgres
docker-compose -f airflow/docker-compose.yaml restart postgres

# Check postgres health
docker-compose -f airflow/docker-compose.yaml ps postgres
```

**Tools Used**: apache-airflow 3.0.1, postgresql, docker-compose

**Documentation**: See [airflow/README.md](airflow/README.md) for detailed setup instructions.

---

## Pipeline Overview

The complete ML pipeline follows these stages:

```
1. EDA (Exploration)
   └─> Understand data, identify patterns, baseline model
   
2. RAY (Optimization)
   └─> Find best hyperparameters with Ray Tune + MLflow
       └─> Output: Best model + configuration
   
3. DEPLOY (Production)
   └─> Package model in Docker container
       └─> Serve via REST API for predictions
       
4. MONITOR (Observability)
   └─> Track predictions, detect drift, monitor performance
       └─> Alert on drift → Trigger retraining
       
5. AIRFLOW (Orchestration)
   └─> Automate deployment and monitoring workflows
       └─> Schedule: Deploy @once, Monitor daily
```

**Data Flow**:
```
diabetic_data.csv 
    → EDA analysis 
    → Ray Tune HPO (best hyperparameters)
    → MLflow tracking (best model saved)
    → Docker export (model packaged)                    ┌─────────────────┐
    → REST API (predictions served)                     │                 │
    → Prediction logging (monitored)                    │    AIRFLOW      │
    → Drift detection (statistical tests)               │  Orchestration  │
    → [If drift detected] → Retrain model (back to RAY) │   Scheduler     │
                                                        │                 │
                                                        └─────────────────┘
                                                          @once | Daily
```

**Continuous Feedback Loop** (Orchestrated by Airflow):
```
                    ┌─────────────────────────────────┐
                    │   AIRFLOW Orchestration         │
                    │   - Deploy DAG (@once)          │
                    │   - Monitor DAG (daily 02:00)   │
                    └───────────┬─────────────────────┘
                                ↓
Production Data → MONITOR → Drift Detection
                               ↓ (if drift)
                          Retrain Signal
                               ↓
                  RAY (HPO) → DEPLOY → Production
                               ↑
                               │ (rebuild image)
                     AIRFLOW automates this loop
```

## Future Work

The project is designed to be extended with:
- **Real-time Dashboard**: Live monitoring dashboard for model performance and drift metrics (Streamlit/Grafana)
- **HuggingFace Deployment**: Alternative deployment platform for wider accessibility
- **Web Interface**: LLM-powered chatbot for dataset queries and explanations
- **A/B Testing Framework**: Advanced comparison of model versions in production
- **Feature Store**: Centralized feature engineering and storage
- **Model Performance Monitoring**: Track actual accuracy/AUC in production (currently only drift detection)
- **Concept Drift Detection**: Separate feature drift from label shift analysis
- **SHAP-based Drift**: Explain which features contribute most to drift

## Requirements

### Core Dependencies

The project requires Python 3.11+ and the following libraries:

#### EDA Component
- pandas >= 1.3.0
- numpy >= 1.21.0
- seaborn >= 0.11.0
- matplotlib >= 3.3.0
- scikit-learn >= 1.0.0

#### RAY Component
- ray[tune] >= 2.0.0
- xgboost >= 1.7.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- mlflow >= 1.20.0
- optuna >= 3.0.0 (optional, for advanced search)

#### DEPLOY Component
- docker (system requirement)
- mlflow >= 1.20.0
- xgboost >= 3.0.0
- scikit-learn >= 1.7.0
- pandas >= 2.3.0
- numpy >= 2.3.0

#### MONITOR Component
- scipy >= 1.16.0 (statistical tests: KS, PSI)
- pandas >= 1.3.0
- numpy >= 1.21.0
- requests >= 2.25.0 (API communication)

### Installation

Install all dependencies:

```bash
# Create virtual environment
python3.11 -m venv patient_env
source patient_env/bin/activate  # On Windows: patient_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

Or install by component:

```bash
# For EDA only
pip install pandas numpy seaborn matplotlib scikit-learn jupyter

# For RAY HPO
pip install ray[tune] xgboost scikit-learn mlflow optuna

# For DEPLOY
pip install mlflow xgboost scikit-learn docker

# For MONITOR
pip install scipy pandas numpy requests
```

## Running with Docker

For a containerized environment with all dependencies pre-installed, you can use the provided Docker image. This is especially useful for reproducible environments across different systems.

### Docker Setup

#### 1. Login and Pull the Image

```bash
# Login to Docker Hub (if required)
docker login

# Pull the pre-built environment image
docker pull abuchin/patient-env:1
```

### Running Components with Docker

#### Option 1: Code Only (Self-Contained)

Mount only your code directory and reference data within the mounted volume:

```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  -v "$(realpath ../data)":/data:ro \
  -e MLFLOW_TRACKING_URI=file:/work/RAY/mlruns \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py --data /data/diabetic_data.csv
```

**Usage Pattern**:
- `-v "$PWD":/work` - Mounts current directory to `/work` in container
- `-w /work` - Sets working directory inside container
- `--data /work/...` - References data inside the mounted volume

#### Option 2: Code + External Data Directory (Recommended)

Mount your code and data separately (data as read-only):

```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  -v /home/ec2-user/projects/patient_selection/data:/data:ro \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py --data /data/diabetic_data.csv
```

**Usage Pattern**:
- `-v "$PWD":/work` - Mounts code directory
- `-v /path/to/data:/data:ro` - Mounts data directory read-only (`:ro`)
- `--data /data/...` - References data in the mounted data directory

### Docker Usage Examples

#### Running RAY Hyperparameter Optimization

```bash
# From the code directory
cd /path/to/patient_selection/patient_readmission_kaggle

docker run --rm -it \
  -v "$PWD":/work -w /work \
  -v /path/to/data:/data:ro \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py \
    --data /data/diabetic_data.csv \
    --num-samples 30 \
    --cpus-per-trial 2
```

#### Running with Custom Parameters

```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  -v /home/ec2-user/projects/patient_selection/data:/data:ro \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py \
    --data /data/diabetic_data.csv \
    --num-samples 50 \
    --cpus-per-trial 4 \
    --test-size 0.2 \
    --ray-dir ray_exp \
    --seed 42
```

#### Running Jupyter Notebook in Docker

```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  -v /home/ec2-user/projects/patient_selection/data:/data:ro \
  -p 8888:8888 \
  abuchin/patient-env:1 \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then access the notebook at `http://localhost:8888` (check terminal for token).

### Docker Command Breakdown

| Flag | Purpose |
|------|---------|
| `--rm` | Automatically remove container when it exits |
| `-it` | Interactive terminal (combine `-i` and `-t`) |
| `-v "$PWD":/work` | Mount current directory to `/work` |
| `-w /work` | Set working directory inside container |
| `-v /path:/data:ro` | Mount data directory as read-only |
| `-p 8888:8888` | Port forwarding (host:container) |
| `abuchin/patient-env:1` | Docker image name and tag |

### Advantages of Docker Approach

✅ **Reproducibility**: Identical environment across all systems  
✅ **Isolation**: No conflicts with system packages  
✅ **Portability**: Works on Windows, Linux, macOS  
✅ **Easy Setup**: No manual dependency installation  
✅ **Version Control**: Tagged images ensure consistency  

### Docker Best Practices

1. **Use Read-Only Mounts for Data**: 
   ```bash
   -v /path/to/data:/data:ro
   ```
   Prevents accidental data modification.

2. **Mount Code as Editable**:
   ```bash
   -v "$PWD":/work
   ```
   See code changes immediately without rebuilding.

3. **Preserve Output Directories**:
   Ensure MLflow runs, Ray results, and models are saved inside mounted volumes:
   ```bash
   # Results will persist in your local directory
   python RAY/ray_tune_xgboost.py --ray-dir /work/ray_exp --mlruns-dir /work/mlruns
   ```

4. **Resource Limits** (Optional):
   ```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  -v "$PWD/output:/home/appuser/ray_results" \
  -v "$(realpath ../data)":/data:ro \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py --data /data/diabetic_data.csv
   ```

### Troubleshooting Docker

**Issue**: Permission denied errors  
**Solution**: Run with user mapping:
```bash
docker run --rm -it --user $(id -u):$(id -g) \
  -v "$PWD":/work -w /work \
  abuchin/patient-env:1 \
  python RAY/ray_tune_xgboost.py --data /work/data/file.csv
```

**Issue**: Cannot access files created by Docker  
**Solution**: Fix permissions after running:
```bash
sudo chown -R $USER:$USER ray_exp mlruns
```

**Issue**: Out of disk space  
**Solution**: Clean up Docker resources:
```bash
docker system prune -a
```

---

## Getting Started

### Complete Workflow

Follow these steps to run the entire pipeline from data exploration to deployment:

#### Step 1: Exploratory Data Analysis

```bash
# Navigate to EDA directory
cd code/EDA/

# Launch Jupyter notebook
jupyter notebook EDA.ipynb

# Run all cells to:
# - Explore the dataset
# - Visualize feature distributions
# - Train baseline Random Forest model
# - Analyze feature importance
```

#### Step 2: Hyperparameter Optimization

```bash
# Navigate to RAY directory
cd ../RAY/

# Run Ray Tune HPO (adjust parameters as needed)
python ray_tune_xgboost.py \
    --data /home/ec2-user/projects/patient_selection/data/diabetic_data.csv \
    --num-samples 50 \
    --cpus-per-trial 4 \
    --test-size 0.2

# Monitor progress with MLflow UI (in another terminal)
mlflow ui --backend-store-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
    --host 127.0.0.1 --port 5000

# Access MLflow UI at http://localhost:5000
```

#### Step 3: Model Deployment

```bash
# Navigate to DEPLOY directory
cd ../DEPLOY/

# Build Docker image with best model
python build_docker_image.py \
    --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
    --experiment xgb_diabetic_readmission_hpo \
    --image-tag diabetic-xgb:serve \
    --serve-port 5001

# Run the container
docker run --rm -p 5001:5001 diabetic-xgb:serve

# Test predictions (in another terminal)
curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_records": [
         {
           "age": 55,
           "time_in_hospital": 3,
           "num_lab_procedures": 45,
           "num_medications": 15
         }
       ]
     }'
```

#### Step 4: Model Monitoring

```bash
# Navigate to MONITOR directory
cd ../MONITOR/

# Run drift detection on production logs
python scripts/run_monitor.py

# View generated reports
# - reports/evidently_report_*.html (comprehensive analysis)
# - reports/evidently_summary_*.html (quick overview)
# - reports/drift_alert_*.json (alert status)

# Check MLflow for monitoring history
mlflow ui --backend-store-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
    --host 127.0.0.1 --port 5000

# Access monitoring experiment: "MediWatch-Monitoring"
```

### Quick Start (Individual Components)

#### Run Only EDA
```bash
cd code/EDA/
jupyter notebook EDA.ipynb
```

#### Run Only HPO
```bash
cd code/RAY/
python ray_tune_xgboost.py --data /path/to/diabetic_data.csv --num-samples 30
```

#### Deploy Existing Model
```bash
cd code/DEPLOY/
python build_docker_image.py
docker run --rm -p 5001:5001 diabetic-xgb:serve
```

#### Run Monitoring
```bash
cd code/MONITOR/
python scripts/run_monitor.py
```

## Expected Results & Performance

### EDA Phase
- **Dataset Size**: ~100,000 patient records with 50+ features
- **Target Distribution**: Imbalanced (more non-readmitted than readmitted patients)
- **Baseline Model**: Random Forest with ~65-70% accuracy
- **Key Features**: time_in_hospital, num_medications, num_lab_procedures, age

### RAY HPO Phase
- **Trials**: 30-50 hyperparameter configurations tested
- **Duration**: ~30-60 minutes (depends on CPUs and num_samples)
- **Best Model Performance** (typical):
  - ROC-AUC: 0.70-0.75
  - Average Precision: 0.30-0.40
  - Accuracy: 0.65-0.70
  - F1-Score: 0.45-0.55
- **Improvement**: 5-10% improvement over baseline Random Forest

### DEPLOY Phase
- **Docker Image Size**: ~500MB-1GB
- **Build Time**: 2-5 minutes
- **Startup Time**: 5-10 seconds
- **Inference Speed**: <100ms per prediction
- **API Response Format**: JSON with probability scores

### MONITOR Phase
- **Drift Detection**: Automatic feature and dataset drift analysis
- **Report Generation**: ~5-10 seconds per run
- **Alert Accuracy**: Configurable thresholds (30% feature drift, p<0.05)
- **Storage**: Daily prediction logs in Parquet format
- **MLflow Tracking**: All reports logged as artifacts

## Monitoring & Validation

### MLflow Tracking
Access the MLflow UI to compare experiments:

```bash
# On EC2 instance
mlflow ui --backend-store-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
    --host 127.0.0.1 --port 5000

# From local machine (SSH tunnel)
ssh -i /path/to/key.pem -N -L 5001:127.0.0.1:5000 ec2-user@<EC2-DNS>
```

View:
- All hyperparameter trials and their metrics
- Best model configuration and performance
- Model artifacts and preprocessing pipelines
- Cross-validation results across folds

### Model Validation
Before deploying to production, validate the model:

```python
import mlflow
import pandas as pd

# Load best model
model = mlflow.sklearn.load_model("runs:/<run-id>/model")

# Load test data
test_data = pd.read_csv("test_data.csv")

# Make predictions
predictions = model.predict_proba(test_data)[:, 1]

# Evaluate
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_data['readmitted'], predictions)
print(f"Test AUC: {auc:.4f}")
```

## Data Location

The main dataset should be located at:
```
/home/ec2-user/projects/patient_selection/data/diabetic_data.csv
```

## Troubleshooting

### Common Issues

#### EDA Phase
- **Kernel crashes**: Reduce dataset size or increase memory
- **Missing packages**: Install jupyter and visualization libraries

#### RAY Phase
- **Ray initialization fails**: Check available system resources
- **Out of memory**: Reduce `num_samples` or `cpus_per_trial`
- **MLflow URI errors**: Ensure absolute paths are used
- **Slow trials**: Increase `cpus_per_trial` for faster computation

#### DEPLOY Phase
- **Docker build fails**: Ensure Docker daemon is running
- **Port conflicts**: Use different port with `--serve-port`
- **Model not found**: Verify MLflow tracking URI and experiment name
- **Import errors in container**: Rebuild image, check requirements.txt

#### MONITOR Phase
- **No prediction logs**: Ensure model is serving and predictions are being made
- **Evidently import errors**: Install evidently with `pip install evidently`
- **Reference data not found**: Check path in `configs/monitoring.yaml`
- **Drift false positives**: Adjust thresholds in configuration file
- **Report generation fails**: Ensure sufficient data in logs (minimum ~100 samples)

### Getting Help

1. Check component-specific README files:
   - [EDA/README.md](EDA/README.md)
   - [RAY/README.md](RAY/README.md)
   - [DEPLOY/README.md](DEPLOY/README.md)

2. Verify installation:
   ```bash
   python -c "import ray; import mlflow; import xgboost; print('All packages installed')"
   ```

3. Check logs:
   ```bash
   # Ray logs
   cat ~/ray_results/*/progress.csv
   
   # MLflow logs
   mlflow runs list --experiment-name xgb_diabetic_readmission_hpo
   
   # Docker logs
   docker logs <container-id>
   
   # Monitoring logs
   ls -la code/MONITOR/reports/
   cat code/MONITOR/reports/drift_alert_*.json
   ```


## Project Milestones

- [x] **Phase 1**: Exploratory Data Analysis
  - [x] Data loading and inspection
  - [x] Feature distribution analysis
  - [x] Baseline Random Forest and XGboost models
  - [x] Feature importance analysis

- [x] **Phase 2**: Hyperparameter Optimization
  - [x] Ray Tune integration
  - [x] XGBoost model implementation
  - [x] MLflow experiment tracking
  - [x] Cross-validation evaluation
  - [x] Best model selection

- [x] **Phase 3**: Model Deployment
  - [x] Docker containerization
  - [x] MLflow model serving
  - [x] REST API endpoints
  - [x] Deployment automation script

- [x] **Phase 4**: Model Monitoring & Automated Retraining
  - [x] Automated drift detection (PSI, KS test)
  - [x] Feature-level and prediction-level drift analysis
  - [x] Multi-gate drift logic (critical features, extreme drift)
  - [x] Automated retraining pipeline triggered by drift
  - [x] Docker image rebuild after retraining
  - [x] JSON drift reports for tracking
  - [x] Integration with Ray Tune HPO

- [x] **Phase 5**: Pipeline Orchestration
  - [x] Airflow setup with Docker Compose
  - [x] Deploy DAG (runs once on startup)
  - [x] Monitor DAG (daily scheduled drift detection)
  - [x] Automated task dependency management
  - [x] Web UI for pipeline monitoring
  - [x] Environment configuration management

- [ ] **Phase 6**: Production Enhancement (Future)
  - [ ] Real-time monitoring dashboard (Streamlit/Grafana)
  - [ ] Model performance tracking (accuracy, AUC over time)
  - [ ] Advanced A/B testing framework
  - [ ] Distributed Airflow with CeleryExecutor

- [ ] **Phase 7**: User Interface (Future)
  - [ ] Web application
  - [ ] LLM-powered chatbot
  - [ ] Interactive predictions
  - [ ] Dataset query interface

## Contributing

When contributing to this project:

1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Update README files when adding features
3. **Testing**: Validate changes with sample data
4. **MLflow**: Log all experiments with descriptive names
5. **Docker**: Test containers locally before pushing

## License

This project is part of a patient readmission prediction system designed for healthcare applications.

## References

- **Dataset**: [Diabetes 130-US hospitals for years 1999-2008](https://www.kaggle.com/datasets/brandao/diabetes)
- **Ray Tune**: [Documentation](https://docs.ray.io/en/latest/tune/index.html)
- **MLflow**: [Documentation](https://mlflow.org/docs/latest/index.html)
- **XGBoost**: [Documentation](https://xgboost.readthedocs.io/)
- **Docker**: [Documentation](https://docs.docker.com/)
- **Evidently AI**: [Documentation](https://docs.evidentlyai.com/)

## Contact & Support

For questions or issues related to:
- **Data exploration**: Review EDA notebook and README
- **Model training**: Check RAY component documentation
- **Deployment**: Refer to DEPLOY README
- **Monitoring**: Review drift reports and monitoring configuration
- **General questions**: Open an issue or contact project maintainers
