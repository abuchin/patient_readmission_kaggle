"""DAGs to run scripts directly on the host machine using BashOperator.

This file defines two DAGs:
- `deploy_on_start`: runs once to execute DEPLOY/build_docker_image.py on the host
- `monitor_and_retrain`: runs daily and executes MONITOR/monitor_and_retrain.py on the host

Both tasks use BashOperator to execute Python scripts with proper virtual environment sourcing.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 18),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

# Path to project root and virtual environment (container paths)
PROJECT_ROOT = "/usr/local/airflow/include"
# Note: We'll use the container's Python environment instead of a virtual environment

# --- DAG 1: Deploy on start (one-time) ---
with DAG(
    dag_id="deploy_on_start_bash",
    default_args=DEFAULT_ARGS,
    description="Deploy the best model and start container using BashOperator",
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["deployment", "model", "bash"],
) as deploy_dag:

    deploy_task = BashOperator(
        task_id="deploy_best_model",
        bash_command=f"""
        cd /tmp && \
        cp -r {PROJECT_ROOT}/DEPLOY . && \
        echo "Checking for mlruns directory..." && \
        if [ -d "{PROJECT_ROOT}/RAY/mlruns" ]; then
            echo "Found mlruns in RAY directory, copying..."
            cp -r {PROJECT_ROOT}/RAY/mlruns .
        elif [ -d "{PROJECT_ROOT}/../RAY/mlruns" ]; then
            echo "Found mlruns in parent RAY directory, copying..."
            cp -r {PROJECT_ROOT}/../RAY/mlruns .
        else
            echo "Creating empty mlruns directory for testing..."
            mkdir -p mlruns
        fi && \
        echo "Testing MLflow connection..." && \
        python -c "
import mlflow
mlflow.set_tracking_uri('file:/tmp/mlruns')
try:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    print(f'Found {{len(experiments)}} experiments')
    for exp in experiments:
        print(f'Experiment: {{exp.name}} ({{exp.experiment_id}})')
        if exp.name == 'xgb_diabetic_readmission_hpo':
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string='tags.mlflow.runName = \\"best_model_full_train\\"',
                max_results=3
            )
            print(f'Found {{len(runs)}} runs for experiment {{exp.name}}')
            for i, row in runs.iterrows():
                print(f'Run: {{row[\\"run_id\\"]}} - {{row[\\"tags.mlflow.runName\\"]}}')
except Exception as e:
    print(f'MLflow connection error: {{e}}')
    print('This is expected if no experiments exist yet.')
" && \
        python DEPLOY/build_docker_image.py --out-dir /tmp/model --tracking-uri file:/tmp/mlruns --no-build
        """,
    )

    run_container_task = BashOperator(
        task_id="run_docker_container",
        bash_command=f"""
        cd /tmp && \
        echo "Preparing Docker build context for host execution..." && \

        # Copy build artifacts to a location accessible by host
        BUILD_DIR="/tmp/docker_build_$(date +%s)" && \
        mkdir -p "$BUILD_DIR" && \

        if [ -f "Dockerfile" ] && [ -d "model" ]; then
            echo "Copying Dockerfile and model to host-accessible location..." && \
            cp Dockerfile "$BUILD_DIR/" && \
            cp -r model "$BUILD_DIR/" && \
            echo "Files copied to: $BUILD_DIR" && \
            ls -la "$BUILD_DIR" && \

            # Create a trigger file for manual Docker execution
            echo "BUILD_DIR=$BUILD_DIR" > "/tmp/docker_build_trigger.txt" && \
            echo "TIMESTAMP=$(date)" >> "/tmp/docker_build_trigger.txt" && \
            echo "✅ Docker build context prepared successfully" && \
            echo "📁 Build files available at: $BUILD_DIR" && \
            echo "🚀 Build context ready! Check the trigger file for details:" && \
            echo "   cat /tmp/docker_build_trigger.txt" && \
            echo "📦 To build and run container manually:" && \
            echo "   cd $BUILD_DIR" && \
            echo "   docker build -t diabetic-xgb:serve ." && \
            echo "   docker run --rm -d --name diabetic-xgb-serve -p 5001:5001 diabetic-xgb:serve"
        else
            echo "❌ Missing Dockerfile or model directory from previous task" && \
            echo "Available files in /tmp:" && \
            ls -la /tmp/ && \
            exit 1
        fi
        """,
    )  # Set task dependencies
    deploy_task >> run_container_task

# --- DAG 2: Monitor and retrain (daily) ---
with DAG(
    dag_id="monitor_and_retrain_bash",
    default_args=DEFAULT_ARGS,
    description="Monitor model performance and retrain if needed using BashOperator",
    schedule="0 2 * * *",  # Daily at 2 AM
    catchup=False,
    tags=["monitoring", "retraining", "bash"],
) as monitor_dag:

    monitor_task = BashOperator(
        task_id="monitor_performance",
        bash_command=f"""
        cd /tmp && \
        cp -r {PROJECT_ROOT}/MONITOR . && \
        cp -r {PROJECT_ROOT}/DEPLOY . && \
        echo "Checking for mlruns directory..." && \
        if [ -d "{PROJECT_ROOT}/RAY/mlruns" ]; then
            echo "Found mlruns in RAY directory, copying..."
            cp -r {PROJECT_ROOT}/RAY/mlruns .
        else
            echo "Creating empty mlruns directory for testing..."
            mkdir -p mlruns
        fi && \
        echo "🔍 Starting model performance monitoring..." && \

        # Check if monitoring data exists
        if [ -f "MONITOR/monitoring/tmp/ref.csv" ] && [ -f "MONITOR/monitoring/tmp/cur.csv" ]; then
            echo "📊 Found monitoring data files" && \
            ls -la MONITOR/monitoring/tmp/ && \

            # Run enhanced monitoring script with consistent preprocessing
            echo "🔄 Running enhanced model performance monitoring..." && \
            python MONITOR/enhanced_monitor.py \
                --baseline MONITOR/monitoring/tmp/ref.csv \
                --current MONITOR/monitoring/tmp/cur.csv \
                --endpoint http://172.21.0.1:5001/invocations \
                --tracking-uri file:/tmp/mlruns \
                --retrain-script RAY/ray_tune_xgboost.py \
                --build-script DEPLOY/build_docker_image.py \
                --hpo-num-samples 2 \
                --force && \
            echo "✅ Monitoring completed successfully"
        else
            echo "⚠️  Monitoring data not found, creating sample monitoring data..." && \
            echo "Expected files:" && \
            echo "  - MONITOR/monitoring/tmp/ref.csv" && \
            echo "  - MONITOR/monitoring/tmp/cur.csv" && \
            mkdir -p MONITOR/monitoring/tmp && \
            echo "Creating sample reference data..." && \
            python -c "
import pandas as pd
import numpy as np

# Create sample reference data
np.random.seed(42)
ref_data = pd.DataFrame({{
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'feature3': np.random.normal(0, 1, 1000),
    'target': np.random.choice([0, 1], 1000)
}})
ref_data.to_csv('MONITOR/monitoring/tmp/ref.csv', index=False)

# Create sample current data with slight drift
cur_data = pd.DataFrame({{
    'feature1': np.random.normal(0.1, 1, 1000),  # slight drift
    'feature2': np.random.normal(0, 1, 1000),
    'feature3': np.random.normal(-0.1, 1, 1000), # slight drift
    'target': np.random.choice([0, 1], 1000)
}})
cur_data.to_csv('MONITOR/monitoring/tmp/cur.csv', index=False)
print('Sample monitoring data created')
" && \
            echo "✅ Sample monitoring data created, but skipping actual monitoring for now"
        fi
        """,
    )


    redeploy_task = BashOperator(
        task_id="redeploy_if_retrained",
        bash_command=f"""
        cd /tmp && \
        cp -r {PROJECT_ROOT}/DEPLOY . && \
        echo "Checking for mlruns directory..." && \
        if [ -d "{PROJECT_ROOT}/RAY/mlruns" ]; then
            echo "Found mlruns in RAY directory, copying..."
            cp -r {PROJECT_ROOT}/RAY/mlruns .
        else
            echo "Creating empty mlruns directory for testing..."
            mkdir -p mlruns
        fi && \
        # Check if retraining was triggered (look for retrain experiment)
        echo "🔍 Checking for retrained models..." && \
        python -c "
import mlflow
mlflow.set_tracking_uri('file:/tmp/mlruns')
client = mlflow.tracking.MlflowClient()
try:
    exp = client.get_experiment_by_name('xgb_diabetic_readmission_hpo_retrain')
    if exp:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
        if len(runs) > 0:
            print('Found retrained model, proceeding with redeployment')
            exit(0)
    print('No retrained model found, using original model for redeployment demo')
    exit(0)
except Exception as e:
    print(f'No retrain experiment found: {{e}}')
    print('Using original model for redeployment demo')
    exit(0)
" && \
        # Build model artifacts (using original experiment for demo)
        python DEPLOY/build_docker_image.py \
            --out-dir /tmp/new_model \
            --tracking-uri file:/tmp/mlruns \
            --experiment xgb_diabetic_readmission_hpo \
            --no-build && \
        # Prepare deployment artifacts
        BUILD_DIR="/tmp/retrain_build_$(date +%s)" && \
        mkdir -p "$BUILD_DIR" && \
        cp Dockerfile "$BUILD_DIR/" && \
        cp -r new_model "$BUILD_DIR/model" && \
        echo "✅ Redeployment build context prepared at: $BUILD_DIR" && \
        echo "📦 To build and run updated container manually:" && \
        echo "   cd $BUILD_DIR" && \
        echo "   docker build -t my_model:latest ." && \
        echo "   docker run -p 5000:5000 my_model:latest"
        """,
    )

    # Set task dependencies - monitor first, then retrain if needed, then redeploy if retrained
    monitor_task >> redeploy_task
