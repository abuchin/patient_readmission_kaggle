"""DAGs to run DEPLOY on Airflow startup and MONITOR on schedule.

This file defines two DAGs:
- `deploy_on_start`: runs once (schedule_interval='@once') to export and build the model image using DEPLOY/build_docker_image.py
- `monitor_and_retrain`: runs daily and executes MONITOR/monitor_and_retrain.py with reasonable defaults

Both tasks invoke the project scripts via python3 and inherit the container environment so things like
`MLFLOW_TRACKING_URI` can be provided by the Airflow container environment or an `.env`.
"""

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess

# Base path where the project is mounted inside Airflow containers
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Helper to run a script via subprocess and log output/errors
def run_script(cmd, env=None):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, env=env or os.environ.copy(), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Script failed: {cmd}\n{result.stderr}")



# --- DAG 1: Deploy on start (run once) ---
with DAG(
    dag_id="deploy_on_start",
    default_args=DEFAULT_ARGS,
    description="Run DEPLOY/build_docker_image.py once when Airflow starts",
    schedule="@once",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["deploy"],
) as deploy_dag:

    def deploy_py():
        try:
            run_script(f"python3 {PROJECT_ROOT}/DEPLOY/build_docker_image.py --no-build")
        except Exception:
            run_script(f"python3 {PROJECT_ROOT}/DEPLOY/build_docker_image.py")

    deploy_task = PythonOperator(
        task_id="deploy_best_model",
        python_callable=deploy_py,
        dag=deploy_dag,
    )


# --- DAG 2: Monitor and retrain (daily) ---
with DAG(
    dag_id="monitor_and_retrain",
    default_args=DEFAULT_ARGS,
    description="Run MONITOR/monitor_and_retrain.py on a daily schedule",
    schedule="0 2 * * *",  # daily at 02:00 UTC
    start_date=datetime(2025, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["monitor"],
) as monitor_dag:

    # Default arguments passed to the monitor script; override in Airflow UI if needed
    baseline = f"{PROJECT_ROOT}/MONITOR/monitoring/ref.csv"
    current = f"{PROJECT_ROOT}/MONITOR/monitoring/cur.csv"
    endpoint = os.environ.get("MONITOR_ENDPOINT", "http://localhost:5001/invocations")

    def monitor_py():
        cmd = (
            f"python3 {PROJECT_ROOT}/MONITOR/monitor_and_retrain.py "
            f"--baseline {baseline} --current {current} --endpoint {endpoint} "
            f"--retrain-script {PROJECT_ROOT}/RAY/ray_tune_xgboost.py "
            f"--build-script {PROJECT_ROOT}/DEPLOY/build_docker_image.py"
        )
        run_script(cmd)

    monitor_task = PythonOperator(
        task_id="monitor_and_retrain",
        python_callable=monitor_py,
        dag=monitor_dag,
    )
