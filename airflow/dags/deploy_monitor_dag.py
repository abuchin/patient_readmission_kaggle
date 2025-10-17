"""DAGs to run scripts in the project-worker Docker container.

This file defines two DAGs:
- `deploy_on_start`: runs once to execute DEPLOY/build_docker_image.py in the project-worker container
- `monitor_and_retrain`: runs daily and executes MONITOR/monitor_and_retrain.py in the project-worker container

Both tasks use DockerOperator to execute commands in the project-worker container defined in docker-compose.yaml.
The project-worker container has the correct environment and paths set up for the patient_selection project.
"""

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
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
    description="Run DEPLOY script in project-worker container",
    schedule="@once",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["deploy"],
) as deploy_dag:

    deploy_task = DockerOperator(
        task_id="deploy_best_model",
        image="project-worker:latest",
        command="python /home/ec2-user/projects/patient_selection/code/DEPLOY/build_docker_image.py --no-build",
        docker_url="unix://var/run/docker.sock",
        network_mode="airflow-network",
        auto_remove='success',
        dag=deploy_dag,
    )


# --- DAG 2: Monitor and retrain (daily) ---
with DAG(
    dag_id="monitor_and_retrain",
    default_args=DEFAULT_ARGS,
    description="Run MONITOR script in project-worker container",
    schedule="0 2 * * *",  # daily at 02:00 UTC
    start_date=datetime(2025, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["monitor"],
) as monitor_dag:

    # Default arguments passed to the monitor script; override in Airflow UI if needed
    baseline = "/home/ec2-user/projects/patient_selection/code/MONITOR/monitoring/ref.csv"
    current = "/home/ec2-user/projects/patient_selection/code/MONITOR/monitoring/cur.csv"
    endpoint = os.environ.get("MONITOR_ENDPOINT", "http://localhost:5001/invocations")
    
    monitor_cmd = (
        f"python /home/ec2-user/projects/patient_selection/code/MONITOR/monitor_and_retrain.py "
        f"--baseline {baseline} --current {current} --endpoint {endpoint} "
        f"--retrain-script /home/ec2-user/projects/patient_selection/code/RAY/ray_tune_xgboost.py "
        f"--build-script /home/ec2-user/projects/patient_selection/code/DEPLOY/build_docker_image.py"
    )

    monitor_task = DockerOperator(
        task_id="monitor_and_retrain",
        image="project-worker:latest",
        command=monitor_cmd,
        docker_url="unix://var/run/docker.sock",
        network_mode="airflow-network",
        auto_remove='never',
        dag=monitor_dag,
    )
