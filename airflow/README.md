# Airflow Setup Instructions

## How to Run Airflow Locally

1. **Set up the environment file**
   - Run the provided script to create a `.env` file with recommended variables:
     ```bash
     bash set_airflow_env.sh
     ```
   - You can edit `.env` to customize paths, credentials, or secrets.

2. **Start Airflow services**
   - From the project root (`code` directory), run:
     ```bash
     AIRFLOW_PROJ_DIR=$(pwd) docker-compose --env-file airflow/.env -f airflow/docker-compose.yaml up -d
     ```
   - This mounts your project and DAGs into the Airflow containers and loads environment variables from `.env`.

3. **Access the Airflow UI**
   - Open your browser and go to: [http://localhost:8080](http://localhost:8080)
   - Default login: `airflow` / `airflow` (change in `.env` if needed)

4. **Stop Airflow services**
   - Run:
     ```bash
     docker-compose -f airflow/docker-compose.yaml down
     ```

## Troubleshooting
- If DAGs do not appear, check that your DAG files are in `airflow/dags/` and readable.
- If scripts are not found, confirm the project root is mounted at `/opt/airflow/project` in the container.
- Check logs with:
  ```bash
  docker-compose -f airflow/docker-compose.yaml logs airflow-scheduler
  docker-compose -f airflow/docker-compose.yaml logs airflow-webserver
  ```

## Customizing Environment Variables
- Edit `airflow/.env` to set secrets, MLflow URIs, or other service credentials.
- For production, use Docker secrets or Airflow Connections for sensitive values.
