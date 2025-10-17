FROM apache/airflow:3.0.1
USER root
# Add docker group with GID 999 and add airflow user to it
RUN groupadd -g 999 docker && usermod -a -G docker airflow
USER airflow
COPY ../requirements.txt /
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt