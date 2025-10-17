#!/bin/bash
# Generate a .env file for Airflow docker-compose

echo -e "AIRFLOW_UID=$(id -u)" > .env
