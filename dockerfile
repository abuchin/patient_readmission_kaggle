FROM python:3.13-slim

# airflow-worker image
# - Creates ec2-user
# - Optionally copies project folders (build context must contain them)
# - Declares volumes for RAY/DEPLOY/MONITOR so Airflow can mount host dirs
# - Installs requirements.txt if present
# - Exposes /bin/bash entrypoint so Airflow can 'docker run <image> <cmd>' or exec

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash ec2-user \
  && mkdir -p /home/ec2-user/projects/patient_selection/code \
  && chown -R ec2-user:ec2-user /home/ec2-user/projects/patient_selection/code

WORKDIR /home/ec2-user/projects/patient_selection/code

# Copy project folders if they exist in the build context
COPY RAY /home/ec2-user/projects/patient_selection/code/RAY
COPY DEPLOY /home/ec2-user/projects/patient_selection/code/DEPLOY
COPY MONITOR /home/ec2-user/projects/patient_selection/code/MONITOR

VOLUME ["/home/ec2-user/projects/patient_selection/code/RAY", "/home/ec2-user/projects/patient_selection/code/DEPLOY", "/home/ec2-user/projects/patient_selection/code/MONITOR"]

# Install project requirements if provided
COPY requirements.txt /home/ec2-user/projects/patient_selection/code/requirements.txt
RUN set -eux; \
  python -m pip install --upgrade pip setuptools wheel || true; \
  if [ -f /home/ec2-user/projects/patient_selection/code/requirements.txt ]; then \
    pip install --no-cache-dir -r /home/ec2-user/projects/patient_selection/code/requirements.txt; \
  fi

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["while true; do sleep 3600; done"]
