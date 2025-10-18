#!/bin/bash
# Script to run Docker container from the host system
# This is called from Airflow running in Astro

set -e

DOCKER_IMAGE="diabetic-xgb:serve"
CONTAINER_NAME="diabetic-xgb-serve"
BUILD_CONTEXT="/tmp/airflow_build"

echo "Docker container runner script started..."

# Function to check if port is available
check_port() {
    local port=$1
    if ! netstat -tuln | grep ":$port " >/dev/null 2>&1; then
        return 0  # Port is available
    else
        return 1  # Port is in use
    fi
}

# Create build context directory
mkdir -p "$BUILD_CONTEXT"

# Copy model and Dockerfile from Astro temp directory if they exist
if [ -d "/tmp/model" ]; then
    echo "Copying model files..."
    cp -r /tmp/model "$BUILD_CONTEXT/"
fi

if [ -f "/tmp/Dockerfile" ]; then
    echo "Copying Dockerfile..."
    cp /tmp/Dockerfile "$BUILD_CONTEXT/"
fi

# Check if we have the necessary files
if [ ! -f "$BUILD_CONTEXT/Dockerfile" ] || [ ! -d "$BUILD_CONTEXT/model" ]; then
    echo "Error: Missing Dockerfile or model directory"
    echo "Expected files in: $BUILD_CONTEXT"
    ls -la "$BUILD_CONTEXT"
    exit 1
fi

echo "Building Docker image..."
cd "$BUILD_CONTEXT"
docker build -t "$DOCKER_IMAGE" .

echo "Stopping existing container if running..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "Finding available port..."
for port in $(seq 5001 5009); do
    if check_port $port; then
        echo "Starting container on port $port..."
        if docker run --rm -d --name "$CONTAINER_NAME" -p "$port:5001" "$DOCKER_IMAGE"; then
            echo "✅ Container started successfully!"
            echo "📍 Model serving endpoint: http://localhost:$port"
            echo "🐳 Container details:"
            docker ps --filter "name=$CONTAINER_NAME"
            exit 0
        else
            echo "Failed to start container on port $port"
        fi
    else
        echo "Port $port is in use, trying next..."
    fi
done

echo "❌ No available ports found in range 5001-5009"
exit 1