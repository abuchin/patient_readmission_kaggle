# Model Deployment with Docker

This directory contains tools and resources for deploying the best XGBoost model from Ray Tune hyperparameter optimization as a Docker container with MLflow model serving.

## Overview

The deployment workflow:

1. **Exports** the best model from MLflow tracking store
2. **Generates** a production-ready Dockerfile
3. **Builds** a Docker image with all dependencies
4. **Serves** the model via MLflow's REST API endpoint

The deployment uses MLflow's built-in model serving capabilities, providing a standardized REST API for making predictions.

## Files in this Directory

| File | Description |
|------|-------------|
| `build_docker_image.py` | Main script to export model and build Docker image |
| `best_model_show.py` | Utility script to inspect the best model from MLflow |
| `Dockerfile` | Auto-generated Dockerfile (created by build script) |
| `model/` | Exported MLflow model directory (created by build script) |
| `requirements.txt` | System requirements (conda environment) |

## Prerequisites

- Docker installed and running
- MLflow tracking store with trained models (from Ray Tune HPO)
- Python 3.11+ environment
- Access to the MLflow runs directory

### Required Python Packages

```bash
pip install mlflow xgboost scikit-learn pandas numpy
```

## Quick Start

### 1. Build Docker Image

Run the build script to export the model and create a Docker image:

```bash
python build_docker_image.py \
    --tracking-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns \
    --experiment xgb_diabetic_readmission_hpo \
    --image-tag diabetic-xgb:serve \
    --serve-port 5001
```

This will:
- Find the best model from MLflow tracking store
- Export it to `./model` directory
- Generate a `Dockerfile`
- Build a Docker image tagged as `diabetic-xgb:serve`

### 2. Run Docker Container

Start the model serving container:

```bash
docker run --rm -p 5001:5001 diabetic-xgb:serve
```

The model will be available at `http://localhost:5001`

### 3. Make Predictions

Send prediction requests to the API:

```bash
curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_records": [
         {
           "feature1": 1.2,
           "feature2": "value",
           "age": 55,
           "time_in_hospital": 3
         }
       ]
     }'
```

If you have your input data as a csv, you could use this way to access the model.

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: text/csv" \
  --data-binary @/path/to/your/data/diabetic_data.csv
```

## Detailed Usage

### Build Script Options

```bash
python build_docker_image.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tracking-uri` | str | `file://.../RAY/mlruns` | MLflow tracking URI from training |
| `--experiment` | str | `xgb_diabetic_readmission_hpo` | MLflow experiment name |
| `--out-dir` | str | `./model` | Directory to export model |
| `--image-tag` | str | `diabetic-xgb:serve` | Docker image tag |
| `--serve-port` | int | `5001` | Container port for serving |
| `--no-build` | flag | False | Only export model, skip Docker build |

### Examples

#### Export Model Only (Skip Docker Build)

```bash
python build_docker_image.py --no-build
```

This creates the `model/` directory and `Dockerfile` without building the image.

#### Custom Image Tag and Port

```bash
python build_docker_image.py \
    --image-tag my-model:v1.0 \
    --serve-port 8080
```

#### Different MLflow Store

```bash
python build_docker_image.py \
    --tracking-uri file:/path/to/different/mlruns \
    --experiment my_experiment_name
```

## Docker Container Details

### Base Image

- **Base**: `python:3.11-slim`
- **Rationale**: Python 3.11+ required for numpy>=2.3 compatibility
- **Size**: Optimized slim image (~200MB base)

### System Dependencies

- `libgomp1`: Required for XGBoost parallel processing
- Minimal system packages for faster builds and smaller images

### Python Dependencies

The container installs:
- MLflow (model serving framework)
- XGBoost (model inference)
- scikit-learn (preprocessing pipeline)
- pandas, numpy, scipy (data handling)

Dependencies are captured automatically by MLflow in `model/requirements.txt`.

### Environment Variables

- `OMP_NUM_THREADS=1`: Prevents over-threading in containers
- `MKL_NUM_THREADS=1`: Optimizes NumPy performance in containers

### Exposed Ports

- Default: `5001` (configurable via `--serve-port`)

## API Specification

### Prediction Endpoint

**URL**: `http://localhost:5001/invocations`

**Method**: `POST`

**Headers**: `Content-Type: application/json`

**Request Body**:

```json
{
  "dataframe_records": [
    {
      "feature1": value1,
      "feature2": value2,
      ...
    }
  ]
}
```

**Response**:

```json
[0.7234, 0.1234, ...]
```

Returns probability scores for the positive class (readmission = YES).

### Health Check

**URL**: `http://localhost:5001/health`

**Method**: `GET`

**Response**: `200 OK` if service is healthy

### Model Metadata

**URL**: `http://localhost:5001/version`

**Method**: `GET`

Returns MLflow model version information.

## Advanced Usage

### Running in Background

```bash
docker run -d --name xgb-model -p 5001:5001 diabetic-xgb:serve
```

View logs:
```bash
docker logs -f xgb-model
```

Stop container:
```bash
docker stop xgb-model
```

### Custom Memory Limits

```bash
docker run --rm -p 5001:5001 --memory="2g" --cpus="2" diabetic-xgb:serve
```

### Mount Custom Configuration

```bash
docker run --rm -p 5001:5001 \
    -v /path/to/config:/app/config \
    diabetic-xgb:serve
```

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  xgb-model:
    image: diabetic-xgb:serve
    ports:
      - "5001:5001"
    environment:
      - OMP_NUM_THREADS=2
      - MKL_NUM_THREADS=2
    mem_limit: 2g
    cpus: 2
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Batch Predictions

For multiple predictions in one request:

```bash
curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_records": [
         {"feature1": 1.2, "feature2": "A", "age": 55},
         {"feature1": 2.3, "feature2": "B", "age": 67},
         {"feature1": 3.4, "feature2": "C", "age": 42}
       ]
     }'
```

Returns array of predictions: `[0.7234, 0.4567, 0.8901]`

## Remote Deployment

### Push to Container Registry

Tag and push to Docker Hub:

```bash
docker tag diabetic-xgb:serve username/diabetic-xgb:serve
docker push username/diabetic-xgb:serve
```

Tag and push to AWS ECR:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag diabetic-xgb:serve <account-id>.dkr.ecr.us-east-1.amazonaws.com/diabetic-xgb:serve
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/diabetic-xgb:serve
```

### Deploy to EC2 Instance

On your EC2 instance:

```bash
# Pull the image
docker pull username/diabetic-xgb:serve

# Run the container
docker run -d --name xgb-model -p 5001:5001 --restart always username/diabetic-xgb:serve
```

### SSH Tunnel for Remote Access

From your local machine:

```bash
ssh -i /path/to/your/key.pem -N -L 5001:127.0.0.1:5001 ec2-user@<EC2-instance-public-DNS>
```

Access the model at `http://localhost:5001` on your local machine.

## Monitoring and Debugging

### View Container Logs

```bash
docker logs -f xgb-model
```

### Execute Commands in Container

```bash
docker exec -it xgb-model /bin/bash
```

### Check Model Files

```bash
docker exec xgb-model ls -la /app/model/
```

### Test Model Locally

Before deploying, test the exported model:

```python
import mlflow

# Load model
model = mlflow.sklearn.load_model("./model")

# Make prediction
import pandas as pd
sample_data = pd.DataFrame([{
    "feature1": 1.2,
    "feature2": "A",
    # ... all required features
}])

prediction = model.predict_proba(sample_data)[:, 1]
print(f"Prediction: {prediction[0]:.4f}")
```

## Performance Optimization

### Multi-Worker Serving

For high-throughput scenarios, use Gunicorn:

Modify the Dockerfile CMD:

```dockerfile
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "mlflow.pyfunc.scoring_server.wsgi:app"]
```

### Enable GPU Support

If using GPU-enabled XGBoost:

1. Use NVIDIA base image: `FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
2. Add GPU support flag: `docker run --gpus all -p 5001:5001 diabetic-xgb:serve`

### Caching for Faster Builds

Use Docker BuildKit for faster builds:

```bash
DOCKER_BUILDKIT=1 docker build -t diabetic-xgb:serve .
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Error**: `Bind for 0.0.0.0:5001 failed: port is already allocated`

**Solution**: Use a different port
```bash
docker run --rm -p 5002:5001 diabetic-xgb:serve
```

#### 2. Model Not Found

**Error**: `No run named 'best_model_full_train' found`

**Solution**: Ensure you've run the Ray Tune HPO first and that it completed successfully.

#### 3. Out of Memory

**Error**: Container crashes with OOM

**Solution**: Increase Docker memory allocation or add memory limits
```bash
docker run --rm -p 5001:5001 --memory="4g" diabetic-xgb:serve
```

#### 4. Import Errors

**Error**: `ModuleNotFoundError: No module named 'xgboost'`

**Solution**: Rebuild the image, ensuring `model/requirements.txt` exists:
```bash
python build_docker_image.py --no-build  # Re-export model
docker build -t diabetic-xgb:serve .     # Rebuild image
```

#### 5. Slow Predictions

**Issue**: Predictions taking too long

**Solutions**:
- Increase `OMP_NUM_THREADS` and `MKL_NUM_THREADS`
- Allocate more CPUs: `docker run --cpus="4" ...`
- Use batch predictions instead of single requests

### Verify Deployment

Test if the service is running:

```bash
# Check if container is running
docker ps

# Test health endpoint
curl http://localhost:5001/health

# Test prediction with minimal data
curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{"dataframe_records": [{}]}'
```

## Security Considerations

### Production Deployment

1. **Don't expose port publicly**: Use reverse proxy (Nginx, Traefik)
2. **Add authentication**: Implement API key validation
3. **Use HTTPS**: Terminate SSL at load balancer or reverse proxy
4. **Resource limits**: Always set memory and CPU limits
5. **Regular updates**: Keep base images and dependencies updated

### Example with Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /predict {
        proxy_pass http://localhost:5001/invocations;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Best Practices

1. **Version your models**: Tag images with model versions
2. **Automate builds**: Use CI/CD pipelines (GitHub Actions, Jenkins)
3. **Monitor performance**: Add logging and metrics collection
4. **Test before deploy**: Validate predictions match expected results
5. **Document API schema**: Provide clear examples of required features
6. **Health checks**: Implement proper health check endpoints
7. **Graceful shutdown**: Handle SIGTERM signals properly

## Next Steps

- Set up CI/CD pipeline for automated deployments
- Add monitoring with Prometheus and Grafana
- Implement A/B testing framework
- Create client libraries for easy integration
- Add rate limiting and request throttling
- Implement model versioning and rollback capabilities

## Related Documentation

- [Ray Tune HPO README](../RAY/README.md) - Training pipeline documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) - Model tracking and serving
- [Docker Documentation](https://docs.docker.com/) - Container deployment

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review MLflow logs: `docker logs xgb-model`
3. Verify model was exported correctly: `ls -la model/`
4. Test model locally before containerizing
