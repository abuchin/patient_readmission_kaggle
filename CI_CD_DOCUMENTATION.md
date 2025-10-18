# CI/CD Documentation

## Overview

This project includes a comprehensive GitHub Actions CI/CD pipeline designed specifically for MLOps workflows. The pipeline ensures code quality, model reproducibility, and automated deployment of machine learning components.

## Pipeline Architecture

### 🔄 Workflow Triggers
- **Push to main/master/develop**: Full pipeline execution
- **Pull Requests**: Validation and testing
- **Scheduled**: Nightly performance tests
- **Manual**: On-demand execution

### 🏗️ Pipeline Stages

#### 1. **Code Quality & Security** (`lint-and-security`)
```yaml
- Code formatting (Black)
- Import sorting (isort) 
- Linting (flake8)
- Security scanning (bandit, safety)
- Type checking (mypy)
```

#### 2. **Unit Tests & Coverage** (`unit-tests`)
```yaml
- Multi-Python version testing (3.11, 3.12)
- Comprehensive test coverage
- MLOps component validation
- Coverage reporting to Codecov
```

#### 3. **Airflow DAG Tests** (`airflow-tests`)
```yaml
- DAG integrity validation
- Syntax checking
- Astro CLI integration
- Mock environment setup
```

#### 4. **MLflow Model Tests** (`mlflow-tests`)
```yaml
- Model training pipeline validation
- MLflow experiment tracking
- Model artifact verification
- Synthetic dataset generation
```

#### 5. **Docker Build Tests** (`docker-tests`)
```yaml
- Multi-component Docker builds
- Image security validation
- Resource optimization checks
```

#### 6. **Integration Tests** (`integration-tests`)
```yaml
- End-to-end pipeline validation
- Component interaction testing
- Data pipeline verification
```

#### 7. **Performance Tests** (`performance-tests`)
```yaml
- Memory profiling
- Throughput benchmarking
- Load testing (scheduled)
- Resource usage monitoring
```

#### 8. **Security Scan** (`security-scan`)
```yaml
- Advanced vulnerability scanning (Trivy)
- SARIF reporting to GitHub Security
- Dependency analysis
```

#### 9. **Deployment** (`deploy`)
```yaml
- Multi-environment deployment
- Container registry integration
- Automated artifact management
```

## 🚀 Getting Started

### Prerequisites

1. **GitHub Repository Setup**
   ```bash
   # Fork or clone the repository
   git clone https://github.com/yourusername/patient-readmission-prediction.git
   cd patient-readmission-prediction
   ```

2. **Local Development Setup**
   ```bash
   # Quick setup with Make
   make quickstart
   
   # Or manual setup
   make setup
   make create-test-data
   ```

3. **GitHub Secrets Configuration**
   Configure the following secrets in your GitHub repository:

   **Optional Registry Secrets** (for deployment):
   - `REGISTRY_URL`: Container registry URL
   - `REGISTRY_USERNAME`: Registry username
   - `REGISTRY_PASSWORD`: Registry password/token

   **Optional Integration Secrets**:
   - `CODECOV_TOKEN`: For coverage reporting

### 🔧 Local Testing

Test the CI/CD pipeline locally before pushing:

```bash
# Run complete CI/CD validation
make ci-test

# Individual components
make format-check      # Code formatting
make lint             # Linting and security
make test             # Unit tests
make airflow-test     # Airflow DAG tests
make docker-build     # Docker builds
```

### 📊 Pipeline Monitoring

#### GitHub Actions Dashboard
- Navigate to `Actions` tab in your GitHub repository
- Monitor pipeline execution in real-time
- Review logs and artifacts
- Track performance metrics

#### Status Badges
Add these badges to your README:

```markdown
[![CI/CD Pipeline](https://github.com/yourusername/patient-readmission-prediction/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/yourusername/patient-readmission-prediction/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/yourusername/patient-readmission-prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/patient-readmission-prediction)
```

## 🛠️ Configuration

### Pipeline Customization

#### Environment Variables
Configure in `.github/workflows/ci-cd.yml`:

```yaml
env:
  PYTHON_VERSION: '3.12'           # Python version
  MLFLOW_TRACKING_URI: file://./mlruns  # MLflow tracking
  ASTRO_RUNTIME_VERSION: '12.1.1'  # Astro Airflow version
```

#### Test Configuration
Modify `setup.cfg` or `pyproject.toml`:

```ini
[tool:pytest]
testpaths = tests
addopts = --cov=. --cov-report=xml --cov-fail-under=70
```

#### Security Configuration
Update `.pre-commit-config.yaml` for local hooks:

```yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: ["-r", ".", "-x", "tests,venv"]
```

### 📈 MLOps-Specific Features

#### Model Versioning
```yaml
# Automatic model versioning based on Git tags
- name: Tag model version
  run: |
    MODEL_VERSION=${GITHUB_REF#refs/tags/}
    echo "MODEL_VERSION=$MODEL_VERSION" >> $GITHUB_ENV
```

#### Experiment Tracking
```yaml
# MLflow experiment management
- name: Track experiment
  run: |
    export MLFLOW_TRACKING_URI=file://./mlruns
    python RAY/ray_tune_xgboost.py --experiment-name "ci-${GITHUB_SHA:0:7}"
```

#### Data Validation
```yaml
# Automated data quality checks
- name: Validate data
  run: |
    python -c "
    import pandas as pd
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    # Data validation logic
    "
```

## 🔐 Security Best Practices

### Secret Management
- **Never** commit secrets to the repository
- Use GitHub Secrets for sensitive data
- Rotate secrets regularly
- Use scoped permissions

### Dependency Security
```yaml
# Automated dependency scanning
- name: Security check
  run: |
    safety check --json --output safety-report.json
    bandit -r . -f json -o bandit-report.json
```

### Container Security
```yaml
# Docker image security scanning
- name: Scan Docker image
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'patient-readmission:latest'
    format: 'sarif'
    output: 'trivy-results.sarif'
```

## 🚨 Troubleshooting

### Common Issues

#### 1. **Test Failures**
```bash
# Local debugging
make test-fast           # Quick test run
pytest tests/ -v --tb=short  # Detailed output
pytest tests/ -k "test_name" # Specific test
```

#### 2. **Docker Build Issues**
```bash
# Local Docker testing
make docker-build        # Build all images
docker system prune -f   # Clean up
make docker-clean        # Project cleanup
```

#### 3. **Airflow DAG Issues**
```bash
# DAG validation
make airflow-test        # Run DAG tests
astro dev bash -c "airflow dags list"  # List DAGs
```

#### 4. **MLflow Issues**
```bash
# MLflow debugging
make mlflow-ui           # Start MLflow UI
export MLFLOW_TRACKING_URI=file://./mlruns
mlflow experiments search
```

### Pipeline Debugging

#### Enable Debug Logging
```yaml
# Add to workflow steps
- name: Debug step
  run: |
    echo "::set-output name=debug::true"
    echo "Debug information here"
  env:
    ACTIONS_STEP_DEBUG: true
```

#### Artifact Collection
```yaml
# Collect debugging artifacts
- name: Upload debug artifacts
  uses: actions/upload-artifact@v4
  if: failure()
  with:
    name: debug-logs
    path: |
      logs/
      *.log
      test-results/
```

## 📋 Maintenance

### Regular Updates

#### 1. **Dependency Updates**
```bash
# Update dependencies
pip-compile requirements.in
pre-commit autoupdate
```

#### 2. **Security Updates**
```bash
# Security audit
make security-scan
safety check --json
```

#### 3. **Performance Monitoring**
```bash
# Performance benchmarks
make performance-tests
# Monitor resource usage
```

### Pipeline Optimization

#### 1. **Caching Strategy**
```yaml
# Optimize build times
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

#### 2. **Parallel Execution**
```yaml
# Matrix builds for multiple environments
strategy:
  matrix:
    python-version: ['3.11', '3.12']
    os: [ubuntu-latest, windows-latest]
```

#### 3. **Selective Testing**
```yaml
# Conditional testing based on changes
- name: Check changed files
  uses: dorny/paths-filter@v2
  with:
    filters: |
      mlops:
        - 'RAY/**'
        - 'DEPLOY/**'
        - 'MONITOR/**'
```

## 🎯 Best Practices

### 1. **Code Quality**
- Maintain test coverage above 70%
- Follow PEP 8 style guidelines
- Use type hints where applicable
- Document complex functions

### 2. **MLOps Practices**
- Version all models and experiments
- Track hyperparameters and metrics
- Validate data quality in pipeline
- Monitor model drift in production

### 3. **CI/CD Practices**
- Keep pipeline execution under 30 minutes
- Use appropriate caching strategies
- Fail fast on critical issues
- Provide clear error messages

### 4. **Security Practices**
- Scan all dependencies regularly
- Use minimal container images
- Implement least-privilege access
- Audit security configurations

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Astro CLI Documentation](https://docs.astronomer.io/astro/cli/overview)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Pre-commit Hooks](https://pre-commit.com/)

---

For questions or issues with the CI/CD pipeline, please open an issue using our [MLOps Pipeline Issue Template](.github/ISSUE_TEMPLATE/mlops_issue.md).