# Makefile for Patient Readmission MLOps Pipeline
# Usage: make <target>

.DEFAULT_GOAL := help
.PHONY: help install test lint format clean docker-build docker-run airflow-test mlflow-ui

# Colors for terminal output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

# Variables
PYTHON := python3.12
PIP := pip
VENV := patient_env
PROJECT_NAME := patient-readmission-prediction

help: ## Show this help message
	@echo "${BLUE}Patient Readmission MLOps Pipeline${NC}"
	@echo "${BLUE}====================================${NC}"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${GREEN}%-20s${NC} %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Development Environment Setup
# =============================================================================

install: ## Install all dependencies and set up development environment
	@echo "${BLUE}Setting up development environment...${NC}"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	@echo "${GREEN}✅ Development environment ready!${NC}"

install-pre-commit: ## Install pre-commit hooks
	@echo "${BLUE}Installing pre-commit hooks...${NC}"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "${GREEN}✅ Pre-commit hooks installed!${NC}"

setup: install install-pre-commit ## Complete development setup
	@echo "${GREEN}🚀 Development environment fully configured!${NC}"

# =============================================================================
# Code Quality & Testing
# =============================================================================

test: ## Run all tests
	@echo "${BLUE}Running tests...${NC}"
	pytest tests/ -v --cov=. --cov-report=term-missing
	@echo "${GREEN}✅ Tests completed!${NC}"

test-fast: ## Run tests without coverage
	@echo "${BLUE}Running fast tests...${NC}"
	pytest tests/ -v --tb=short
	@echo "${GREEN}✅ Fast tests completed!${NC}"

test-unit: ## Run only unit tests
	@echo "${BLUE}Running unit tests...${NC}"
	pytest tests/ -v -m "not integration and not slow"
	@echo "${GREEN}✅ Unit tests completed!${NC}"

test-integration: ## Run integration tests
	@echo "${BLUE}Running integration tests...${NC}"
	pytest tests/ -v -m "integration"
	@echo "${GREEN}✅ Integration tests completed!${NC}"

lint: ## Run all linting checks
	@echo "${BLUE}Running linting checks...${NC}"
	flake8 . --count --statistics
	mypy . --ignore-missing-imports
	bandit -r . -x "./patient_env/*,./venv/*" -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true
	@echo "${GREEN}✅ Linting completed!${NC}"

format: ## Format code with black and isort
	@echo "${BLUE}Formatting code...${NC}"
	black .
	isort .
	@echo "${GREEN}✅ Code formatted!${NC}"

format-check: ## Check code formatting without making changes
	@echo "${BLUE}Checking code formatting...${NC}"
	black --check --diff .
	isort --check-only --diff .
	@echo "${GREEN}✅ Format check completed!${NC}"

pre-commit: ## Run pre-commit hooks on all files
	@echo "${BLUE}Running pre-commit hooks...${NC}"
	pre-commit run --all-files
	@echo "${GREEN}✅ Pre-commit hooks completed!${NC}"

# =============================================================================
# MLOps Pipeline
# =============================================================================

mlflow-ui: ## Start MLflow UI
	@echo "${BLUE}Starting MLflow UI...${NC}"
	@echo "MLflow UI will be available at: http://localhost:5000"
	cd astro-airflow && mlflow ui --backend-store-uri file:///tmp/mlruns --host 0.0.0.0 --port 5000

train: ## Run model training with Ray Tune
	@echo "${BLUE}Starting model training...${NC}"
	cd RAY && python ray_tune_xgboost.py --tracking-uri file:///tmp/mlruns --experiment-name xgb_diabetic_readmission_hpo
	@echo "${GREEN}✅ Model training completed!${NC}"

deploy: ## Deploy best model
	@echo "${BLUE}Deploying best model...${NC}"
	cd DEPLOY && python build_docker_image.py --tracking-uri file:///tmp/mlruns --experiment xgb_diabetic_readmission_hpo
	@echo "${GREEN}✅ Model deployment completed!${NC}"

monitor: ## Run monitoring pipeline
	@echo "${BLUE}Running monitoring pipeline...${NC}"
	cd MONITOR && python enhanced_monitor.py --baseline monitoring/tmp/ref.csv --current monitoring/tmp/cur.csv --endpoint http://localhost:5001/invocations
	@echo "${GREEN}✅ Monitoring completed!${NC}"

# =============================================================================
# Airflow Development
# =============================================================================

airflow-init: ## Initialize Astro Airflow environment
	@echo "${BLUE}Initializing Astro Airflow...${NC}"
	cd astro-airflow && astro dev init || true
	@echo "${GREEN}✅ Astro Airflow initialized!${NC}"

airflow-start: ## Start Astro Airflow locally
	@echo "${BLUE}Starting Astro Airflow...${NC}"
	@echo "Airflow UI will be available at: http://localhost:8080"
	@echo "Username: admin, Password: admin"
	cd astro-airflow && astro dev start

airflow-stop: ## Stop Astro Airflow
	@echo "${BLUE}Stopping Astro Airflow...${NC}"
	cd astro-airflow && astro dev stop
	@echo "${GREEN}✅ Astro Airflow stopped!${NC}"

airflow-test: ## Test Airflow DAGs
	@echo "${BLUE}Testing Airflow DAGs...${NC}"
	cd astro-airflow && pytest tests/ -v
	@echo "${GREEN}✅ Airflow DAG tests completed!${NC}"

airflow-run-deploy: ## Run deploy DAG
	@echo "${BLUE}Running deploy DAG...${NC}"
	cd astro-airflow && astro run deploy_on_start_bash --verbose
	@echo "${GREEN}✅ Deploy DAG completed!${NC}"

airflow-run-monitor: ## Run monitoring DAG
	@echo "${BLUE}Running monitoring DAG...${NC}"
	cd astro-airflow && astro run monitor_and_retrain_bash --verbose
	@echo "${GREEN}✅ Monitoring DAG completed!${NC}"

# =============================================================================
# Docker Operations
# =============================================================================

docker-build: ## Build all Docker images
	@echo "${BLUE}Building Docker images...${NC}"
	docker build -t $(PROJECT_NAME):latest .
	cd DEPLOY && docker build -t $(PROJECT_NAME)-deploy:latest .
	cd MONITOR && docker build -t $(PROJECT_NAME)-monitor:latest .
	@echo "${GREEN}✅ Docker images built!${NC}"

docker-build-main: ## Build main Docker image
	@echo "${BLUE}Building main Docker image...${NC}"
	docker build -t $(PROJECT_NAME):latest .
	@echo "${GREEN}✅ Main Docker image built!${NC}"

docker-build-deploy: ## Build deployment Docker image
	@echo "${BLUE}Building deployment Docker image...${NC}"
	cd DEPLOY && docker build -t $(PROJECT_NAME)-deploy:latest .
	@echo "${GREEN}✅ Deployment Docker image built!${NC}"

docker-build-monitor: ## Build monitoring Docker image
	@echo "${BLUE}Building monitoring Docker image...${NC}"
	cd MONITOR && docker build -t $(PROJECT_NAME)-monitor:latest .
	@echo "${GREEN}✅ Monitoring Docker image built!${NC}"

docker-run: ## Run main Docker container
	@echo "${BLUE}Running main Docker container...${NC}"
	docker run -it --rm -p 8000:8000 $(PROJECT_NAME):latest

docker-run-deploy: ## Run deployment service
	@echo "${BLUE}Running deployment service...${NC}"
	docker run -it --rm -p 5001:5001 $(PROJECT_NAME)-deploy:latest

docker-clean: ## Clean up Docker images and containers
	@echo "${BLUE}Cleaning up Docker...${NC}"
	docker system prune -af
	@echo "${GREEN}✅ Docker cleanup completed!${NC}"

# =============================================================================
# Data Operations
# =============================================================================

create-test-data: ## Create test datasets
	@echo "${BLUE}Creating test datasets...${NC}"
	$(PYTHON) -c "
	import pandas as pd
	import numpy as np
	import os
	
	# Create test directories
	os.makedirs('test_data', exist_ok=True)
	os.makedirs('MONITOR/monitoring/tmp', exist_ok=True)
	
	# Create sample diabetes data
	np.random.seed(42)
	n_samples = 1000
	data = {
	    'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic'], n_samples),
	    'gender': np.random.choice(['Male', 'Female'], n_samples),
	    'age': np.random.choice(['[70-80)', '[60-70)', '[50-60)'], n_samples),
	    'time_in_hospital': np.random.randint(1, 15, n_samples),
	    'num_medications': np.random.randint(1, 20, n_samples),
	    'readmitted': np.random.choice(['NO', '<30', '>30'], n_samples)
	}
	df = pd.DataFrame(data)
	df.to_csv('test_data/sample_diabetes_data.csv', index=False)
	
	# Create monitoring data
	ref_data = pd.DataFrame({f'feature_{i}': np.random.randn(5000) for i in range(20)})
	cur_data = pd.DataFrame({f'feature_{i}': np.random.randn(5000) + 0.1 for i in range(20)})
	ref_data.to_csv('MONITOR/monitoring/tmp/ref.csv', index=False)
	cur_data.to_csv('MONITOR/monitoring/tmp/cur.csv', index=False)
	
	print('✅ Test datasets created!')
	"
	@echo "${GREEN}✅ Test data created!${NC}"

# =============================================================================
# Cleanup Operations
# =============================================================================

clean: ## Clean up temporary files and caches
	@echo "${BLUE}Cleaning up...${NC}"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/
	rm -f bandit-report.json safety-report.json
	@echo "${GREEN}✅ Cleanup completed!${NC}"

clean-mlflow: ## Clean MLflow tracking data
	@echo "${BLUE}Cleaning MLflow data...${NC}"
	rm -rf mlruns/
	rm -rf astro-airflow/include/mlruns/
	@echo "${GREEN}✅ MLflow data cleaned!${NC}"

clean-all: clean clean-mlflow docker-clean ## Complete cleanup (files, MLflow, Docker)
	@echo "${GREEN}🧹 Complete cleanup finished!${NC}"

# =============================================================================
# CI/CD Operations
# =============================================================================

ci-test: ## Run CI/CD pipeline tests locally
	@echo "${BLUE}Running CI/CD tests locally...${NC}"
	make format-check
	make lint
	make test
	make airflow-test
	make docker-build
	@echo "${GREEN}✅ CI/CD tests completed!${NC}"

security-scan: ## Run security scans
	@echo "${BLUE}Running security scans...${NC}"
	bandit -r . -x "./patient_env/*,./venv/*" -f custom
	safety check
	@echo "${GREEN}✅ Security scan completed!${NC}"

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	@echo "${BLUE}Generating documentation...${NC}"
	@echo "📚 Project documentation available in README.md"
	@echo "🔍 API documentation: Run 'make mlflow-ui' for MLflow tracking"
	@echo "⚙️  Airflow documentation: Run 'make airflow-start' for DAG visualization"
	@echo "${GREEN}✅ Documentation ready!${NC}"

# =============================================================================
# Development Utilities
# =============================================================================

status: ## Show project status
	@echo "${BLUE}Project Status${NC}"
	@echo "${BLUE}==============${NC}"
	@echo ""
	@echo "📁 Project: $(PROJECT_NAME)"
	@echo "🐍 Python: $(shell $(PYTHON) --version 2>/dev/null || echo 'Not found')"
	@echo "📦 Pip: $(shell $(PIP) --version 2>/dev/null | cut -d' ' -f2 || echo 'Not found')"
	@echo "🐳 Docker: $(shell docker --version 2>/dev/null | cut -d' ' -f3 | sed 's/,//' || echo 'Not found')"
	@echo "⚙️  Astro CLI: $(shell astro version 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "📊 MLflow experiments:"
	@if [ -d "mlruns" ]; then \
		echo "  Local: $(shell find mlruns -name 'meta.yaml' | wc -l) experiments"; \
	else \
		echo "  Local: No experiments found"; \
	fi
	@if [ -d "astro-airflow/include/mlruns" ]; then \
		echo "  Airflow: $(shell find astro-airflow/include/mlruns -name 'meta.yaml' | wc -l) experiments"; \
	else \
		echo "  Airflow: No experiments found"; \
	fi
	@echo ""
	@echo "🐳 Docker images:"
	@docker images | grep $(PROJECT_NAME) || echo "  No project images found"

info: status ## Alias for status

# =============================================================================
# Quick Start
# =============================================================================

quickstart: ## Quick start development environment
	@echo "${BLUE}🚀 Quick Start Setup${NC}"
	@echo "${BLUE}==================${NC}"
	make setup
	make create-test-data
	make ci-test
	@echo ""
	@echo "${GREEN}✅ Quick start completed!${NC}"
	@echo ""
	@echo "${YELLOW}Next steps:${NC}"
	@echo "  1. Start Airflow: ${GREEN}make airflow-start${NC}"
	@echo "  2. View MLflow UI: ${GREEN}make mlflow-ui${NC}"
	@echo "  3. Run training: ${GREEN}make train${NC}"
	@echo "  4. Run deployment: ${GREEN}make deploy${NC}"
	@echo "  5. Run monitoring: ${GREEN}make monitor${NC}"