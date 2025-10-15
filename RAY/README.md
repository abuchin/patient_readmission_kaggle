# Ray Tune XGBoost Hyperparameter Optimization

This directory contains a comprehensive hyperparameter optimization (HPO) pipeline for XGBoost classification on the diabetic readmission dataset using Ray Tune and MLflow.

## Overview

The `ray_tune_xgboost.py` script implements a large-scale hyperparameter optimization workflow that:

- **Preprocesses** the diabetic readmission dataset with standardized numeric features and one-hot encoded categorical features
- **Optimizes** XGBoost hyperparameters using Ray Tune with ASHA (Asynchronous Successive Halving Algorithm) scheduler
- **Tracks** experiments using MLflow for comprehensive logging of parameters, metrics, and model artifacts
- **Evaluates** models using 5-fold cross-validation with multiple metrics (AUC, AP, Accuracy, F1)
- **Saves** the best model and configuration for future use

## Features

### Data Preprocessing
- **Target Encoding**: Merges readmission labels (`<30` and `>30` → `YES`, keeps `NO`)
- **Feature Engineering**: 
  - Standardizes numeric features using `StandardScaler`
  - One-hot encodes categorical features with `drop_first=True`
  - Handles unknown categories gracefully
- **Class Balancing**: Automatically computes `scale_pos_weight` for imbalanced datasets

### Hyperparameter Optimization
- **Search Algorithm**: OptunaSearch (with fallback to random search)
- **Scheduler**: ASHA (Asynchronous Successive Halving Algorithm)
- **Search Space**: 8 hyperparameters with appropriate ranges:
  - `n_estimators`: 200-900
  - `max_depth`: 3-10
  - `learning_rate`: 1e-3 to 3e-1 (log scale)
  - `subsample`: 0.6-1.0
  - `colsample_bytree`: 0.6-1.0
  - `min_child_weight`: 1e-1 to 1e1 (log scale)
  - `reg_alpha`: 1e-8 to 1e-1 (log scale)
  - `reg_lambda`: 1e-2 to 1e1 (log scale)

### Model Evaluation
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Metrics**: ROC-AUC, Average Precision, Accuracy, F1-Score
- **Final Evaluation**: Best model trained on full training set, evaluated on test set

### Experiment Tracking
- **MLflow Integration**: Complete experiment tracking with nested runs
- **Artifacts**: Best model, configuration, and preprocessing pipeline saved
- **Metrics**: All validation and test metrics logged
- **Parameters**: All hyperparameters logged for reproducibility

## Requirements

```bash
# Core dependencies
ray[tune]>=2.0.0
xgboost>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
mlflow>=1.20.0

# Optional: Optuna for advanced search
optuna>=3.0.0
```

## Usage

### Basic Usage

```bash
python ray_tune_xgboost.py --data /path/to/diabetic_data.csv
```

## Usage using docker

```bash
docker run --rm -it \
  -v "$PWD":/work -w /work \
  abuchin/patient-env:1 \
  python ray_tune_xgboost.py --data /work/diabetic_data.csv
```

### Advanced Usage

```bash
python ray_tune_xgboost.py \
    --data /home/ec2-user/projects/patient_selection/data/diabetic_data.csv \
    --num-samples 50 \
    --cpus-per-trial 4 \
    --gpus-per-trial 0 \
    --test-size 0.2 \
    --ray-dir ./ray_exp \
    --mlruns-dir /home/ec2-user/projects/patient_selection/code/RAY/mlruns \
    --seed 42
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | **Required** | Path to diabetic_data.csv |
| `--test-size` | float | 0.2 | Test set size (0.0-1.0) |
| `--num-samples` | int | 30 | Number of Ray Tune trials |
| `--cpus-per-trial` | int | 2 | CPUs allocated per trial |
| `--gpus-per-trial` | float | 0.0 | GPUs allocated per trial |
| `--ray-dir` | str | "ray_exp" | Directory for Ray Tune outputs |
| `--mlruns-dir` | str | "./mlruns" | MLflow backend store directory |
| `--seed` | int | 42 | Random seed for reproducibility |

## Output Structure

```
RAY/
├── ray_tune_xgboost.py          # Main HPO script
├── README.md                     # This documentation
├── ray_exp/                      # Ray Tune experiment results
│   └── xgb_hpo/                  # Experiment runs and checkpoints
│       └── best_config.json     # Best hyperparameters found
└── mlruns/                       # MLflow experiment tracking
    └── 209143223549175349/       # Experiment ID
        ├── meta.yaml             # Experiment metadata
        └── [run_ids]/            # Individual trial runs
            ├── artifacts/        # Model artifacts
            ├── metrics/          # Validation metrics
            ├── params/           # Hyperparameters
            └── tags/             # Run metadata
```

## Key Functions

### `build_preprocessor(X)`
Creates a scikit-learn `ColumnTransformer` that:
- Standardizes numeric columns
- One-hot encodes categorical columns
- Handles unknown categories

### `trainable(config, X_train, y_train_num, preprocessor, mlruns_uri)`
Ray Tune trainable function that:
- Performs 5-fold cross-validation
- Trains XGBoost with given hyperparameters
- Logs metrics to MLflow
- Reports results to Ray Tune

### `fit_best_and_log(X_train, y_train_num, X_test, y_test_num, preprocessor, best_config, mlruns_uri)`
Fits the best model on full training data and:
- Evaluates on test set
- Logs final model to MLflow
- Saves configuration as artifact

## Monitoring and Visualization

### MLflow UI
Launch the MLflow tracking UI to monitor experiments:

On the instance run this command:

```bash
mlflow ui --backend-store-uri file:/home/ec2-user/projects/patient_selection/code/RAY/mlruns --host 127.0.0.1 --port 5000
```

Connect to ML Flow from your terminal
```bash
ssh -i /path/to/your/permission_file.pem  -N -L 5001:127.0.0.1:5000 ec2-user@EC2_instance_public_DNS```

### Ray Dashboard
Monitor Ray cluster and trial progress:

```bash
# Ray dashboard is automatically available at http://127.0.0.1:8265
# when running Ray Tune
```

## Example Output

```
Best hyperparameters:
{
  "n_estimators": 684,
  "max_depth": 9,
  "learning_rate": 0.0314,
  "subsample": 0.8234,
  "colsample_bytree": 0.6812,
  "min_child_weight": 7.6861,
  "reg_alpha": 0.0001,
  "reg_lambda": 0.1234
}

Test metrics with best config:
test_auc: 0.7234
test_ap: 0.3456
test_acc: 0.6789
test_f1: 0.4567
```

## Performance Considerations

- **Parallelization**: Ray Tune automatically parallelizes trials across available CPUs
- **Resource Management**: Adjust `cpus_per_trial` based on your system capabilities
- **Memory Usage**: XGBoost with `tree_method="hist"` is memory-efficient
- **Early Stopping**: ASHA scheduler terminates poor-performing trials early

## Troubleshooting

### Common Issues

1. **MLflow URI Issues**: Ensure the `mlruns_dir` path is absolute and writable
2. **Ray Initialization**: Check that Ray can initialize properly in your environment
3. **Memory Issues**: Reduce `num_samples` or `cpus_per_trial` if running out of memory
4. **Optuna Import Error**: Install optuna or the script will fall back to random search

### Debug Mode

Add verbose logging by modifying the `RunConfig`:

```python
run_config = RunConfig(
    name="xgb_hpo",
    storage_path=os.path.abspath(args.ray_dir),
    verbose=3,  # Increase verbosity
)
```

## Contributing

When modifying this script:

1. Maintain backward compatibility with existing MLflow runs
2. Update this README if adding new features
3. Test with different dataset sizes and configurations
4. Ensure proper cleanup of Ray resources

## License

This code is part of the patient selection project. Please refer to the main project license for usage terms.
