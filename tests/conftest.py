"""
Test configuration for pytest
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory"""
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def sample_diabetes_data(test_data_dir):
    """Create sample diabetes dataset for testing"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "race": np.random.choice(
            ["Caucasian", "AfricanAmerican", "Hispanic", "Other"], n_samples
        ),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "age": np.random.choice(
            ["[70-80)", "[60-70)", "[50-60)", "[40-50)", "[30-40)"], n_samples
        ),
        "time_in_hospital": np.random.randint(1, 15, n_samples),
        "num_lab_procedures": np.random.randint(0, 50, n_samples),
        "num_procedures": np.random.randint(0, 10, n_samples),
        "num_medications": np.random.randint(1, 25, n_samples),
        "number_outpatient": np.random.randint(0, 10, n_samples),
        "number_emergency": np.random.randint(0, 5, n_samples),
        "number_inpatient": np.random.randint(0, 10, n_samples),
        "number_diagnoses": np.random.randint(1, 15, n_samples),
        "A1Cresult": np.random.choice(["None", "Norm", ">7", ">8"], n_samples),
        "metformin": np.random.choice(["No", "Steady", "Up", "Down"], n_samples),
        "insulin": np.random.choice(["No", "Steady", "Up", "Down"], n_samples),
        "change": np.random.choice(["No", "Ch"], n_samples),
        "diabetesMed": np.random.choice(["No", "Yes"], n_samples),
        "readmitted": np.random.choice(["NO", "<30", ">30"], n_samples),
    }

    df = pd.DataFrame(data)
    file_path = os.path.join(test_data_dir, "sample_diabetes_data.csv")
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture(scope="session")
def monitoring_data(test_data_dir):
    """Create monitoring reference and current data"""
    np.random.seed(42)
    n_samples = 5000

    # Reference data
    ref_data = pd.DataFrame(
        {f"feature_{i}": np.random.randn(n_samples) for i in range(20)}
    )
    ref_data["target"] = np.random.choice([0, 1, 2], n_samples)

    # Current data with slight drift
    np.random.seed(123)
    cur_data = pd.DataFrame(
        {
            f"feature_{i}": np.random.randn(n_samples) + (0.1 if i < 5 else 0)
            for i in range(20)
        }
    )
    cur_data["target"] = np.random.choice([0, 1, 2], n_samples)

    ref_path = os.path.join(test_data_dir, "ref_data.csv")
    cur_path = os.path.join(test_data_dir, "cur_data.csv")

    ref_data.to_csv(ref_path, index=False)
    cur_data.to_csv(cur_path, index=False)

    return ref_path, cur_path


@pytest.fixture(scope="function")
def mlflow_tracking_uri(tmp_path):
    """Set up temporary MLflow tracking"""
    tracking_uri = f"file://{tmp_path}/mlruns"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    return tracking_uri
