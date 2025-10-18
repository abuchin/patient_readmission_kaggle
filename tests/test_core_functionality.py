"""
Unit tests for data preprocessing functionality
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class TestDataPreprocessing:
    """Test data preprocessing functionality"""
    
    def test_data_loading(self, sample_diabetes_data):
        """Test that sample data loads correctly"""
        df = pd.read_csv(sample_diabetes_data)
        
        assert len(df) == 1000
        assert 'readmitted' in df.columns
        assert 'race' in df.columns
        assert 'gender' in df.columns
        
    def test_categorical_encoding(self, sample_diabetes_data):
        """Test categorical variable encoding"""
        df = pd.read_csv(sample_diabetes_data)
        
        # Get categorical columns
        categorical_cols = ['race', 'gender', 'age', 'A1Cresult', 'metformin', 'insulin', 'change', 'diabetesMed']
        
        # Test OneHotEncoder
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
        encoded = encoder.fit_transform(df[categorical_cols])
        
        assert encoded.shape[0] == len(df)
        assert encoded.shape[1] > len(categorical_cols)  # Should expand due to one-hot encoding
        
    def test_numeric_scaling(self, sample_diabetes_data):
        """Test numeric variable scaling"""
        df = pd.read_csv(sample_diabetes_data)
        
        # Get numeric columns
        numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                       'num_medications', 'number_outpatient', 'number_emergency', 
                       'number_inpatient', 'number_diagnoses']
        
        # Test StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[numeric_cols])
        
        assert scaled.shape == df[numeric_cols].shape
        # Check that scaled data has mean ~0 and std ~1
        assert abs(scaled.mean()) < 0.1
        assert abs(scaled.std() - 1) < 0.1
        
    def test_full_preprocessing_pipeline(self, sample_diabetes_data):
        """Test complete preprocessing pipeline"""
        df = pd.read_csv(sample_diabetes_data)
        
        # Separate features and target
        X = df.drop('readmitted', axis=1)
        y = df['readmitted']
        
        # Define feature types
        numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                           'num_medications', 'number_outpatient', 'number_emergency', 
                           'number_inpatient', 'number_diagnoses']
        categorical_features = ['race', 'gender', 'age', 'A1Cresult', 'metformin', 
                              'insulin', 'change', 'diabetesMed']
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Fit and transform
        X_processed = preprocessor.fit_transform(X)
        
        assert X_processed.shape[0] == len(df)
        assert X_processed.shape[1] > len(numeric_features)  # Should be expanded by one-hot encoding
        
    def test_target_encoding(self, sample_diabetes_data):
        """Test target variable encoding"""
        df = pd.read_csv(sample_diabetes_data)
        
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(df['readmitted'])
        
        assert len(y_encoded) == len(df)
        assert set(y_encoded) == {0, 1, 2}  # Should have 3 classes
        assert len(le.classes_) == 3


class TestModelTraining:
    """Test model training functionality"""
    
    def test_xgboost_import(self):
        """Test that XGBoost can be imported"""
        try:
            import xgboost as xgb
            assert hasattr(xgb, 'XGBClassifier')
        except ImportError:
            pytest.skip("XGBoost not available")
            
    def test_simple_model_training(self, sample_diabetes_data):
        """Test basic model training"""
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            pytest.skip("Required packages not available")
            
        df = pd.read_csv(sample_diabetes_data)
        
        # Simple preprocessing for testing
        numeric_cols = ['time_in_hospital', 'num_medications', 'number_diagnoses']
        X = df[numeric_cols].fillna(0)
        
        le = LabelEncoder()
        y = le.fit_transform(df['readmitted'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train simple model
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1, 2})


class TestMLflowIntegration:
    """Test MLflow integration"""
    
    def test_mlflow_import(self):
        """Test that MLflow can be imported"""
        try:
            import mlflow
            import mlflow.xgboost
            assert hasattr(mlflow, 'start_run')
        except ImportError:
            pytest.skip("MLflow not available")
            
    def test_mlflow_logging(self, mlflow_tracking_uri):
        """Test basic MLflow logging"""
        try:
            import mlflow
            import mlflow.sklearn
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import make_classification
        except ImportError:
            pytest.skip("Required packages not available")
            
        # Create simple dataset
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        # Train model and log to MLflow
        with mlflow.start_run():
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("n_samples", len(X))
            mlflow.sklearn.log_model(model, "model")
            
            run_id = mlflow.active_run().info.run_id
            
        # Verify run was logged
        runs = mlflow.search_runs()
        assert len(runs) >= 1
        assert run_id in runs['run_id'].values


class TestMonitoring:
    """Test monitoring and drift detection functionality"""
    
    def test_evidently_import(self):
        """Test that Evidently can be imported"""
        try:
            import evidently
            from evidently.report import Report
            assert hasattr(evidently, 'report')
        except ImportError:
            pytest.skip("Evidently not available")
            
    def test_drift_detection(self, monitoring_data):
        """Test basic drift detection"""
        try:
            import evidently
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError:
            pytest.skip("Evidently not available")
            
        ref_path, cur_path = monitoring_data
        ref_data = pd.read_csv(ref_path)
        cur_data = pd.read_csv(cur_path)
        
        # Create drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_data, current_data=cur_data)
        
        # Get results
        results = report.as_dict()
        assert 'metrics' in results
        
    def test_data_quality_checks(self, monitoring_data):
        """Test data quality validation"""
        ref_path, cur_path = monitoring_data
        ref_data = pd.read_csv(ref_path)
        cur_data = pd.read_csv(cur_path)
        
        # Basic quality checks
        assert len(ref_data) > 0
        assert len(cur_data) > 0
        assert ref_data.columns.tolist() == cur_data.columns.tolist()
        assert not ref_data.isnull().all().any()
        assert not cur_data.isnull().all().any()


class TestDockerIntegration:
    """Test Docker-related functionality"""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfiles exist"""
        import os
        
        # Check main Dockerfile
        assert os.path.exists("Dockerfile") or os.path.exists("dockerfile")
        
        # Check component Dockerfiles
        assert os.path.exists("DEPLOY/Dockerfile")
        assert os.path.exists("MONITOR/Dockerfile")
        
    def test_requirements_files_exist(self):
        """Test that requirements files exist"""
        import os
        
        assert os.path.exists("requirements.txt")
        assert os.path.exists("astro-airflow/requirements.txt")