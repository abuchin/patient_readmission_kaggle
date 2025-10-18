---
name: MLOps Pipeline Issue
about: Report issues specific to the MLOps pipeline components
title: '[MLOPS] '
labels: 'mlops'
assignees: ''

---

**Pipeline Component**
Which MLOps component is having issues?
- [ ] Data ingestion/preprocessing
- [ ] Feature engineering
- [ ] Model training (Ray Tune + XGBoost)
- [ ] Model validation
- [ ] Model deployment
- [ ] Model serving (REST API)
- [ ] Model monitoring
- [ ] Drift detection
- [ ] Automated retraining
- [ ] MLflow experiment tracking
- [ ] Airflow orchestration
- [ ] Docker containerization

**Issue Type**
- [ ] Performance issue
- [ ] Data quality issue
- [ ] Model accuracy issue
- [ ] Deployment failure
- [ ] Monitoring alert
- [ ] Pipeline failure
- [ ] Resource usage issue
- [ ] Configuration issue

**Describe the issue**
A clear and concise description of the MLOps issue.

**Pipeline Context**
- Experiment name: [e.g. xgb_diabetic_readmission_hpo]
- Run ID (if applicable): [e.g. 85a49ebf8c794687898fa1e3c07b69e1]
- Model version: [e.g. v1.2.3]
- Environment: [dev/staging/production]

**Data Context**
- Training data size: [e.g. 101,766 samples]
- Features: [e.g. 13 numeric + 36 categorical = 2436 after preprocessing]
- Target distribution: [e.g. NO: 54%, <30: 11%, >30: 35%]
- Data drift detected: [Yes/No]

**Model Performance**
- Training accuracy: [e.g. 0.85]
- Validation accuracy: [e.g. 0.82]
- Current accuracy: [e.g. 0.78]
- Performance degradation: [e.g. 5%]

**Error Messages/Logs**
```
Paste relevant error messages and logs here
```

**MLflow Tracking**
- Tracking URI: [e.g. file:/tmp/mlruns]
- Experiment ID: [e.g. 391594787653651940]
- Artifacts location: [e.g. /tmp/mlruns/391594787653651940/]

**Monitoring Data**
- Reference data: [e.g. ref.csv, 18MB]
- Current data: [e.g. cur.csv, 18MB]
- Drift score: [e.g. PSI = 0.15]
- Feature drift detected: [e.g. 3 out of 49 features]

**Environment Details**
- Python version: [e.g. 3.12]
- MLflow version: [e.g. 2.17.0]
- XGBoost version: [e.g. 2.1.0]
- Scikit-learn version: [e.g. 1.5.0]
- Airflow version: [e.g. 2.7.0]
- Docker version: [e.g. 24.0.0]

**Reproduction Steps**
1. Load data: `python -c "import pandas as pd; df = pd.read_csv('data.csv')"`
2. Run training: `astro run deploy_on_start_bash`
3. Check monitoring: `astro run monitor_and_retrain_bash`
4. Observe error: [describe what happens]

**Expected vs Actual Behavior**
- **Expected**: [What should happen]
- **Actual**: [What actually happens]

**Impact Assessment**
- [ ] Pipeline completely broken
- [ ] Performance degraded but functional
- [ ] Minor issue, workaround available
- [ ] Cosmetic issue

**Urgency**
- [ ] Critical - Production down
- [ ] High - Affecting model quality
- [ ] Medium - Workflow impacted
- [ ] Low - Enhancement/optimization