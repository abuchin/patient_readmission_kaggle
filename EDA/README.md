# EDA - Exploratory Data Analysis

## Overview

This notebook performs comprehensive exploratory data analysis on the diabetic readmission dataset, builds baseline machine learning models, and establishes the foundation for the patient readmission prediction project.

## Dataset

- **Source**: Diabetes 130-US hospitals dataset (1999-2008)
- **Size**: ~100,000 patient records
- **Features**: 50+ attributes including demographics, medical procedures, medications, and diagnostic information
- **Target**: Patient readmission status (merged into binary: YES/NO)

## Key Activities

### 1. Data Exploration
- **Distribution Analysis**: Examined numeric and categorical feature distributions
- **Missing Data**: Identified that missing values are replaced with "Unknown/Invalid"
- **Feature Types**:
  - Numeric: `num_lab_procedures`, `num_medications`, `number_diagnoses`, `number_inpatient`
  - Categorical: `race`, `gender`, `age`, medications, diagnostic codes

### 2. Data Preprocessing
- **Target Engineering**: Merged `<30` and `>30` readmission labels into single "YES" class
- **Train/Test Split**: 80/20 stratified split to maintain class balance
- **Feature Transformation**:
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features (drop_first=True)
  - Final feature space: 2,360 dimensions after encoding

### 3. Dimensionality Reduction (PCA)
- **Components**: Reduced to 50 principal components
- **Variance Explained**: ~80% with 50 components
- **Finding**: High-dimensional dataset with non-linear relationships suggests using all original features for modeling

### 4. Model Development

#### Random Forest Classifier
- **Configuration**: 200 estimators, balanced class weights, SMOTE oversampling
- **Performance**: 66% accuracy
  - Precision (NO): 0.67, (YES): 0.65
  - Recall (NO): 0.73, (YES): 0.58
  - F1-Score (NO): 0.70, (YES): 0.61
- **Top Features**: `patient_nbr`, `encounter_id`, `num_lab_procedures`, `time_in_hospital`

#### XGBoost Classifier
- **Configuration**: 200 estimators, max_depth=6, learning_rate=0.05, SMOTE oversampling
- **Performance**: 67% accuracy (best baseline model)
  - Precision (NO): 0.69, (YES): 0.65
  - Recall (NO): 0.71, (YES): 0.62
  - F1-Score (NO): 0.70, (YES): 0.64
- **Top Features**: `number_inpatient`, `time_in_hospital`, `num_lab_procedures`, `discharge_disposition_id`

## Key Findings

### Data Characteristics
1. **Distribution Patterns**: Some features are normally distributed (`num_lab_procedures`, `num_medications`) while others are skewed, indicating non-linear relationships
2. **Class Imbalance**: Dataset has more non-readmitted than readmitted patients
3. **Demographics**: Caucasian majority in race distribution; age relatively balanced across groups
4. **High Dimensionality**: 2,360 features after encoding suggests complex feature space

### Model Insights
1. **XGBoost Superiority**: XGBoost slightly outperforms Random Forest (67% vs 66% accuracy)
2. **Balanced Errors**: Both models show relatively balanced Type I and Type II errors
3. **Key Predictors**: Number of inpatient visits, time in hospital, and lab procedures are highly predictive
4. **SMOTE Impact**: Using SMOTE oversampling and balanced class weights significantly improved model recall

### Clinical Interpretation
- **number_inpatient**: Patients with more prior inpatient visits are more likely to be readmitted (intuitive finding)
- **time_in_hospital**: Longer hospital stays correlate with readmission risk
- **num_lab_procedures**: More extensive testing indicates complexity of patient condition

## Conclusions

1. **Baseline Performance**: 67% accuracy with XGBoost provides solid baseline (vs 50% random guess)
2. **Model Selection**: **XGBoost chosen as baseline model** for further optimization due to:
   - Better performance than Random Forest
   - Fast training on tabular data
   - Balanced precision/recall across classes
3. **Next Steps**: XGBoost model is ready for large-scale hyperparameter optimization using Ray Tune
4. **Feature Strategy**: Use all original features (not PCA) for better pattern recognition

## Files Generated

- `rf.pkl`: Saved Random Forest model
- `xgb_model.json`: Saved XGBoost model
- Various visualizations: distributions, PCA plots, confusion matrices, feature importance charts

## Tools Used

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML Libraries**: scikit-learn, xgboost
- **Class Balancing**: imblearn (SMOTE)
- **Dimensionality Reduction**: PCA

## Usage

```bash
# Launch notebook
jupyter notebook EDA.ipynb

# Run all cells to:
# 1. Load and explore the dataset
# 2. Preprocess features and engineer target
# 3. Perform PCA analysis
# 4. Train Random Forest and XGBoost models
# 5. Evaluate and compare model performance
# 6. Analyze feature importance
```

## Next Phase

The XGBoost baseline model identified in this analysis will be used for:
- Large-scale hyperparameter optimization with Ray Tune (see `../RAY/`)
- MLflow experiment tracking
- Production deployment via Docker (see `../DEPLOY/`)

## References

For detailed implementation and full analysis, see `EDA.ipynb`.
