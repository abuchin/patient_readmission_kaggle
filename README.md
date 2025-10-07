# Patient Readmission Prediction Project

## Overview

This project aims to create a machine learning model that can predict whether a given patient will be readmitted to the hospital based on collected patient data. The model will be tuned using Ray, orchestrated using Airflow, and deployed on HuggingFace. Additionally, it will be connected to a website where users can ask questions about the dataset and get answers from an LLM.

## Project Structure

```
patient_selection/
├── code/
│   └── EDA.ipynb          # Exploratory Data Analysis notebook
├── data/
│   ├── diabetic_data.csv  # Main dataset
│   └── readmission.zip    # Additional data files
└── README.md              # This file
```

## Dataset

The project uses a diabetic patient dataset (`diabetic_data.csv`) that contains various patient attributes and outcomes. The dataset includes:

- **Demographic information**: Gender, race, age
- **Medical procedures**: Number of lab procedures, medications
- **Diagnostic information**: Number of diagnoses, inpatient visits
- **Target variable**: Readmission status

The dataset could be found on kaggle. The file name diabetic_data.csv is what we want and would reuse for the whole project.
https://www.kaggle.com/datasets/brandao/diabetes

### Data Characteristics

Based on the exploratory data analysis:

1. **Missing Data Handling**: Features with missing values are replaced with "Unknown/Invalid" values
2. **Distribution Patterns**:
   - Some numeric features (num_lab_procedures, num_medications) are close to normally distributed
   - Other features (number_diagnoses, number_inpatient) are not normally distributed
   - This suggests that non-linear methods might be beneficial for pattern recognition

3. **Categorical Features**:
   - Most patients are Caucasian, with smaller representation of other race groups
   - Age distribution is relatively equal across different groups
   - Drug usage distribution is uneven, with some patients having taken drugs while others haven't

## Methodology

### Data Preprocessing

1. **Target Variable**: The `readmitted` column serves as the target variable
2. **Feature Engineering**: Categorical features are converted using one-hot encoding
3. **Data Splitting**: 
   - 80% training, 20% testing
   - Stratified splitting to maintain equal representation of readmitted/not-readmitted patients
   - Training data is further split into two halves for potential model retraining in case of data drift

### Model Development

The project implements a **Random Forest Classifier** with the following configuration:
- 200 estimators (trees)
- No maximum depth limit (trees expand fully)
- Balanced class weights to handle class imbalance
- Multi-core processing enabled

### Model Evaluation

The model performance is evaluated using:
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Detailed metrics including precision, recall, and F1-score
- **Confusion Matrix**: Visual representation of prediction vs. actual outcomes
- **Feature Importance**: Analysis of which features contribute most to predictions

## Key Findings

1. **Feature Importance**: The model identifies the top 20 most important features for predicting readmission
2. **Model Performance**: The Random Forest model provides baseline performance metrics for patient readmission prediction
3. **Data Quality**: The dataset contains various data quality considerations that need to be addressed during preprocessing

## Future Work

The project is designed to be extended with:
- Model fine-tuning using Ray
- Orchestration with Airflow
- Deployment on HuggingFace
- Integration with a web interface for LLM-powered dataset queries

## Requirements

The project uses the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Getting Started

1. Ensure you have the required dependencies installed
2. Place your dataset in the `data/` directory
3. Run the EDA notebook to explore the data and train the initial model
4. Follow the model development pipeline outlined in the notebook

## Data Location

The main dataset should be located at:
```
/home/ec2-user/projects/patient_selection/data/diabetic_data.csv
```

## License

This project is part of a patient readmission prediction system designed for healthcare applications.
