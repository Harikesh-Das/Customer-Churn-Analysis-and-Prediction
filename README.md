# Customer Churn Analysis and Prediction

This project analyzes customer churn data from a telecommunications company and builds predictive models to identify at-risk customers.

## Project Overview

The project aims to:
- Analyze customer churn patterns
- Identify key factors influencing churn
- Build machine learning models to predict churn
- Provide actionable insights for customer retention

## Dataset

The dataset contains information about telecom customers including:
- Customer demographics (gender, age, partner, dependents)
- Account information (tenure, contract type, payment method)
- Services (phone, internet, streaming services)
- Charges (monthly and total charges)
- Churn status (target variable)

## Features

### Data Preparation
- Loads and preprocesses the dataset
- Handles missing values
- Encodes categorical variables
- Prepares data for machine learning

### Data Splitting
- Splits data into 80% training and 20% testing sets
- Ensures representative distribution of churn cases

### Feature Selection
- Identifies most important features using Random Forest
- Selects top 10 features for model training
- Provides feature importance visualization

### Model Selection & Training
- Tests multiple algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Trains models on selected features

### Model Evaluation
- Comprehensive performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Confusion matrix analysis
- ROC curve visualization
- Model comparison charts

### Business Insights
- Analyzes churn patterns by contract type
- Examines tenure-based churn rates
- Provides actionable recommendations

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete analysis:
```bash
python customer_churn_analysis.py
```

The script will:
1. Load and preprocess the data
2. Split data into training/testing sets
3. Select important features
4. Train multiple models
5. Evaluate and compare models
6. Generate visualizations
7. Provide business insights

## Output

The script generates:
- `feature_importance.png` - Feature importance visualization
- `model_evaluation.png` - Model comparison and evaluation charts
- Console output with detailed analysis and recommendations

## Key Findings

- Month-to-month contracts have the highest churn rate
- Customers with tenure less than 12 months are at higher risk
- High monthly charges correlate with increased churn
- The Random Forest model typically performs best for this dataset

## Recommendations

1. Focus retention efforts on month-to-month contract customers
2. Implement early intervention programs for new customers
3. Monitor customers with high monthly charges
4. Use the trained model for proactive customer identification
5. Develop targeted retention strategies for high-risk segments
