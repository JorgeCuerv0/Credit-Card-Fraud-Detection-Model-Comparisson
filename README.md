# Credit Card Fraud Detection: Classification Model Comparisons

This repository contains a Jupyter Notebook demonstrating and comparing various classification models for detecting fraudulent transactions. The goal is to evaluate the performance of different models and highlight their strengths and weaknesses in identifying fraudulent behavior.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Models Implemented](#models-implemented)
- [Performance Metrics](#performance-metrics)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Overview
Fraud detection is a critical application of machine learning in the finance and e-commerce industries. This project explores multiple classification models to detect fraudulent transactions in a highly imbalanced dataset. The notebook provides insights into model performance and strategies for improving accuracy, recall, and precision.

## Technologies Used
- **Python**: Primary programming language
- **Jupyter Notebook**: For creating and visualizing the project
- **Libraries**:
  - `scikit-learn`: For implementing logistic regression and evaluating metrics
  - `xgboost`: For implementing the Extreme Gradient Boosting (XGBoost) model
  - `tensorflow`: For building a sequential neural network
  - `pandas`, `numpy`: For data manipulation and preprocessing
  - `matplotlib`, `seaborn`: For data visualization

## Project Workflow
1. **Data Preprocessing**:
   - Log transformation for skewed data
   - Feature engineering to create meaningful predictors
   - One-hot encoding for categorical variables
   - Standardization of numerical features

2. **Model Implementation**:
   - Logistic Regression
   - Neural Network
   - Extreme Gradient Boosting (XGBoost)

3. **Evaluation Metrics**:
   - Precision, Recall, F1-Score
   - Accuracy
   - ROC-AUC Score

4. **Comparison and Insights**:
   - Performance of models is compared to highlight strengths and weaknesses.

## Models Implemented
1. **Logistic Regression**:
   - A basic yet effective linear classifier for fraud detection.
   - Results showed high precision but lower recall for detecting fraud.

2. **Neural Network**:
   - A sequential model with custom layers and dropout for overfitting prevention.
   - Improved recall and balanced performance with an ROC-AUC score of 0.97.

3. **Extreme Gradient Boosting (XGBoost)**:
   - An advanced tree-based model that boosts accuracy by learning from errors in prior trees.
   - Delivered the highest recall of 87% for fraudulent transactions, achieving an ROC-AUC score of 0.98.

## Performance Metrics
| Model               | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | 80%       | 66%    | 73%      | 0.97    |
| Neural Network      | 87%       | 78%    | 82%      | 0.97    |
| XGBoost             | 25%       | 87%    | 39%      | 0.98    |

## Future Work
Explore other advanced models such as Random Forest and Support Vector Machines.
Implement techniques to handle class imbalance more effectively.
Perform hyperparameter tuning to further improve model performance.

## Acknowledgements
The dataset used for this project is sourced from Kaggle.
Special thanks to the developers of the libraries used in this project.
