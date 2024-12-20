# Regularization and Hyper-Parameter Tuning

## Overview

In [this notebook](./Regularization_Hyper-parameter_Tuning.ipynb), we perform Ridge Regression and Ridge Logistic Regression on two datasets: wine quality and bank marketing. The notebook focuses on data preprocessing, model training, regularization, and hyper-parameter tuning using cross-validation.

## Datasets

- **bank_marketing**: Bank marketing campaign data.
- **wine_quality**: Wine quality ratings data.

## Preprocessing

- Dropped irrelevant features (e.g., `month`, `day_of_week`).
- One-hot encoding for categorical data.
- Min-max normalization.
- Data split: 80% training, 20% testing.

## Models and Methods

- **Ridge Regression** (for wine quality prediction).
- **Ridge Logistic Regression** (for bank marketing classification).
- Hyper-parameter tuning using k-fold cross-validation (learning rates and regularization constants).

## Evaluation

Metrics: RMSE, log-loss, convergence plots.
