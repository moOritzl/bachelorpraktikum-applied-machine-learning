# Generalized Linear Models and Polynomial Regression

## Overview

In [this notebook](./Generalized_Linear_Models_Polynomial_Regression.ipynb), we explore Ordinary Least Squares, Ridge, LASSO, and Stochastic Gradient Descent regressions. We also implement polynomial regression and analyze the effects of different polynomial degrees and regularization.

## Datasets

- **D1**: Synthetic data with quadratic relationships.
- **winequality-red.csv**: Red wine quality data with multiple features.

## Preprocessing

- Min-max normalization applied to `winequality-red.csv`.
- Synthetic data `D1` generated with noise.

## Models and Methods

- **Ordinary Least Squares**: Baseline linear regression.
- **Ridge Regression**: L2 regularization with hyperparameter tuning.
- **LASSO Regression**: L1 regularization with hyperparameter tuning.
- **Stochastic Gradient Descent**: Regularized linear regression.
- **Polynomial Regression**: High-degree polynomial fitting and regularization.

## Evaluation

Metrics: RMSE, relative difference between train and test RMSE, cross-validation scores, convergence plots.
