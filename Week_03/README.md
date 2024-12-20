# Linear Regression with Gradient Descent

## Overview

In [this notebook](./Data_Processing_Gradient_Descent.ipynb), we preprocess datasets and implement linear regression using gradient descent, including fixed and dynamic step length
methods.

## Datasets

- **airq402.csv**: Air quality and airline data.
- **winequality-red.csv & winequality-white.csv**: Wine quality datasets.

## Preprocessing

- One-hot encoding for categorical data.
- No missing values found; rows dropped if needed.
- Data split into 80% training and 20% testing.

## Gradient Descent

- **Fixed Step Lengths**: Tested α = 0.001 (slow), 0.01 (faster), 0.1 (unstable).
- **Dynamic Methods**: Armijo (fast but unstable initially), Bold Driver (inconsistent).
