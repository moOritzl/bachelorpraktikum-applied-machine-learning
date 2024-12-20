# Linear Classification with AdaGrad

## Overview

In [this notebook](./Linear_Classification_AdaGrad.ipynb), we implement linear classification using gradient descent and
AdaGrad for adaptive step length. The notebook covers data preprocessing, model training, and convergence analysis for
two datasets: bank marketing and occupancy detection.

## Datasets

- **Bank Marketing Dataset**:  
  Contains features related to direct marketing campaigns for a bank.

- **Occupancy Detection Dataset**:  
  Consists of features related to indoor environmental conditions.

## Preprocessing

- Dropped unnecessary features (e.g., dates, categorical columns).
- Handled missing values by filling with 0.
- One-hot encoding for categorical data.
- Min-max normalization applied to all features.
- Data split: 80% training, 20% testing.

## Models

- **Gradient Descent with Bold Driver**
- **AdaGrad for Adaptive Step Length**

## Evaluation

Metrics: Training loss difference, test log-loss, convergence plots.
