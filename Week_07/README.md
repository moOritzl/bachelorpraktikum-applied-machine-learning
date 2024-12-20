# K-Nearest Neighbor (KNN) Classification

## Overview

In [this notebook](./K-Nearest_Neighbor.ipynb), we implement the K-Nearest Neighbor (KNN) algorithm for classification. The notebook explores data preprocessing, custom implementation of KNN, model evaluation, hyperparameter tuning, and comparison with tree-based methods.

## Datasets

- **Iris Dataset**: Three-class classification of iris flower species.
- **Wine Quality Dataset**: Multiclass classification of red wine quality.

## Preprocessing

- Min-max normalization applied to both datasets.
- Data split: 70% training, 30% testing.

## Models and Methods

- **Custom KNN Implementation**: Euclidean distance-based classifier.
- **Hyperparameter Tuning**: Optimal `k` determined using cross-validation.
- **Comparison**: Evaluated performance against `KNeighborsClassifier` and `DecisionTreeClassifier` from `sklearn`.

## Evaluation

- **Metrics**: Accuracy, confusion matrix.
- **Results**:
  - **Iris Dataset**: 95.56% accuracy with `k=3`.
  - **Wine Quality Dataset**: 59.79% accuracy with `k=14`.
- **Tree-Based Comparison**: Decision Tree achieves similar accuracy for the Iris dataset and slightly better results for the Wine dataset.

## Optimization

- **Grid Search**: Determined optimal `k` and Decision Tree hyperparameters (`max_depth` and `criterion`).
