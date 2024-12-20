# Spam Filter SVM

## Overview

In [this notebook](./Spam_Filter_SVM.ipynb), Support Vector Machines (SVM) are used to implement spam filters for emails and SMS messages. The notebook covers data preprocessing, model training and evaluation, and comparisons with the K-Nearest Neighbors (KNN) model.

## Datasets

- **Spambase**: Email spam dataset from the UCI repository.
- **SMS Spam Collection**: A dataset of SMS messages classified as spam or ham.

## Preprocessing

- MinMax scaling for numerical data.
- Tfidf vectorization and Truncated SVD for text data.
- Data split: 70% training, 30% testing.

## Models and Methods

- **SVM**: Various kernels (linear, polynomial, RBF, sigmoid).
- **KNN**: Optimized neighbors and distance metrics.

## Evaluation

- SVM with RBF kernel (C=10): 98.45% accuracy.
- KNN (1 neighbor, Minkowski): 96.17% accuracy.
