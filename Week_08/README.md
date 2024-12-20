# Recommender Systems with Matrix Factorization

## Overview

In [this notebook](./Recommender_Systems_Matrix_Factorization.ipynb), we analyze the MovieLens dataset, preprocess data, and implement matrix factorization using stochastic gradient descent (SGD) and NMF. The goal is to build a recommender system and optimize its performance.

## Datasets

- **u.data**: User ratings of movies (100,000 ratings by 943 users on 1682 movies).
- **u.item**: Movie metadata, including genres and release dates.
- **u.user**: User demographics (age, gender, occupation, zip code).

## Preprocessing

- Dropped irrelevant columns (`video_release_date`, `IMDb_URL`, `zip_code`).
- One-hot encoding for movie genres.
- Data split for training and testing using KFold cross-validation.

## Models

- **Matrix Factorization using SGD**: Implemented custom matrix factorization with adaptive learning rate and bias terms.
- **NMF (Non-negative Matrix Factorization)**: Applied using scikit-learn's `NMF` library.

## Evaluation

Metrics: RMSE (Root Mean Squared Error) evaluated through cross-validation.
