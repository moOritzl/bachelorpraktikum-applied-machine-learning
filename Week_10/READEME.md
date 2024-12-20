# K-Means Clustering

## Overview

In [this notebook](./K-Means_Clustering.ipynb), we implement the K-Means clustering algorithm from scratch, apply it to the iris dataset, and cluster the 20 Newsgroups dataset with text preprocessing and dimensionality reduction. The notebook focuses on the Elbow Method, performance evaluation, and comparison with Sklearn's implementation.

## Datasets

- **iris.scale**: Iris dataset for clustering.
- **20 Newsgroups**: Text data categorized into 20 different topics.

## Preprocessing

- Text vectorization using `TfidfVectorizer`.
- Dimensionality reduction with `TruncatedSVD`.
- Data conversion to LIBSVM format.

## Models

- Custom K-Means Clustering implementation.
- Sklearn’s `KMeans` for performance comparison.

## Evaluation

Metrics: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI).
