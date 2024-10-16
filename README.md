# Bachelorpraktikum: Applied Machine Learning

This repository contains my work for the **Bachelorpraktikum: Applied Machine Learning**. The practical course focuses on implementing key machine learning algorithms and concepts through a series of labs. The labs cover various data analysis tasks, model training, and performance evaluation using Python and its essential libraries like Pandas, Numpy, and Scikit-learn.

## Course Overview

The practical course consists of the following topics:
1. **Introduction to Machine Learning and Python Libraries**
   - Overview of machine learning concepts
   - Introduction to Python libraries: NumPy, Pandas, Scikit-learn, Matplotlib
2. **Linear Regression**
   - Simple and multiple linear regression
   - Model evaluation metrics
3. **Classification**
   - Logistic regression, K-Nearest Neighbors (KNN), Naive Bayes, Support Vector Machines (SVM)
   - Model evaluation and metrics for classification
4. **Decision Trees and Ensemble Methods**
   - Decision trees, Random forests, Ensemble learning techniques
5. **K-Nearest Neighbors (KNN)**
   - Distance metrics, Choosing optimal K value, Applications in classification and regression
6. **Naive Bayes**
   - Gaussian, Multinomial, and Bernoulli classifiers, Text classification
7. **K-Means Clustering**
   - Clustering fundamentals, Evaluation metrics
8. **Principal Component Analysis (PCA)**
   - Dimensionality reduction, Visualization of principal components
9. **Autoencoders**
   - Unsupervised learning, Anomaly detection
10. **Capstone Project**
    - Integration of techniques, Real-world dataset application

## Week 01: Data IO, Numpy, and Linear Regression

### Task 1: Word Count and Matrix Operations with Python Built-ins, Pandas, and Numpy
- Implemented a word counting program using Python built-in functions that reads a text file and counts the occurrences of unique words. After filtering out common words such as 'the', 'a', 'an', and 'be', a bar chart was created to display the top 10 most frequent words. Additionally, a `Pandas` DataFrame was constructed from the filtered words to facilitate further analysis and visualization.
- For the second part, a matrix with random values was created using `Numpy`, and a vector was initialized from a normal distribution. Each row of the matrix was multiplied element-wise by the vector, and the results were accumulated in a new vector. The mean and standard deviation of the resulting vector were calculated and visualized using a histogram.

### Task 2: Simple Linear Regression
- Implemented linear regression from scratch by generating synthetic data sets and using the "Learn Simple Linear Regression" algorithm. The regression line was calculated and plotted, and the effects of different variances on the model were observed. Additionally, the model was implemented using `numpy.linalg.lstsq` and `sklearn.linear_model.LinearRegression` to compare results and discuss differences in handling inputs and model fitting.

You can find the code for this week in the [Week 01](./Week_01) folder.

---

Stay tuned for more updates!
