import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        """
        Simple Linear Regression model from the first exercise.
        """
        self.b_0 = None
        self.b_1 = None

    def fit(self, X, y):
        """
        Fit the simple linear regression model to the given data.
        """
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.b_1 = numerator / denominator
        self.b_0 = y_mean - self.b_1 * x_mean

    def predict(self, X):
        """
        Predict using the simple linear regression model.
        """
        if self.b_0 is None or self.b_1 is None:
            raise ValueError("The model has not been trained yet. Please call fit() first.")
        return self.b_0 + self.b_1 * X
