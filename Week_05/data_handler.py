import pandas as pd
from typing import Tuple
from ucimlrepo import fetch_ucirepo


def split_data(X, y, frac, random_state=69):
    """
    Splits data into train and test sets.

    :param X: Independent variables.
    :param y: Dependent variables.
    :param frac: Fraction of data used for training.
    :param random_state: Random seed for splitting the data.
    :return: X_train, X_test, y_train, y_test as Numpy arrays.
    """
    X_train = X.sample(frac=frac, axis=0, random_state=random_state)
    X_test = X.drop(X_train.index)
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    return X_train.values, X_test.values, y_train.values.flatten(), y_test.values.flatten()
