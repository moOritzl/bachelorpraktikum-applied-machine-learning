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
    :return: X_train, X_test, y_train, y_test.
    """
    Xtrain = X.sample(frac=frac, axis=0, random_state=random_state)
    Xtest = X.drop(Xtrain.index)
    ytrain = y.loc[Xtrain.index]
    ytest = y.loc[Xtest.index]
    return Xtrain, Xtest, ytrain, ytest
