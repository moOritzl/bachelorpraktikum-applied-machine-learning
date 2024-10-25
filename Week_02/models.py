import numpy as np
from math import sqrt


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


class LinRegNormEq:
    def __init__(self):
        """
        Linear Regression model using the Normal Equation.
        """
        self.B = None

    def fit(self, X, y, algorithm='gauss'):
        """
        Fit the linear regression model to the given data.
        :param X: input feature matrix
        :param y: target values
        :param algorithm: 'gauss', 'cholesky', 'qr' (algorithm to solve the normal equation)
        :return: None
        """
        ALGORITHMS = ['gauss', 'cholesky', 'qr']
        if algorithm not in ALGORITHMS:
            raise ValueError(f'Algorithm must be one of {ALGORITHMS}')

        X = X.to_numpy()
        y = y.to_numpy().reshape(-1, 1)

        X = np.hstack([np.ones((X.shape[0], 1)), X])

        A = X.T @ X
        b = X.T @ y

        # we need to avoid zeros on the main diagonal for gauss and make sure the Matrix is positive definite for cholesky, so we regulate it (add λI do make it more stable)
        A += 1e-8 * np.eye(A.shape[0])

        if algorithm == 'gauss':
            self.B = self._gauss(A, b)
        elif algorithm == 'cholesky':
            self.B = self._cholesky(A, b)
        else:
            self.B = self._qr(A, b)

        # print(f'Shape of B after {algorithm}: {self.B.shape}')  # for debugging reasons

    def predict(self, X):
        """
        Predict using the linear regression model.
        :param X: input feature matrix to predict values for
        :return: predicted values
        """
        if self.B is None:
            raise ValueError("The model has not been trained yet. Please call fit() first.")

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.B

    @staticmethod
    def _gauss(A, b):
        """
        Gaussian Elimination algorithm to solve systems of linear equations.
        The implementation is largely inspired by the YouTube video: https://youtu.be/gAmMxdI0EKs?si=wrCU1qpLQTnXmsHb
        :param A: coefficient matrix
        :param b: constant vector
        :return: solution vector
        """
        n = len(b)
        m = n - 1
        i = 0
        x = np.zeros(n)

        b = b.reshape(-1, 1)

        augmented_matrix = np.concatenate((A, b), axis=1)

        while i < n:
            if augmented_matrix[i][i] == 0.0:
                raise ZeroDivisionError("Zero on main Diagonal")

            for j in range(i + 1, n):
                scaling_factor = augmented_matrix[j][i] / augmented_matrix[i][i]
                augmented_matrix[j] = augmented_matrix[j] - (scaling_factor * augmented_matrix[i])
            i += 1

        x[m] = augmented_matrix[m][n] / augmented_matrix[m][m]

        for k in range(n - 2, -1, -1):
            x[k] = augmented_matrix[k][n]

            for j in range(k + 1, n):
                x[k] = x[k] - augmented_matrix[k][j] * x[j]
            x[k] = x[k] / augmented_matrix[k][k]

        return x.reshape(-1, 1)

    @staticmethod
    def _is_symmetric(matrix, tol=1e-10):
        return np.allclose(matrix, matrix.T, atol=tol)

    @staticmethod
    def _cholesky(A, b):
        """
        Cholesky decomposition algorithm to solve systems of linear equations.
        We can use this, because the matrix A is always symmetric (since it is X.T @ X).
        :param A: coefficient matrix
        :param b: constant vector
        :return: solution vector
        """

        n = A.shape[0]
        L = np.zeros_like(A)

        # inspired from https://github.com/TayssirDo/Cholesky-decomposition, but adjusted to calculate the lower triangular matrix
        for i in range(n):
            L[i, i] = sqrt(A[i, i])
            for j in range(i + 1, n):
                L[j, i] = A[j, i] / L[i, i]
                A[j, j:] = A[j, j:] - L[j, i] * L[i, j:]

        # Solve L * y = b
        y = np.linalg.solve(L, b)

        # Solve L.T * B = y
        B = np.linalg.solve(L.T, y)

        return B

    @staticmethod
    def _qr(A, b):
        """
        QR decomposition algorithm to solve systems of linear equations.
        The implementation is largely inspired by: https://youtu.be/kpk6x2Z6Nfs?si=t5iBnkaD5Pt5qmPQ
        :param A: coefficient matrix
        :param b: constant vector
        :return: solution vector
        """

        m, n = A.shape

        Q = np.zeros((m, n))

        for i, column in enumerate(A.T):
            Q[:, i] = column

            for prev in Q.T[:i]:
                Q[:, i] -= (prev @ column) / (prev @ prev) * prev

        Q /= np.linalg.norm(Q, axis=0)
        R = Q.T @ A

        # Q, R = np.linalg.qr(A)  # for testing

        # Now that we have Q and R, solve R * B = Q.T * b
        y = Q.T @ b
        B = np.linalg.solve(R, y)

        return B
