"""Building blocks of the one-vs-all logistic regression model."""

import numpy as np


def sigmoid(score) -> np.ndarray:
    """Convert a score into a probability between 0 and 1.
    Work on a single number or element-wise on a numpy array."""

    clipped_score = np.clip(score, -500, 500)
    sig = 1 / (1 + np.exp(-clipped_score))
    return sig


def add_bias_column(array) -> np.ndarray:
    """Add a column of 1 in first position, so the bias theta0 can be
    computed in the same product X @ theta as the other weights.
    Accept a DataFrame or a 2D array of shape (n, features),
    return a numpy array of shape (n, features + 1)."""

    X = np.asarray(array)
    bias = np.ones(X.shape[0])
    X = np.column_stack((bias, X))
    return X


def predict_proba(X, theta) -> np.ndarray:
    """Compute the probability of belonging to the class for each
    student: sigmoid(X @ theta).
    X must contain the bias column (see add_bias_column).
    Return one probability per student, shape (n,)."""

    z = X @ theta
    h = sigmoid(z)
    return h


def ft_vec_mean(arr: np.ndarray) -> float:
    """Return the mean of a numpy array, vectorized.
    Unlike ft_mean (utils/stats.py), it does not loop in Python:
    used in the model computations, which run many times."""

    return np.sum(arr) / len(arr)


def calc_cost(h: np.ndarray, y: np.ndarray) -> float:
    """Compute the log loss J between the predictions h and the
    true labels y (0 or 1): -mean(y*log(h) + (1-y)*log(1-h)).
    h is clipped to avoid log(0).
    Return 0 for perfect predictions, about 0.693 when h = 0.5."""

    clipped_h = np.clip(h,  1e-15, 1 - (1e-15))
    all_cost = y * np.log(clipped_h) + (1 - y) * np.log(1 - clipped_h)
    mean = ft_vec_mean(all_cost)
    J = - mean

    return J


def calc_gradient(X: np.ndarray, h: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the gradient of the log loss for each theta:
    mean over the students of (h - y) * x_j, i.e. X.T @ (h - y) / m.
    X must contain the bias column, so theta0 is handled like the others.
    Return one value per theta, shape (features + 1,)."""

    error = h - y
    gradient = (X.T @ error) / y.shape[0]

    return gradient
