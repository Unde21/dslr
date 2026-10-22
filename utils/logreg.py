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
