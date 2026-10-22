"""Building blocks of the one-vs-all logistic regression model."""

import numpy as np


def sigmoid(score) -> np.ndarray:
    """Convert a score into a probability between 0 and 1.
    Work on a single number or element-wise on a numpy array."""

    clipped_score = np.clip(score, -500, 500)
    sig = 1 / (1 + np.exp(-clipped_score))
    return sig
