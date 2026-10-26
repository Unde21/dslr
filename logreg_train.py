"""Train a one-vs-all logistic regression to predict the Hogwarts House.

Usage: python3 logreg_train.py dataset_train.csv [--batch SIZE]
    no --batch: batch gradient descent (all students per update)
    --batch 1:  stochastic gradient descent
    --batch n:  mini-batch gradient descent"""

import sys
import argparse
import pandas as pd
import numpy as np
import json
from utils.load_data import load_data
from utils.stats import (ft_mean, ft_std, ft_count)
from utils.logreg import (add_bias_column,
                          predict_proba, calc_cost, calc_gradient)
from utils.normalize import normalize


FEATURE = [
    "Defense Against the Dark Arts",
    "Flying",
    "Charms",
    "Transfiguration",
    "History of Magic",
    "Ancient Runes",
    "Divination",
    "Muggle Studies",
    "Herbology"
]

LEARNING_RATE = 0.01
MAX_EPOCHS = 10000
PRECISION = 1e-7
SEED = 42


def calc_mean_std(df: pd.DataFrame) -> dict:
    """Calculate the mean and standard deviation of each column of df,
    ignoring NaN.
    Return a dict {column: {"mean": mean, "std": std}}."""

    norm_params = {}

    for col in df.columns:
        values = pd.to_numeric(df[col], errors='coerce').dropna().tolist()

        if not values:
            continue

        count = ft_count(values)
        mean = ft_mean(values, count)
        std = ft_std(values, mean, count)
        norm_params[col] = {
            "mean": mean,
            "std": std
        }

    return norm_params


def binary_labels(houses: pd.Series, house: str) -> pd.Series:
    """Build the one-vs-all labels for `house`.
    Return one value per student, in the same order as `houses`:
    1 if the student belongs to `house`, 0 otherwise."""

    labels = houses == house
    labels = labels.astype(int)
    return labels


def fit_features(df: pd.DataFrame,
                 features=FEATURE) -> tuple[pd.DataFrame, dict]:
    """Select the features columns (FEATURE by default), then fill NaN
    and normalize them.
    Return the normalized features and the
    {feature: {"mean": mean, "std": std}} used, to be saved
    for the prediction."""

    feature_df = df[features]
    norm_params = calc_mean_std(feature_df)
    x_norm = normalize(feature_df, norm_params)
    return x_norm, norm_params


def get_houses(df: pd.DataFrame) -> tuple[list, pd.Series]:
    """Get the houses from the 'Hogwarts House' column.
    Return the sorted list of house names and the column itself
    (one true house per student).
    Exit if the column is missing or contains no house."""

    try:
        house_names = sorted(df["Hogwarts House"].dropna().unique())
        true_houses = df["Hogwarts House"]
    except KeyError:
        print("Error: column 'Hogwarts House'"
              " not found in the dataset.", file=sys.stderr)
        sys.exit(1)

    if not house_names:
        print("Error: column 'Hogwarts House' contains no house."
              " Use the training dataset.", file=sys.stderr)
        sys.exit(1)

    return house_names, true_houses


def train_classifier(X: np.ndarray, y: np.ndarray, batch_size: int,
                     max_epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE,
                     precision=PRECISION) -> np.ndarray:
    """Train one binary classifier by gradient descent.
    Start from theta = 0. At each epoch, the students are shuffled
    (the SEED makes the result reproducible), then split in batches
    of `batch_size` students. For each batch: predict h, compute the gradient
    on the batch, theta -= learning_rate * gradient.
    - batch_size = None (all students): batch gradient descent
    - batch_size = 1: stochastic gradient descent (SGD)
    - batch_size = n: mini-batch gradient descent
    At the end of each epoch, the cost is computed on all the students.
    Early stopping: stop when the cost changes by less than precision
    between two epochs, or after max_epochs epochs at most.
    The hyperparameters default to LEARNING_RATE, PRECISION
    and MAX_EPOCHS.
    Print the cost every 400 epochs to follow the training.
    Return the trained theta, shape (features + 1,)."""

    theta = np.zeros(X.shape[1])
    if batch_size is None:
        batch_size = X.shape[0]
    prev_cost = 0

    rng = np.random.default_rng(SEED)
    for epoch in range(max_epochs):
        order = rng.permutation(X.shape[0])
        for start in range(0, X.shape[0], batch_size):
            batch_indices = order[start: start + batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            h_batch = predict_proba(X_batch, theta)
            gradient = calc_gradient(X_batch, h_batch, y_batch)
            theta = theta - learning_rate * gradient
        h_full = predict_proba(X, theta)
        new_cost = calc_cost(h_full, y)
        delta_cost = abs(prev_cost - new_cost)
        prev_cost = new_cost
        if delta_cost < precision:
            print(f"converged at epoch {epoch},"
                  f" cost: {new_cost}.")
            break

        if epoch % 400 == 0:
            print(new_cost)

    else:
        print(f"reached max epochs, cost: {prev_cost}.")
    return theta


def train_one_vs_all(X: np.ndarray, house_names: list,
                     true_houses: pd.Series, batch_size: int,
                     max_epochs=MAX_EPOCHS,
                     learning_rate=LEARNING_RATE, precision=PRECISION) -> dict:
    """Train one binary classifier per house (one-vs-all):
    for each house, build its 0/1 labels from true_houses,
    then train a theta on X.
    batch_size, max_epochs, learning_rate and precision are passed
    to train_classifier (batch_size = None: batch gradient descent).
    Return a dict {house: theta}, theta as a numpy array."""

    theta_houses = {}
    for house in house_names:
        labels = binary_labels(true_houses, house)

        y = np.asarray(labels)

        print("=" * 64, f"\n{house}")
        theta_houses[house] = train_classifier(X, y, batch_size,
                                               max_epochs=max_epochs,
                                               learning_rate=learning_rate,
                                               precision=precision)

    return theta_houses


def save_data(norm_params: dict, theta_houses: dict, features=FEATURE) -> None:
    """Save the model to weights.json: the ordered features list
    used for the training (FEATURE by default),
    the normalization params and the theta of each house
    (converted to lists, as numpy arrays are not JSON serializable).
    Exit if the file can not be written."""

    thetas = {name: arr.tolist() for name, arr in theta_houses.items()}
    data = {
        "features": features,
        "norm_params": norm_params,
        "thetas": thetas
    }

    try:
        with open("weights.json", "w") as f:
            json.dump(data, f, indent=4)

    except (PermissionError, IsADirectoryError, OSError) as error:
        print(f"Error: cannot write '{error.filename}': "
              f"{(error.strerror or str(error)).lower()}.",
              file=sys.stderr)
        sys.exit(1)


def main():
    """Main function that run the logistic regression training."""

    parser = argparse.ArgumentParser(
        description="Train a one-vs-all logistic regression and save the"
                    " model to weights.json.",
        epilog="examples:\n"
               "  python3 logreg_train.py dataset_train.csv\n"
               "  python3 logreg_train.py dataset_train.csv --batch 1\n"
               "  python3 logreg_train.py dataset_train.csv --batch 32",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataset", help="path to the training .csv file.")
    parser.add_argument("--batch", type=int, default=None,
                        metavar="SIZE",
                        help="number of students per update of theta:\n"
                             "  (none) batch gradient descent, all students"
                             " (default)\n"
                             "  1      stochastic gradient descent (SGD)\n"
                             "  n      mini-batch gradient descent,"
                             " n students")
    args = parser.parse_args()

    df = load_data(args.dataset)

    house_names, true_houses = get_houses(df)

    x_norm, norm_params = fit_features(df)
    X = add_bias_column(x_norm)
    theta_houses = train_one_vs_all(X, house_names, true_houses, args.batch)
    save_data(norm_params, theta_houses)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nlogistic regression training canceled.")
        sys.exit(130)
