"""Train a one-vs-all logistic regression to predict the Hogwarts House.

Usage: python3 logreg_train.py dataset_train.csv"""

import sys
import pandas as pd
import numpy as np
import json
from utils.load_data import load_data
from utils.stats import ft_mean
from utils.stats import ft_std
from utils.stats import ft_count
from utils.logreg import add_bias_column
from utils.logreg import predict_proba
from utils.logreg import calc_cost
from utils.logreg import calc_gradient
# import utils as u

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

LEARNING_RATE = 0.5
N_ITERATIONS = 2000


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


def normalize(df: pd.DataFrame, norm_params: dict) -> pd.DataFrame:
    """Replace NaN by the column mean, then apply the z-score
    (x - mean) / std to each column, using the values of norm_params.
    df is not modified: return a normalized copy."""

    df_norm = df.copy()

    for col in df_norm:
        df_norm[col] = df_norm[col].fillna(norm_params[col]["mean"])
        df_norm[col] = (df_norm[col] - norm_params[col]["mean"])
        df_norm[col] /= norm_params[col]["std"]

    return df_norm


def binary_labels(houses: pd.Series, house: str) -> pd.Series:
    """Build the one-vs-all labels for `house`.
    Return one value per student, in the same order as `houses`:
    1 if the student belongs to `house`, 0 otherwise."""

    labels = houses == house
    labels = labels.astype(int)
    return labels


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select the FEATURE columns, then fill NaN and normalize them.
    Return the normalized features and the
    {feature: {"mean": mean, "std": std}} used, to be saved
    for the prediction."""

    feature_df = df[FEATURE]
    norm_params = calc_mean_std(feature_df)
    x_norm = normalize(feature_df, norm_params)
    return x_norm, norm_params


def get_houses(df: pd.DataFrame) -> tuple[list, pd.Series]:
    """Get the houses from the 'Hogwarts House' column.
    Return the sorted list of house names and the column itself
    (one true house per student).
    Exit if the column is missing or contains no house."""

    try:
        houses_names = sorted(df["Hogwarts House"].dropna().unique())
        true_houses = df["Hogwarts House"]
    except KeyError:
        print("Error: column 'Hogwarts House'"
              " not found in the dataset.", file=sys.stderr)
        sys.exit(1)

    if not houses_names:
        print("Error: column 'Hogwarts House' contains no house."
              " Use the training dataset.", file=sys.stderr)
        sys.exit(1)

    return houses_names, true_houses


def train_classifier(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Train one binary classifier by gradient descent.
    Start from theta = 0, then repeat N_ITERATIONS times:
    predict h, compute the gradient, theta -= learning_rate * gradient.
    Print the cost every 200 iterations to follow the training.
    Return the trained theta, shape (features + 1,)."""

    theta = np.zeros(X.shape[1])

    for i in range(N_ITERATIONS):
        h = predict_proba(X, theta)
        gradient = calc_gradient(X, h, y)
        theta = theta - LEARNING_RATE * gradient
        if i % 400 == 0:
            cost = calc_cost(h, y)
            print(cost)

    return theta


def train_one_vs_all(X: np.ndarray, houses_name: list,
                     true_houses: pd.Series) -> dict:
    """Train one binary classifier per house (one-vs-all):
    for each house, build its 0/1 labels from true_houses,
    then train a theta on X.
    Return a dict {house: theta}, theta as a numpy array."""

    # labels_per_house = []
    theta_houses = {}
    for house in houses_name:
        labels = binary_labels(true_houses, house)

        y = np.asarray(labels)

        print("=" * 45, f"\n{house}")
        theta_houses[house] = train_classifier(X, y)

    return theta_houses


def save_data(norm_params: dict, theta_houses: dict) -> None:
    """Save the model to weights.json: the ordered FEATURE list,
    the normalization params and the theta of each house
    (converted to lists, as numpy arrays are not JSON serializable).
    Exit if the file can not be written."""

    thetas = {name: arr.tolist() for name, arr in theta_houses.items()}
    data = {
        "features": FEATURE,
        "norm_params": norm_params,
        "thetas": thetas
    }

    try:
        with open("weights.json", "w") as f:
            json.dump(data, f, indent=4)

    except (PermissionError, OSError, IsADirectoryError) as error:
        print(f"Error: failed to save data: {error}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function that run the logistic regression training."""

    ac = len(sys.argv)
    if ac != 2:
        print("Error: the program must take one .csv file as argument.",
              file=sys.stderr)
        sys.exit(1)

    df = load_data(sys.argv[1])

    houses_name, true_houses = get_houses(df)

    x_norm, norm_params = prepare_features(df)
    # print(f"{x_norm.describe()}")
    X = add_bias_column(x_norm)
    # print(X)
    theta_houses = train_one_vs_all(X, houses_name, true_houses)
    # print("=" * 45, "\ntheta for each houses")
    # print(theta_houses)
    save_data(norm_params, theta_houses)


if __name__ == "__main__":
    main()
