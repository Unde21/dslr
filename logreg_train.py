"""Train a one-vs-all logistic regression to predict the Hogwarts House.

Usage: python3 logreg_train.py dataset_train.csv"""

import sys
import pandas as pd
import numpy as np
from utils.load_data import load_data
from utils.stats import ft_mean
from utils.stats import ft_std
from utils.stats import ft_count
from utils.logreg import add_bias_column
from utils.logreg import predict_proba
from utils.logreg import calc_cost


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


def calc_mean_std(df: pd.DataFrame) -> dict:
    """Calculate the mean and standard deviation of each column of df,
    ignoring NaN.
    Return a dict {column: (mean, std)}."""

    norm_params = {}

    for col in df.columns:
        values = pd.to_numeric(df[col], errors='coerce').dropna().tolist()

        if not values:
            continue

        count = ft_count(values)
        mean = ft_mean(values, count)
        std = ft_std(values, mean, count)
        norm_params[col] = (mean, std)

    return norm_params


def normalize(df: pd.DataFrame, norm_params: dict) -> pd.DataFrame:
    """Replace NaN by the column mean, then apply the z-score
    (x - mean) / std to each column, using the values of norm_params.
    df is not modified: return a normalized copy."""

    df_norm = df.copy()

    for col in df_norm:
        df_norm[col] = df_norm[col].fillna(norm_params[col][0])
        df_norm[col] = (df_norm[col] - norm_params[col][0])
        df_norm[col] /= norm_params[col][1]

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
    Return the normalized features and the {feature: (mean, std)}
    used, to be saved for the prediction."""

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


def main():
    """Main function that run the logistic regression training."""

    ac = len(sys.argv)
    if ac != 2:
        print("Error: the program must take one .csv file as argument.",
              file=sys.stderr)
        sys.exit(1)

    df = load_data(sys.argv[1])

    houses_names, true_houses = get_houses(df)

    x_norm, norm_params = prepare_features(df)
    # print(f"{x_norm.describe()}")

    labels_per_house = []
    for house in houses_names:
        labels = binary_labels(true_houses, house)
        labels_per_house.append(labels)
        print(sum(labels))

    y = np.asarray(labels)
    print(y)
    X = add_bias_column(x_norm)
    print(X)

    theta = np.zeros(X.shape[1])
    h = predict_proba(X, theta)
    print(h)

    calc_cost(h, y)


if __name__ == "__main__":
    main()
