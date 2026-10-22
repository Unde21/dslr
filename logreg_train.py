"""Train a one-vs-all logistic regression to predict the Hogwarts House.

Usage: python3 logreg_train.py dataset_train.csv"""

import sys
import pandas as pd
from utils.load_data import load_data
from utils.stats import ft_mean
from utils.stats import ft_std
from utils.stats import ft_count

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


def main():
    """Main function that run the logistic regression training."""

    ac = len(sys.argv)
    if ac != 2:
        print("Error: the program must take one .csv file as argument.",
              file=sys.stderr)
        sys.exit(1)

    df = load_data(sys.argv[1])

    feature_df = df[FEATURE]
    norm_params = calc_mean_std(feature_df)
    x_norm = normalize(feature_df, norm_params)
    # print(f"{x_norm.describe()}")

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

    labels_per_house = []
    for house in houses_names:
        labels = binary_labels(true_houses, house)
        labels_per_house.append(labels)
        print(sum(labels))


if __name__ == "__main__":
    main()
