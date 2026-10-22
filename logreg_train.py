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


def calc_mean_std(df: pd.DataFrame, ) -> dict:
    """Calculate the mean and standard deviation of each column of df,
    ignoring NaN.
    Return a dict {column: (mean, std)}."""

    stats_dict = {}

    for col in df.columns:
        values = pd.to_numeric(df[col], errors='coerce').dropna().tolist()

        if not values:
            continue

        count = ft_count(values)
        mean = ft_mean(values, count)
        std = ft_std(values, mean, count)
        stats_dict[col] = (mean, std)

    return stats_dict


def normalize(df: pd.DataFrame, stats_dict: dict) -> pd.DataFrame:
    """Replace NaN by the column mean, then apply the z-score
    (x - mean) / std to each column, using the values of stats_dict.
    df is not modified: return a normalized copy."""

    df_norm = df.copy()

    for col in df_norm:
        df_norm[col] = df_norm[col].fillna(stats_dict[col][0])
        df_norm[col] = (df_norm[col] - stats_dict[col][0]) / stats_dict[col][1]

    return df_norm


def main():
    """Main function that run the logistic regression training."""

    ac = len(sys.argv)
    if ac != 2:
        print("the program must take one .csv file as argument.")
        sys.exit(1)

    df = load_data(sys.argv[1])

    clean_df = df[FEATURE]
    stats_dict = calc_mean_std(clean_df)
    df_norm = normalize(clean_df, stats_dict)
    print(f"{df_norm.describe()}")


if __name__ == "__main__":
    main()
