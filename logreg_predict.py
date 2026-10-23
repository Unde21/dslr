"""Predict the Hogwarts House of each student with the model trained
by logreg_train.py.

Usage: python3 logreg_predict.py dataset_test.csv weights.json"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from utils.load_data import load_data
from utils.normalize import normalize
from utils.logreg import (add_bias_column,
                          predict_probas, predict_houses)


def load_weights(filepath: str) -> dict:
    """Load the model saved by logreg_train.py from a JSON file.
    Return a dict with the keys "features", "norm_params" and "thetas".
    Exit if the file is missing, unreadable, malformed
    or does not contain these keys."""

    path = Path(filepath)
    if not path.exists():
        print(f"Error: '{path}' not found.",
              file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"'{path}' invalid format.")
            required_keys = {"features", "norm_params", "thetas"}
            if not required_keys.issubset(data.keys()):
                raise ValueError(
                    f"'{path}' file is missing required parameters."
                )
            return data

    except json.JSONDecodeError:
        print(f"Error: '{path}' is corrupted or malformed.", file=sys.stderr)
        sys.exit(1)

    except (PermissionError, FileNotFoundError,
            IsADirectoryError) as error:
        print(f"Error: cannot read '{error.filename}': "
              f"{(error.strerror or str(error)).lower()}.", file=sys.stderr)
        sys.exit(1)

    except (TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def prepare_features(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Select the features used for the training, then fill NaN and
    normalize them with the norm_params saved in weights
    (the training mean and std, not recomputed on df).
    Return the normalized features."""

    feature_df = df[weights["features"]]
    x_norm = normalize(feature_df, weights["norm_params"])
    return x_norm


def save_predictions(df: pd.Series, predicted_houses: list) -> None:
    """Write houses.csv in the format of the subject:
    'Index,Hogwarts House', one line per student.
    df is the 'Index' column of the dataset, in the same order
    as predicted_houses. Exit if the file can not be written."""

    index = np.asarray(df)
    new_df = pd.DataFrame({
        "Index": index,
        "Hogwarts House": predicted_houses
    })

    try:
        new_df.to_csv("houses.csv", index=False)
    except (PermissionError, IsADirectoryError, OSError) as error:
        print(f"Error: cannot write '{error.filename}': "
              f"{(error.strerror or str(error)).lower()}.",
              file=sys.stderr)
        sys.exit(1)


def main():
    """Main function that run the prediction."""

    ac = len(sys.argv)
    if ac != 3:
        print("Error: the program must take one .csv and"
              " 'weights.json' files as arguments.",
              file=sys.stderr)
        sys.exit(1)

    df = load_data(sys.argv[1])
    model = load_weights(sys.argv[2])

    x_norm = prepare_features(df, model)
    X = add_bias_column(x_norm)

    probas_matrix, houses_name = predict_probas(X, model)
    predicted_houses = predict_houses(probas_matrix, houses_name)
    save_predictions(df["Index"], predicted_houses)


if __name__ == "__main__":
    main()
