"""Houses of the Hogwarts students, shared by the training and the plots."""

import sys
import pandas as pd


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
