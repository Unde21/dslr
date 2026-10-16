"""Script that display an scatter plot of dataset given at run time.
The data must be a .csv file."""

import pandas as pd
import sys
import matplotlib.pyplot as plt
from load_data import load_data


def disp_scalter_plot(df: pd.DataFrame):
    """Create and display scalter plot, to answer the question:
    What are the two features that are similar?"""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if 'Index' in numeric_df:
        numeric_df = numeric_df.drop('Index', axis=1)
    corr_matrix = numeric_df.corr()
    print(corr_matrix)


def main():
    """Run the script and get the file as parameter."""

    ac = len(sys.argv)

    try:
        assert ac == 2, "the program must take one .csv file as argument."
    except AssertionError as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)

    df = load_data(sys.argv[1])
    disp_scalter_plot(df)
    # print(df.corr())


if __name__ == "__main__":
    main()
