"""Main function that run the program that reproduice the
    methods .describe()"""

from utils.load_data import load_data
from utils.stats import ft_describe
import pandas as pd
import sys


def main():
    """Main function"""

    ac = len(sys.argv)
    if ac != 2:
        print("Error: the program must take one .csv file as argument.",
              file=sys.stderr)
        sys.exit(1)

    df = load_data(sys.argv[1])
    pd.set_option("display.max_columns", None)
    print(ft_describe(df))


if __name__ == "__main__":
    main()
