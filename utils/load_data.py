"""Load and parse the datasets"""

import sys
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Function that receive Dataset in csv format,
    and return it as Pandas DataFrame if none error occured."""

    try:
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not path.lower().endswith((".csv")):
            raise ValueError("file extension must be .csv.")

        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("csv file is empty.")

        return df

    except (FileNotFoundError, PermissionError,
            IsADirectoryError) as error:
        print(f"Error: cannot read '{error.filename}': "
              f"{(error.strerror or str(error)).lower()}.", file=sys.stderr)
        sys.exit(1)

    except pd.errors.EmptyDataError:
        print("Error: csv file is empty.", file=sys.stderr)
        sys.exit(1)

    except (TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def check_columns(df: pd.DataFrame, columns: list) -> None:
    """Exit if one of `columns` is missing from df."""

    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"Error: missing column(s) in the dataset: "
              f"{', '.join(missing)}.", file=sys.stderr)
        sys.exit(1)
