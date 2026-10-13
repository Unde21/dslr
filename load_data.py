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
            raise ValueError("file extension must be .csv")

        df = pd.read_csv(path)
        df_clean = df.dropna()
        return df_clean

    except (TypeError, ValueError, FileNotFoundError,
            PermissionError, IsADirectoryError) as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
