"""Normalization shared by logreg_train and logreg_predict."""

import pandas as pd


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
