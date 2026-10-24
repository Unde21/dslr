"""Reproduce the function .describe(), with extra fields (bonus).

Every ft_* stat function expects `values` as a list WITHOUT NaN:
the caller must clean the data before calling them."""

import pandas as pd

STATS_NAME = [
    'count',
    'mean',
    'std',
    'min',
    '25%',
    '50%',
    '75%',
    'max',
    'missing',
    'range',
    'variance',
    'IQR'
]


def ft_count(values) -> float:
    """Count the number total of value to the col,
    and return it.
    `values` must be a list without NaN."""
    return float(sum(1 for val in values))


def ft_mean(values, count) -> float:
    """Calcul the mean of value to the col,
    and return it.
    `values` must be a list without NaN, `count` its length."""
    if count == 0:
        return float('nan')
    total_sum = sum(val for val in values)
    return total_sum / count


def ft_std(values, mean, count) -> float:
    """Calcul the standard deviation to the col,
    and return it.
    `values` must be a list without NaN, `count` its length."""
    if count <= 1:
        return float('nan')

    variance_sum = sum(((x - mean) ** 2) for x in values)
    variance = variance_sum / (count - 1)
    return variance ** 0.5


def ft_min(values) -> float:
    """Get the min value of the col and return it.
    `values` must be a list without NaN."""
    if not values:
        return float('nan')

    min = values[0]
    for val in values:
        if val <= min:
            min = val
    return float(min)


def ft_max(values) -> float:
    """Get the max value of the col and return it.
    `values` must be a list without NaN."""
    if not values:
        return float('nan')

    max = values[0]
    for val in values:
        if val >= max:
            max = val
    return float(max)


def ft_percentile(values, count, percent: float) -> float:
    """Get the percentile:
        - value of percent(%) of data is under the percent.
    and return it.
    `values` must be a list without NaN, `count` its length."""
    if not values:
        return float('nan')

    sorted_values = sorted(values)
    if count == 1:
        return float(sorted_values[0])

    pos = percent * (count - 1)
    i = int(pos)
    f = pos - i

    if i + 1 >= count:
        return float(sorted_values[i])

    return float(sorted_values[i] + f
                 * (sorted_values[i + 1] - sorted_values[i]))


def ft_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the .describe() of pandas on the numeric columns of df:
    count, mean, std, min, 25%, 50%, 75% and max, computed without NaN.
    Bonus fields, added after the mandatory ones:
    - missing: number of NaN (or non numeric values) in the column
    - range: max - min
    - variance: std ** 2 (sample variance, divided by n - 1)
    - IQR: 75% - 25%, spread of the middle half of the values
    Return a DataFrame with one column per feature, one row per field."""
    numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            pass

    data_stats = {}

    for col in numeric_cols:
        len_col = len(df[col])
        values = pd.to_numeric(df[col], errors='coerce').dropna().tolist()

        if not values:
            continue

        count = ft_count(values)
        mean = ft_mean(values, count)
        std = ft_std(values, mean, count)
        min = ft_min(values)
        p25 = ft_percentile(values, count, 0.25)
        p50 = ft_percentile(values, count, 0.50)
        p75 = ft_percentile(values, count, 0.75)
        max = ft_max(values)
        miss = len_col - count
        range = max - min
        var = std ** 2
        iqr = p75 - p25
        data_stats[col] = [count, mean, std, min, p25,
                           p50, p75, max, miss, range, var, iqr]

    result_df = pd.DataFrame(data_stats, index=STATS_NAME)
    return result_df
