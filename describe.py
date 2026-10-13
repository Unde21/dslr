"""Reproduice the function .describe()"""

import pandas as pd


def ft_count(values) -> float:
    """Count the number total of valid value to the col,
    and return it."""
    return float(sum(1 for val in values if not pd.isna(val)))


def ft_mean(values, count) -> float:
    """Calcul the mean of valid value to the col,
    and return it."""
    if count == 0:
        return float('nan')
    total_sum = sum(val for val in values if not pd.isna(val))
    return total_sum / count


def ft_std(values, mean, count) -> float:
    """Calcul the standard deviation to the col,
    and return it."""
    # clean_values = [x for x in values if x == x and x is not None]
    if count <= 1:
        return float('nan')

    variance_sum = sum(((x - mean) ** 2) for x in values)
    variance = variance_sum / (count - 1)
    return variance ** 0.5


def ft_min(values) -> float:
    """Get the min value of the col and return it."""
    if not values:
        return float('nan')

    min = values[0]
    for val in values:
        if val <= min:
            min = val
    return float(min)


def ft_max(values) -> float:
    """Get the max value of the col and return it."""
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
    and return it."""
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
    """Function that reproduce the .describe()"""
    numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            pass

    data_stats = {}
    stats_name = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    for col in numeric_cols:
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
        data_stats[col] = [count, mean, std, min, p25, p50, p75, max]

    result_df = pd.DataFrame(data_stats, index=stats_name)
    return result_df
