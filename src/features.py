"""Fixed input encoding for the occupancy regression models.

Only the five agreed inputs are read. No targets, dates from the future,
training averages or validation statistics are used to construct features.
"""

import numpy as np
import pandas as pd

INPUT_COLUMNS = [
    "headcount_now", "headcount_lag15", "headcount_lag30",
    "time_of_day", "day_of_week"
]
COUNT_COLUMNS = INPUT_COLUMNS[:3]
WEEKDAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
]
ENCODED_COLUMNS = COUNT_COLUMNS + ["time_sin", "time_cos"] + [
    "is_" + day for day in WEEKDAY_NAMES[1:]
]


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Represent the five agreed inputs as 11 numerical regression columns.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per forecast. Required columns are the three headcounts,
        ``time_of_day`` in Melbourne decimal hours (0 <= hour < 24), and
        ``day_of_week`` (Monday=0 through Sunday=6). Counts must be finite,
        non-negative whole numbers. Extra columns, including targets, are ignored.

    Returns
    -------
    pandas.DataFrame
        Three unchanged count columns, sine/cosine of the 24-hour clock, and
        six weekday flags (Tuesday to Sunday). Monday is the reference day.
        Row order and index are preserved. All returned values are floats.

    Raises
    ------
    ValueError
        If required inputs are missing, empty, non-numeric, non-finite, or
        outside their stated domain. Invalid examples are not silently dropped.

    Notes
    -----
    This conversion is fixed: it learns no parameters and has no fit stage.
    The same conversion must be used for training and subsequent forecasts.
    One daily sine/cosine pair is a starting assumption, not a proven best
    representation. It permits only a smooth single-cycle daily contribution
    in an additive linear model. The weekday flags allow separate day effects.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    missing = [column for column in INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if len(data) == 0:
        raise ValueError("At least one forecast row is required.")

    values = data[INPUT_COLUMNS].astype(float).copy()
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Inputs must be finite; missing values are not allowed.")
    counts = values[COUNT_COLUMNS]
    if (counts < 0).any().any() or (counts % 1 != 0).any().any():
        raise ValueError("Headcounts must be non-negative whole numbers.")
    hour = values["time_of_day"]
    if ((hour < 0) | (hour >= 24)).any():
        raise ValueError("time_of_day must be at least 0 and less than 24.")
    weekday = values["day_of_week"]
    if ((weekday < 0) | (weekday > 6) | (weekday % 1 != 0)).any():
        raise ValueError("day_of_week must be an integer from 0 to 6.")

    features = counts.copy()
    # A full day makes one full turn around a circle; midnight has no artificial jump.
    angle = 2 * np.pi * hour / 24
    features["time_sin"] = np.sin(angle)
    features["time_cos"] = np.cos(angle)

    # Monday needs no flag: all six flags are zero on Monday.
    for day_number in range(1, 7):
        column = "is_" + WEEKDAY_NAMES[day_number]
        features[column] = (weekday == day_number).astype(float)

    return features[ENCODED_COLUMNS]
