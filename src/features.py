"""Give both regression models the same numerical representation of the inputs.

Only the three historical counts, time of day and weekday are used. The
conversion does not read future targets or learn statistics from other rows.
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
    """Represent counts, clock time and weekday without changing row order.

    Args:
        data (pandas.DataFrame): One row per forecast with headcount_now,
            headcount_lag15, headcount_lag30, time_of_day and day_of_week.
            Counts are finite whole numbers at least zero, measured in people.
            Time is Melbourne clock time in decimal hours, from 0 up to but
            not including 24. Weekday labels run from Monday 0 to Sunday 6.
            Other columns, including future targets, are ignored.

    Returns:
        pandas.DataFrame: Eleven float columns in a fixed order: three counts,
            two clock coordinates and six weekday indicators. Monday has no
            indicator and acts as the reference day. The input index is kept.

    Raises:
        ValueError: The input is not a DataFrame, is empty, lacks required
            columns, or has missing, infinite or out of range values.
        TypeError: An input object cannot be converted to numbers.

    Notes:
        Sine and cosine keep times just before and after midnight close together.
        Weekday indicators let each day have a separate effect instead of treating
        its label as an amount. Nothing is fitted during this conversion.
        For linear regression, one sine and cosine pair allows one smooth daily
        cycle. It does not describe every possible pattern of room use.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    missing = [column for column in INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if len(data) == 0:
        raise ValueError("At least one forecast row is required.")

    # Read only the selected inputs so target and audit columns cannot enter the model.
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

    # Treat midnight as a continuation of the same clock, not a large jump in time.
    angle = 2 * np.pi * hour / 24
    features["time_sin"] = np.sin(angle)
    features["time_cos"] = np.cos(angle)


    # Use Monday as the reference so a separate indicator is not needed for every day.
    for day_number in range(1, 7):
        column = "is_" + WEEKDAY_NAMES[day_number]
        features[column] = (weekday == day_number).astype(float)

    # Keep the same column order whenever training or making predictions.
    return features[ENCODED_COLUMNS]
