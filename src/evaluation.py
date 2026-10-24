"""Persistence forecasts and error calculations shared by the model notebooks.

Using the same calculations keeps the comparison consistent. Observations and
predictions must describe the same examples in the same order. These functions
do not load data or decide which training, validation or test period to use.
"""

import numpy as np
from numpy.typing import ArrayLike


def _as_vector(values: ArrayLike, name: str) -> np.ndarray:
    """Check that a set of values is suitable for the error calculations.

    Args:
        values (ArrayLike): A nonempty sequence of numbers with one dimension.
            Missing and infinite values are not accepted.
        name (str): Input name to include in an error message.

    Returns:
        numpy.ndarray: The values as a float array in their original order.

    Raises:
        ValueError: The values are empty, have the wrong shape, contain missing
            or infinite values, or cannot be converted from numeric text.
        TypeError: An input object cannot be converted to numbers.
    """
    array = np.asarray(values, dtype=float)
    # Reject nested inputs so each position represents one forecast.
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    # A missing value must not silently affect the reported average error.
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain missing or infinite values.")
    return array


def persistence_predict(current_counts: ArrayLike) -> np.ndarray:
    """Use the current headcount as the forecast for the future target.

    This provides a simple reference method with no training stage.

    Args:
        current_counts (ArrayLike): One current count per forecast, in people.
            The sequence must contain finite numbers at least zero. Observation
            times and the five minute rule are checked during data preparation.

    Returns:
        numpy.ndarray: A separate copy of the counts in the same order.
            Each count is treated as the corresponding future prediction.

    Raises:
        ValueError: Counts are empty, negative, missing, infinite, have the
            wrong shape, or cannot be converted from numeric text.
        TypeError: An input object cannot be converted to numbers.
    """
    counts = _as_vector(current_counts, "current_counts")
    if (counts < 0).any():
        raise ValueError("current_counts must be non-negative.")

    # Keep predictions separate so editing them cannot change the input counts.
    return counts.copy()


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Measure how far forecasts are from the corresponding recorded counts.

    Args:
        y_true (ArrayLike): Recorded target counts in people, one per example.
        y_pred (ArrayLike): Predictions for those same examples in the same
            order. Decimal and negative predictions are kept as supplied.

    Returns:
        dict[str, float]: The keys mae and rmse, both measured in people.
            MAE is the average size of the error. RMSE gives larger errors more
            influence. Zero means no error on these examples, not a percentage.

    Raises:
        ValueError: Either sequence is empty, contains missing or infinite
            values, has the wrong shape, cannot be converted from numeric text,
            or has a different length from the other sequence.
        TypeError: An input object cannot be converted to numbers.

    Notes:
        Values are paired by position, not by pandas index labels. Rows are not
        sorted or removed here. Residuals are recorded count minus prediction.
        Predictions are not rounded or set to zero before scoring.
    """
    observed = _as_vector(y_true, "y_true")
    predicted = _as_vector(y_pred, "y_pred")
    # Each prediction needs exactly one corresponding recorded count.
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same number of values.")

    # Keep raw errors visible rather than hiding negative or decimal predictions.
    residuals = observed - predicted
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}
