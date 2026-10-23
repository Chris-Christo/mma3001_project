"""Persistence predictions and shared regression-error calculations.

All observations and predictions must be supplied in the same row order.
Functions do not load data, fit a model, or access the final test set.
The notebook controls which chronological partition is evaluated.
"""

import numpy as np
from numpy.typing import ArrayLike


def _as_vector(values: ArrayLike, name: str) -> np.ndarray:
    """Convert a non-empty, finite, one-dimensional numeric input to an array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain missing or infinite values.")
    return array


def persistence_predict(current_counts: ArrayLike) -> np.ndarray:
    """Predict future headcount as the latest accepted current headcount.

    Parameters
    ----------
    current_counts : array-like of shape (n_examples,)
        Non-negative counts in people, one per forecasting example. Data
        preparation is responsible for checking observation age and timing.

    Returns
    -------
    numpy.ndarray
        A copy of the input counts in the same order, interpreted as forecasts.
        This rule has no fitted parameters and does not use future targets.

    Raises
    ------
    ValueError
        If the input is empty, non-numeric, not one-dimensional, non-finite,
        or contains a negative count.
    """
    counts = _as_vector(current_counts, "current_counts")
    if (counts < 0).any():
        raise ValueError("current_counts must be non-negative.")
    # A separate copy prevents edits to predictions from changing the inputs.
    return counts.copy()


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Calculate unweighted MAE and RMSE for matching observations and forecasts.

    Parameters
    ----------
    y_true : array-like of shape (n_examples,)
        Sensor-reported target counts in people.
    y_pred : array-like of shape (n_examples,)
        Predictions for exactly the same examples, in exactly the same order.
        Fractional or negative raw regression predictions are evaluated as-is;
        this function does not round or clip them.

    Returns
    -------
    dict
        Keys ``mae`` and ``rmse``, both in people. Zero means no error on the
        evaluated examples. Neither metric is a percentage accuracy.

    Raises
    ------
    ValueError
        If either input is empty, non-numeric, non-finite or not one-dimensional,
        or if their lengths differ. Missing rows are not silently removed.

    Notes
    -----
    Matching is positional, not by pandas index labels. The caller must preserve
    forecast-row alignment. Residuals use observed minus predicted, as in Week 5.5.
    """
    observed = _as_vector(y_true, "y_true")
    predicted = _as_vector(y_pred, "y_pred")
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same number of values.")

    residuals = observed - predicted
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}
