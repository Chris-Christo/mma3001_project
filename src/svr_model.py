"""Construct the scaled RBF support-vector regressor used in Notebook 5.

Input encoding stays in src.features; error calculations stay in src.evaluation.
This module does not load data or choose parameters from evaluation results.
"""

from numbers import Real
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def build_svr_model(C: float = 1.0, epsilon: float = 0.1,
                    gamma: float = 0.1) -> Pipeline:
    """Create an unfitted StandardScaler -> RBF SVR pipeline.

    Parameters
    ----------
    C : float, default=1.0
        Positive penalty weight for training errors outside the epsilon tube.
        Larger values put more emphasis on reducing those errors relative to
        controlling model complexity; they do not guarantee better forecasts.
    epsilon : float, default=0.1
        Non-negative half-width of the insensitive tube, in people because the
        target is not scaled. This is not an error guarantee or time tolerance.
    gamma : float, default=0.1
        Positive RBF kernel coefficient, applied after input standardisation.
        Larger values make similarities more local in scaled feature space.
        This project's bounded search uses numerical gamma values only.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Named steps are ``scaler`` and ``svr``. Call fit on training data only.
        The scaler centres and scales every encoded input column, including
        weekday flags. Constant columns use a scale of one. Predictions retain
        their raw fractional or negative values; no rounding or clipping occurs.

    Raises
    ------
    ValueError
        If C or gamma is not finite and positive, or epsilon is not finite and
        non-negative. Booleans and strings are not accepted as numbers.

    Notes
    -----
    Pass the output of src.features.make_features, not the full prepared table.
    The pipeline prevents validation statistics being fitted accidentally when
    predict is called, but the caller must still respect chronological splitting
    and keep targets and future information out of the inputs.
    The solver uses tol=0.001, a 256 MB kernel-cache limit and no iteration cap.
    Standard SVR has no built-in non-negative-output or uncertainty guarantee.

    Examples
    --------
    >>> model = build_svr_model(C=1.0, epsilon=0.1, gamma=0.1)
    >>> model.named_steps["svr"].kernel
    'rbf'
    """
    for name, value in [("C", C), ("epsilon", epsilon), ("gamma", gamma)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number.")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if (name == "epsilon" and value < 0) or (name != "epsilon" and value <= 0):
            requirement = "non-negative" if name == "epsilon" else "positive"
            raise ValueError(f"{name} must be {requirement}.")

    # Keeping scaling with the estimator ensures predict reuses training scaling.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=float(C), epsilon=float(epsilon),
                    gamma=float(gamma), tol=0.001, cache_size=256,
                    max_iter=-1))
    ])
