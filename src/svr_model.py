"""Build the SVR and its input scaling as one model.

Input preparation stays in src.features and error calculations stay in
src.evaluation. This file does not load data or select settings from results.
"""

from numbers import Real
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def build_svr_model(C: float = 1.0, epsilon: float = 0.1,
                    gamma: float = 0.1) -> Pipeline:
    """Create an SVR that reuses its training scale when making predictions.

    Args:
        C (float): Positive penalty on training errors outside the epsilon band.
            A larger value puts more emphasis on reducing those errors, but
            does not guarantee better future forecasts. Defaults to 1.0.
        epsilon (float): Allowed error on either side of the training target
            before the error penalty applies. It is measured in people and
            must be at least zero. Defaults to 0.1. It is not a guaranteed
            forecast accuracy or the five minute measurement rule.
        gamma (float): Positive setting controlling how local the RBF
            similarities are after scaling. Larger values make similarities
            more local. This project accepts numeric values only. Defaults to 0.1.

    Returns:
        sklearn.pipeline.Pipeline: An unfitted model with steps named scaler
            and svr. Fit it on the training examples only. Every encoded input
            column is centred and scaled, including weekday indicators.
            Constant columns use a scale of one. Targets stay in people.

    Raises:
        ValueError: A setting is missing a valid numeric value, is infinite,
            or is outside its allowed range. Booleans and strings are rejected.

    Notes:
        Supply the output of src.features.make_features, not the full data table.
        Calling predict reuses the fitted scale without learning from new rows.
        Keeping future targets out of the inputs is still the caller's responsibility.
        The solver tolerance is 0.001 and the kernel cache limit is 256 MB.
        There is no limit on solver iterations. Predictions are not rounded or
        forced to be positive, and the model does not return uncertainty bounds.

    Examples:
        >>> model = build_svr_model(C=1.0, epsilon=0.1, gamma=0.1)
        >>> model.named_steps["svr"].kernel
        'rbf'
    """
    # Catch invalid settings before starting an expensive fit.
    for name, value in [("C", C), ("epsilon", epsilon), ("gamma", gamma)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number.")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if (name == "epsilon" and value < 0) or (name != "epsilon" and value <= 0):
            requirement = "non-negative" if name == "epsilon" else "positive"
            raise ValueError(f"{name} must be {requirement}.")


    # Keep scaling with the model so prediction reuses the training scale.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=float(C), epsilon=float(epsilon),
                    gamma=float(gamma), tol=0.001, cache_size=256,
                    max_iter=-1))
    ])
