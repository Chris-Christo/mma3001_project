"""Check SVR construction, scaling and prediction using small made up examples.

These checks do not establish accuracy on the building dataset.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from src.features import make_features
from src.svr_model import build_svr_model


def small_training_data():
    """Provide a small repeatable dataset so SVR behaviour can be checked quickly."""
    data = pd.DataFrame({
        "headcount_now": [0, 1, 3, 5, 7, 9, 11, 13],
        "headcount_lag15": [0, 0, 2, 4, 6, 8, 10, 12],
        "headcount_lag30": [0, 0, 1, 3, 5, 7, 9, 11],
        "time_of_day": [0, 3, 6, 9, 12, 15, 18, 21],
        "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0],
    })
    return make_features(data), np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=float)


def test_factory_returns_an_unfitted_scaled_rbf_model():
    """Confirm that creating a model does not train it or change its requested settings."""
    model = build_svr_model(C=2.0, epsilon=0.5, gamma=0.01)
    assert list(model.named_steps) == ["scaler", "svr"]
    estimator = model.named_steps["svr"]
    assert estimator.kernel == "rbf"
    assert (estimator.C, estimator.epsilon, estimator.gamma) == (2.0, 0.5, 0.01)
    assert not hasattr(model.named_steps["scaler"], "mean_")
    assert not hasattr(estimator, "support_")


@pytest.mark.parametrize("parameters", [
    {"C": 0}, {"C": -1}, {"C": np.nan}, {"C": np.inf}, {"C": True},
    {"epsilon": -0.1}, {"epsilon": np.inf}, {"epsilon": "0.1"},
    {"gamma": 0}, {"gamma": -1}, {"gamma": np.nan}, {"gamma": "scale"},
])
def test_invalid_settings_are_rejected(parameters):
    """Catch invalid settings before spending time on fitting."""
    with pytest.raises(ValueError):
        build_svr_model(**parameters)


def test_zero_epsilon_is_permitted():
    """Allow a zero width error band because it is a valid SVR setting."""
    assert build_svr_model(epsilon=0).named_steps["svr"].epsilon == 0


def test_predictions_require_fitting_first():
    """Prevent forecasts from a model that has not learned from any examples."""
    x, _ = small_training_data()
    with pytest.raises(NotFittedError):
        build_svr_model().predict(x)


def test_scaling_uses_only_the_training_rows():
    """Check that later observations cannot change the scale learned from training."""
    x, y = small_training_data()
    model = build_svr_model().fit(x, y)
    scaler = model.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, x.mean().to_numpy())
    np.testing.assert_allclose(scaler.var_, x.var(ddof=0).to_numpy())
    assert scaler.n_samples_seen_ == len(x)

    future = x.iloc[:2].copy()
    # Extreme later values make accidental changes to training scaling easier to detect.
    future["headcount_now"] = [1000, 2000]
    means_before = scaler.mean_.copy()
    model.predict(future)
    np.testing.assert_array_equal(scaler.mean_, means_before)
    assert scaler.n_samples_seen_ == len(x)


def test_prediction_does_not_refit_support_vectors_or_weights():
    """Keep the learned model fixed during evaluation."""
    x, y = small_training_data()
    model = build_svr_model().fit(x, y)
    estimator = model.named_steps["svr"]
    support_before = estimator.support_vectors_.copy()
    weights_before = estimator.dual_coef_.copy()
    predicted = model.predict(x.iloc[[4, 1, 6]])
    assert predicted.shape == (3,)
    np.testing.assert_array_equal(estimator.support_vectors_, support_before)
    np.testing.assert_array_equal(estimator.dual_coef_, weights_before)


def test_predictions_preserve_requested_row_order():
    """Keep each forecast attached to the example it was made for."""
    x, y = small_training_data()
    model = build_svr_model().fit(x, y)
    all_predictions = model.predict(x)
    np.testing.assert_allclose(model.predict(x.iloc[[5, 0, 3]]), all_predictions[[5, 0, 3]])


def test_fitting_does_not_change_input_features():
    """Preserve the prepared inputs for fair comparisons with other methods."""
    x, y = small_training_data()
    original = x.copy(deep=True)
    build_svr_model().fit(x, y)
    pd.testing.assert_frame_equal(x, original)


def test_constant_fractional_target_is_not_rounded():
    """Check that decimal forecasts survive without an unreported rounding step."""
    x, _ = small_training_data()
    model = build_svr_model(epsilon=0).fit(x, np.full(len(x), 2.25))
    np.testing.assert_allclose(model.predict(x), 2.25, atol=1e-8)


def test_smooth_nonlinear_example_on_withheld_points():
    """Check the fitting workflow against a simple known curve."""
    x = np.linspace(-2, 2, 121).reshape(-1, 1)
    y = 1 + x[:, 0] ** 2
    # These withheld points test a known curve, not the chronological forecasting design.
    train_rows = np.arange(len(x)) % 3 != 0
    model = build_svr_model(C=100, epsilon=0.01, gamma=0.5)
    model.fit(x[train_rows], y[train_rows])
    error = np.abs(model.predict(x[~train_rows]) - y[~train_rows]).mean()
    assert error < 0.06


def test_non_finite_features_are_rejected_at_fitting():
    """Prevent missing inputs from entering model fitting."""
    x, y = small_training_data()
    x.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        build_svr_model().fit(x, y)


def test_non_finite_targets_are_rejected():
    """Prevent missing answers from entering model fitting."""
    x, y = small_training_data()
    y[0] = np.nan
    with pytest.raises(ValueError):
        build_svr_model().fit(x, y)


def test_missing_encoded_column_is_rejected_at_prediction():
    """Stop forecasts when the supplied inputs no longer match the fitted model."""
    x, y = small_training_data()
    model = build_svr_model().fit(x, y)
    with pytest.raises(ValueError):
        model.predict(x.drop(columns=x.columns[-1]))


def test_saved_pipeline_reproduces_predictions(tmp_path):
    """Check that saving and loading a local model preserves its predictions."""
    x, y = small_training_data()
    model = build_svr_model().fit(x, y)
    path = tmp_path / "local_test_model.joblib"
    joblib.dump(model, path)
    # Load only the temporary model created by this test, not an unknown external file.
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(x), model.predict(x))
