"""Small synthetic checks with known answers; no project sensor data is needed."""

import numpy as np
import pytest
from src.evaluation import persistence_predict, regression_metrics


def test_persistence_returns_the_current_counts():
    predicted = persistence_predict([0, 4, 12])
    np.testing.assert_array_equal(predicted, [0, 4, 12])


def test_persistence_does_not_change_its_input():
    current = np.array([2.0, 5.0])
    predicted = persistence_predict(current)
    predicted[0] = 99.0
    np.testing.assert_array_equal(current, [2.0, 5.0])


def test_persistence_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        persistence_predict([1, -1])


@pytest.mark.parametrize("bad_input", [[], [[1, 2]], [1, np.nan], [1, np.inf], ["not a count"]])
def test_persistence_rejects_invalid_inputs(bad_input):
    with pytest.raises(ValueError):
        persistence_predict(bad_input)


def test_metrics_match_a_hand_calculation():
    # Residuals are [1, -2, 2]: MAE = 5/3; RMSE = sqrt(9/3).
    scores = regression_metrics([3, 5, 10], [2, 7, 8])
    assert scores["mae"] == pytest.approx(5 / 3)
    assert scores["rmse"] == pytest.approx(np.sqrt(3))


def test_perfect_predictions_have_zero_error():
    assert regression_metrics([0, 2, 10], [0, 2, 10]) == {"mae": 0.0, "rmse": 0.0}


def test_zero_headcounts_are_valid_observations():
    scores = regression_metrics([0, 0], [0, 2])
    assert scores["mae"] == pytest.approx(1.0)
    assert scores["rmse"] == pytest.approx(np.sqrt(2))


def test_fractional_predictions_are_not_rounded():
    assert regression_metrics([1], [1.5]) == {"mae": 0.5, "rmse": 0.5}


def test_negative_raw_predictions_are_not_silently_clipped():
    assert regression_metrics([1], [-1]) == {"mae": 2.0, "rmse": 2.0}


def test_different_lengths_are_rejected():
    with pytest.raises(ValueError, match="same number"):
        regression_metrics([1, 2], [1])


@pytest.mark.parametrize("observed,predicted", [
    ([], []),
    ([[1, 2]], [1, 2]),
    ([1, 2], [[1, 2]]),
    ([1, np.nan], [1, 2]),
    ([1, 2], [1, np.nan]),
    ([1, np.inf], [1, 2]),
    ([1, 2], [1, np.inf]),
])
def test_metrics_reject_invalid_arrays(observed, predicted):
    with pytest.raises(ValueError):
        regression_metrics(observed, predicted)
