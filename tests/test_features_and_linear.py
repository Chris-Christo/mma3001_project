"""Synthetic checks of input encoding and the intended linear-regression workflow.

These do not use the building dataset or measure real forecasting performance.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from src.features import make_features, INPUT_COLUMNS, ENCODED_COLUMNS


def example_data():
    return pd.DataFrame({
        "headcount_now": [2, 4, 6, 8],
        "headcount_lag15": [1, 3, 5, 7],
        "headcount_lag30": [0, 2, 4, 6],
        "time_of_day": [0, 6, 12, 18],
        "day_of_week": [0, 1, 5, 6]
    }, index=[9, 2, 8, 3])


def test_column_order_row_order_and_unchanged_counts():
    data = example_data()
    x = make_features(data)
    assert list(x.columns) == ENCODED_COLUMNS
    assert x.index.equals(data.index)
    np.testing.assert_array_equal(x[INPUT_COLUMNS[:3]], data[INPUT_COLUMNS[:3]])


def test_clock_encoding_at_four_known_times():
    x = make_features(example_data())
    np.testing.assert_allclose(x["time_sin"], [0, 1, 0, -1], atol=1e-12)
    np.testing.assert_allclose(x["time_cos"], [1, 0, -1, 0], atol=1e-12)


def test_times_on_either_side_of_midnight_are_close():
    data = example_data().iloc[:2].copy()
    data["time_of_day"] = [23.75, 0.25]
    x = make_features(data)[["time_sin", "time_cos"]].to_numpy()
    assert np.linalg.norm(x[0] - x[1]) < 0.14


def test_weekday_flags_and_monday_reference():
    x = make_features(example_data())
    day_columns = ENCODED_COLUMNS[5:]
    np.testing.assert_array_equal(x[day_columns].sum(axis=1), [0, 1, 1, 1])
    assert x.loc[2, "is_tuesday"] == 1
    assert x.loc[8, "is_saturday"] == 1
    assert x.loc[3, "is_sunday"] == 1


def test_features_do_not_change_when_targets_change():
    data = example_data()
    expected = make_features(data)
    data["target_headcount"] = [999, 0, 999, 0]
    data["target_time"] = "future information"
    pd.testing.assert_frame_equal(make_features(data), expected)


def test_encoding_one_row_does_not_depend_on_other_rows():
    data = example_data()
    alone = make_features(data.iloc[[0]])
    together = make_features(data).iloc[[0]]
    pd.testing.assert_frame_equal(alone, together)


def test_input_is_not_modified():
    data = example_data()
    original = data.copy(deep=True)
    make_features(data)
    pd.testing.assert_frame_equal(data, original)


def test_missing_required_column_is_rejected():
    with pytest.raises(ValueError, match="Missing input columns"):
        make_features(example_data().drop(columns="headcount_lag15"))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        make_features(example_data().iloc[:0])


@pytest.mark.parametrize("column,bad_value", [
    ("headcount_now", -1), ("headcount_lag15", 1.5),
    ("headcount_lag30", np.nan), ("headcount_now", np.inf),
    ("time_of_day", -0.25), ("time_of_day", 24),
    ("day_of_week", -1), ("day_of_week", 7), ("day_of_week", 2.5),
])
def test_invalid_input_values_are_rejected(column, bad_value):
    data = example_data().astype(float)
    data.loc[data.index[0], column] = bad_value
    with pytest.raises(ValueError):
        make_features(data)


def test_non_numeric_input_is_rejected():
    data = example_data()
    data["headcount_now"] = "not a count"
    with pytest.raises(ValueError):
        make_features(data)


def synthetic_data(n=240):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "headcount_now": rng.integers(0, 20, n),
        "headcount_lag15": rng.integers(0, 20, n),
        "headcount_lag30": rng.integers(0, 20, n),
        "time_of_day": rng.integers(0, 96, n) / 4,
        "day_of_week": rng.integers(0, 7, n),
    })


def test_pipeline_recovers_a_known_linear_relation_on_withheld_rows():
    # This artificial target has an exact linear relationship, unlike real occupancy.
    data = synthetic_data()
    x = make_features(data)
    weights = np.arange(1, len(x.columns) + 1) / 10
    y = 2.5 + x.to_numpy() @ weights
    model = LinearRegression().fit(x.iloc[:180], y[:180])
    np.testing.assert_allclose(model.predict(x.iloc[180:]), y[180:], atol=1e-10)
    np.testing.assert_allclose(model.coef_, weights, atol=1e-10)
    assert model.intercept_ == pytest.approx(2.5)


def test_predicting_later_rows_does_not_refit_the_model():
    x = make_features(synthetic_data())
    y = 1 + 0.5 * x["headcount_now"].to_numpy()
    model = LinearRegression().fit(x.iloc[:180], y[:180])
    before = model.coef_.copy()
    intercept_before = model.intercept_
    predictions = model.predict(x.iloc[180:])
    assert predictions.shape == (60,)
    np.testing.assert_array_equal(model.coef_, before)
    assert model.intercept_ == intercept_before
