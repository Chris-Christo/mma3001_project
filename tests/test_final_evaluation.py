"""Synthetic implementation tests for final-comparison safeguards, not real accuracy."""

import numpy as np
import pandas as pd
import pytest
from src.final_evaluation import (
    validate_partition, verify_prediction_alignment,
    select_validation_model, mae_reduction_percent,
)


@pytest.fixture
def prepared():
    forecast = pd.date_range("2024-05-06 10:00", periods=3, freq="15min", tz="Australia/Melbourne")
    data = pd.DataFrame({
        "forecast_time": forecast,
        "target_time": forecast + pd.Timedelta(minutes=30),
        "headcount_now": [3, 4, 5],
        "headcount_lag15": [2, 3, 4],
        "headcount_lag30": [1, 2, 3],
        "time_of_day": [10, 10.25, 10.5],
        "day_of_week": [0, 0, 0],
        "target_headcount": [4, 5, 4],
    })
    offsets = {"headcount_now": 0, "headcount_lag15": -15,
               "headcount_lag30": -30, "target_headcount": 30}
    for name, minutes in offsets.items():
        data[name + "_observed_at"] = forecast + pd.Timedelta(minutes=minutes - 2)
        data[name + "_age_minutes"] = 2.0
    settings = {
        "train_start": "2024-03-01T00:00:00+11:00",
        "validation_start": "2025-09-01T00:00:00+10:00",
        "test_start": "2026-01-01T00:00:00+11:00",
        "study_end_exclusive": "2026-04-01T00:00:00+11:00",
        "horizon_minutes": 30,
        "max_reading_age_minutes": 5,
        "split_counts": {"train": 3, "validation": 3, "test": 3},
        "local_timezone": "Australia/Melbourne",
    }
    return data, settings


@pytest.fixture
def scores():
    return pd.DataFrame({
        "model": ["persistence", "linear_regression", "svr"],
        "split": ["validation"] * 3,
        "examples": [10] * 3,
        "mae_people": [1.0, 0.9, 0.8],
        "rmse_people": [1.5, 1.2, 1.3],
    })


def predictions_for(data):
    result = data[["forecast_time", "target_time", "target_headcount_observed_at"]].copy()
    result["observed_headcount"] = data["target_headcount"].to_numpy()
    result["predicted_headcount"] = data["headcount_now"].to_numpy(dtype=float)
    return result


def test_valid_partition_is_accepted_without_mutation(prepared):
    data, settings = prepared
    original = data.copy(deep=True)
    validate_partition(data, settings, "train")
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("split", ["validation", "test", "unknown"])
def test_wrong_partition_rejected(prepared, split):
    data, settings = prepared
    with pytest.raises(ValueError):
        validate_partition(data, settings, split)


@pytest.mark.parametrize("minutes", [-1, 6])
def test_future_or_stale_input_rejected(prepared, minutes):
    data, settings = prepared
    data["headcount_now_observed_at"] = data["forecast_time"] - pd.Timedelta(minutes=minutes)
    with pytest.raises(ValueError):
        validate_partition(data, settings, "train")


@pytest.mark.parametrize("minutes", [0, 5])
def test_freshness_endpoints_accepted(prepared, minutes):
    data, settings = prepared
    data["headcount_now_observed_at"] = data["forecast_time"] - pd.Timedelta(minutes=minutes)
    data["headcount_now_age_minutes"] = float(minutes)
    validate_partition(data, settings, "train")


def test_duplicate_forecast_rejected(prepared):
    data, settings = prepared
    data.loc[1, "forecast_time"] = data.loc[0, "forecast_time"]
    with pytest.raises(ValueError):
        validate_partition(data, settings, "train")


def test_reversed_order_rejected(prepared):
    data, settings = prepared
    with pytest.raises(ValueError):
        validate_partition(data.iloc[::-1], settings, "train")


def test_wrong_horizon_rejected(prepared):
    data, settings = prepared
    data["target_time"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        validate_partition(data, settings, "train")


def test_target_crossing_end_rejected(prepared):
    data, settings = prepared
    settings["validation_start"] = "2024-05-06T10:45:00+10:00"
    with pytest.raises(ValueError):
        validate_partition(data, settings, "train")


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 1.5])
def test_invalid_target_rejected(prepared, value):
    data, settings = prepared
    data["target_headcount"] = data["target_headcount"].astype(float)
    data.loc[0, "target_headcount"] = value
    with pytest.raises(ValueError):
        validate_partition(data, settings, "train")


@pytest.mark.parametrize("column", ["time_of_day", "day_of_week", "headcount_now_age_minutes"])
def test_inconsistent_audit_or_clock_rejected(prepared, column):
    data, settings = prepared
    data.loc[0, column] += 1
    with pytest.raises(ValueError):
        validate_partition(data, settings, "train")


def test_missing_column_rejected(prepared):
    data, settings = prepared
    with pytest.raises(ValueError):
        validate_partition(data.drop(columns="headcount_now"), settings, "train")


def test_wrong_row_count_rejected(prepared):
    data, settings = prepared
    with pytest.raises(ValueError):
        validate_partition(data.iloc[:2], settings, "train")


def test_matching_predictions_accepted(prepared):
    data, _ = prepared
    predictions = predictions_for(data)
    predictions.loc[0, "predicted_headcount"] = -0.25
    verify_prediction_alignment(data, predictions)


@pytest.mark.parametrize("problem", ["reorder", "target", "time", "nan", "fewer", "missing"])
def test_prediction_mismatches_rejected(prepared, problem):
    data, _ = prepared
    predictions = predictions_for(data)
    if problem == "reorder":
        predictions = predictions.iloc[::-1]
    elif problem == "target":
        predictions.loc[0, "observed_headcount"] += 1
    elif problem == "time":
        predictions.loc[0, "target_time"] += pd.Timedelta(seconds=1)
    elif problem == "nan":
        predictions.loc[0, "predicted_headcount"] = np.nan
    elif problem == "fewer":
        predictions = predictions.iloc[:2]
    else:
        predictions = predictions.drop(columns="predicted_headcount")
    with pytest.raises(ValueError):
        verify_prediction_alignment(data, predictions)


def test_selection_uses_validation_mae_not_lowest_rmse(scores):
    assert select_validation_model(scores) == "svr"


def test_selection_tie_uses_rmse(scores):
    scores.loc[2, "mae_people"] = 0.9
    assert select_validation_model(scores) == "linear_regression"


def test_exact_tie_is_reproducible(scores):
    scores["mae_people"] = 1.0
    scores["rmse_people"] = 1.0
    assert select_validation_model(scores) == "linear_regression"


@pytest.mark.parametrize("split", ["train", "test"])
def test_non_validation_scores_forbidden(scores, split):
    scores.loc[0, "split"] = split
    with pytest.raises(ValueError, match="validation rows only"):
        select_validation_model(scores)


@pytest.mark.parametrize("problem", ["duplicates", "different_counts", "negative", "nan", "empty"])
def test_invalid_score_tables_rejected(scores, problem):
    if problem == "duplicates":
        scores.loc[0, "model"] = "svr"
    elif problem == "different_counts":
        scores.loc[0, "examples"] = 9
    elif problem == "negative":
        scores.loc[0, "mae_people"] = -1
    elif problem == "nan":
        scores.loc[0, "rmse_people"] = np.nan
    else:
        scores = scores.iloc[:0]
    with pytest.raises(ValueError):
        select_validation_model(scores)


@pytest.mark.parametrize("baseline,model,expected", [(2, 1, 50), (2, 3, -50), (2, 2, 0)])
def test_relative_mae_change(baseline, model, expected):
    assert mae_reduction_percent(baseline, model) == expected


def test_zero_baseline_has_undefined_relative_change():
    assert np.isnan(mae_reduction_percent(0, 1))
    assert np.isnan(mae_reduction_percent(0, 0))


@pytest.mark.parametrize("baseline,model", [(-1, 1), (1, -1), (np.nan, 1), (1, np.inf)])
def test_bad_maes_rejected(baseline, model):
    with pytest.raises(ValueError):
        mae_reduction_percent(baseline, model)
