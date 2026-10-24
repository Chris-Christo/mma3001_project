"""Checks used by the final, single-space forecasting comparison.

These functions do not fit models, read files, or use test scores to select a
model. They validate prepared rows, verify saved-prediction alignment, select
from validation scores only, and calculate an explicitly labelled error change.
"""

import numpy as np
import pandas as pd
from src.features import INPUT_COLUMNS, make_features


def validate_partition(data: pd.DataFrame, settings: dict, split: str) -> None:
    """Check a prepared partition against the agreed time and data rules.

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared forecast examples, including inputs, target, and source-time
        audit columns. Timestamps may be text or timezone-aware timestamps.
    settings : dict
        Contents of Notebook 2's preparation_settings.json.
    split : {'train', 'validation', 'test'}
        Partition whose chronological bounds must be respected.

    Raises
    ------
    ValueError
        If rows are missing, duplicated, out of order, outside their partition,
        have invalid counts/time features, or use stale/future observations.

    Notes
    -----
    No rows are changed or discarded. Passing these checks verifies preparation
    rules, not sensor accuracy, forecasting accuracy, or physical room identity.
    """
    periods = {
        "train": (settings["train_start"], settings["validation_start"]),
        "validation": (settings["validation_start"], settings["test_start"]),
        "test": (settings["test_start"], settings["study_end_exclusive"]),
    }
    if split not in periods:
        raise ValueError("split must be train, validation, or test.")
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("The partition must be a non-empty DataFrame.")
    if not data.columns.is_unique:
        raise ValueError("Repeated column names are not supported.")
    horizon = pd.Timedelta(minutes=settings["horizon_minutes"])
    max_age = pd.Timedelta(minutes=settings["max_reading_age_minutes"])
    if max_age < pd.Timedelta(0) or horizon <= max_age:
        raise ValueError("The horizon must exceed the non-negative freshness allowance.")
    offsets = {
        "headcount_now": pd.Timedelta(0),
        "headcount_lag15": pd.Timedelta(minutes=-15),
        "headcount_lag30": pd.Timedelta(minutes=-30),
        "target_headcount": horizon,
    }
    required = INPUT_COLUMNS + ["forecast_time", "target_time", "target_headcount"]
    required += [name + "_observed_at" for name in offsets]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"Missing prepared columns: {missing}")
    if len(data) != settings["split_counts"][split]:
        raise ValueError("Row count differs from the preparation settings.")

    # Encoding validates all five inputs without learning statistics or using targets.
    make_features(data)
    target = data["target_headcount"].to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target < 0).any() or (target % 1 != 0).any():
        raise ValueError("Targets must be finite, non-negative whole-number counts.")

    forecast = pd.to_datetime(data["forecast_time"], utc=True, errors="raise")
    nominal_target = pd.to_datetime(data["target_time"], utc=True, errors="raise")
    if forecast.isna().any() or nominal_target.isna().any():
        raise ValueError("Forecast and target timestamps must not be missing.")
    if not forecast.is_unique or not forecast.is_monotonic_increasing:
        raise ValueError("Forecast rows must be unique and chronological.")
    start, end = (pd.Timestamp(value) for value in periods[split])
    if not (forecast >= start).all() or not (nominal_target < end).all():
        raise ValueError("A forecast or its target crosses the partition boundary.")
    if not ((nominal_target - forecast) == horizon).all():
        raise ValueError("A forecast has the wrong nominal horizon.")

    for name, offset in offsets.items():
        observed = pd.to_datetime(data[name + "_observed_at"], utc=True, errors="raise")
        age = forecast + offset - observed
        if age.isna().any() or not age.between(pd.Timedelta(0), max_age).all():
            raise ValueError(f"{name} has missing, future, or stale measurement times.")
        if name == "target_headcount" and not (observed > forecast).all():
            raise ValueError("Target readings must be observed after forecast issuance.")
        age_column = name + "_age_minutes"
        if age_column in data.columns:
            saved_age = data[age_column].to_numpy(dtype=float)
            computed_age = age.dt.total_seconds().to_numpy() / 60
            if not np.allclose(saved_age, computed_age, rtol=0, atol=1e-8):
                raise ValueError(f"Saved observation ages disagree for {name}.")

    local = forecast.dt.tz_convert(settings["local_timezone"])
    hours = local.dt.hour + local.dt.minute / 60
    if not np.allclose(data["time_of_day"], hours, rtol=0, atol=1e-10):
        raise ValueError("Time-of-day inputs disagree with Melbourne forecast times.")
    if not np.array_equal(data["day_of_week"].to_numpy(), local.dt.dayofweek.to_numpy()):
        raise ValueError("Weekday inputs disagree with the forecast dates.")


def verify_prediction_alignment(data: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Check that saved predictions describe exactly the same rows and targets.

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared partition, in forecast-time order.
    predictions : pandas.DataFrame
        Saved model output containing forecast_time, target_time,
        target_headcount_observed_at, observed_headcount and predicted_headcount.

    Raises
    ------
    ValueError
        If rows, timestamps or observed targets differ, or predictions are
        missing/non-finite. Fractional and negative raw predictions are allowed.

    Notes
    -----
    Rows are compared positionally. The function does not silently sort, join,
    or drop examples to make two different evaluations look comparable.
    """
    time_columns = ["forecast_time", "target_time", "target_headcount_observed_at"]
    required = time_columns + ["observed_headcount", "predicted_headcount"]
    if any(name not in predictions.columns for name in required):
        raise ValueError("The saved prediction table is missing required columns.")
    if len(data) == 0 or len(data) != len(predictions):
        raise ValueError("Prepared and prediction row counts differ or are empty.")
    for name in time_columns:
        left = pd.to_datetime(data[name], utc=True).reset_index(drop=True)
        right = pd.to_datetime(predictions[name], utc=True).reset_index(drop=True)
        if left.isna().any() or right.isna().any() or not (left == right).all():
            raise ValueError(f"Saved predictions are not aligned: {name}.")
    times = pd.to_datetime(predictions["forecast_time"], utc=True)
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Prediction rows must be unique and chronological.")
    expected = data["target_headcount"].to_numpy(dtype=float)
    observed = predictions["observed_headcount"].to_numpy(dtype=float)
    predicted = predictions["predicted_headcount"].to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Predictions or recorded targets are non-finite.")
    if not np.array_equal(expected, observed):
        raise ValueError("Saved observed headcounts differ from prepared targets.")


def select_validation_model(scores: pd.DataFrame) -> str:
    """Select minimum validation MAE, then RMSE, then model name for exact ties.

    Parameters
    ----------
    scores : pandas.DataFrame
        Exactly one validation row per model, with columns model, split,
        examples, mae_people and rmse_people. All models use the same row count;
        verify_prediction_alignment separately checks the actual row identities.

    Returns
    -------
    str
        Name of the selected model. Runtime is reported separately, not used
        as an unstated accuracy-selection criterion.

    Raises
    ------
    ValueError
        If test/training rows, duplicate models, invalid errors, or unequal
        example counts are supplied. The function cannot inspect test results.
    """
    required = ["model", "split", "examples", "mae_people", "rmse_people"]
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        raise ValueError("A non-empty validation score table is required.")
    if any(name not in scores.columns for name in required):
        raise ValueError("The validation table is missing required columns.")
    if not (scores["split"] == "validation").all():
        raise ValueError("Selection accepts validation rows only; never test scores.")
    names = scores["model"]
    if not names.is_unique or not names.map(lambda name: isinstance(name, str) and bool(name)).all():
        raise ValueError("Model names must be non-empty, unique strings.")
    numeric = scores[["examples", "mae_people", "rmse_people"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 1:] < 0).any():
        raise ValueError("Scores must be finite and errors non-negative.")
    counts = numeric[:, 0]
    if (counts <= 0).any() or (counts % 1 != 0).any() or not (counts == counts[0]).all():
        raise ValueError("Models must have equal, positive whole-number example counts.")
    ranked = scores.sort_values(["mae_people", "rmse_people", "model"])
    return str(ranked.iloc[0]["model"])


def mae_reduction_percent(baseline_mae: float, model_mae: float) -> float:
    """Return relative MAE reduction; return NaN if the baseline error is zero.

    Parameters
    ----------
    baseline_mae, model_mae : float
        Finite, non-negative MAEs for the same forecast examples.

    Returns
    -------
    float
        100 * (baseline_mae - model_mae) / baseline_mae. Positive means less
        error; negative means worse than baseline. This is not percent accuracy.
        A zero denominator has no defined relative change, so returns NaN.

    Raises
    ------
    ValueError
        If either MAE is negative or non-finite.
    """
    baseline = float(baseline_mae)
    model = float(model_mae)
    if not np.isfinite([baseline, model]).all() or baseline < 0 or model < 0:
        raise ValueError("MAEs must be finite and non-negative.")
    if baseline == 0:
        return float("nan")
    return 100 * (baseline - model) / baseline
