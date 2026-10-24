"""Checks that keep the final comparison consistent with the project design.

The functions check prepared examples, match saved predictions to their targets,
select a model from validation scores and report changes in MAE. They do not
load files, train models or use test scores to select a model.
"""

import numpy as np
import pandas as pd
from src.features import INPUT_COLUMNS, make_features


def validate_partition(data: pd.DataFrame, settings: dict, split: str) -> None:
    """Check that one data split follows the preparation rules.

    Args:
        data (pandas.DataFrame): Prepared examples with the five model inputs,
            future target, forecast time, target time and observation times.
            Dates may be stored as text or timestamps. Rows must be in time order.
        settings (dict): The preparation_settings.json contents saved by
            Notebook 2, including dates, row counts and timing allowances.
        split (str): The split to check: train, validation or test.

    Returns:
        None: Completes without changing the data when the checks pass.

    Raises:
        ValueError: Rows or columns are missing or repeated, counts are invalid,
            rows are out of order, times cross a split boundary, a reading is
            too old or too late, or saved time inputs disagree with the dates.
        KeyError: A required preparation setting is missing.

    Notes:
        The checks do not repair or remove rows. Passing them supports correct
        preparation, not sensor accuracy or forecasting accuracy. The function
        does not verify the physical room associated with the space identifier.
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
    # Even the oldest acceptable target must come after the forecast starts.
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
    # Catch missing or extra examples before comparing model results.
    if len(data) != settings["split_counts"][split]:
        raise ValueError("Row count differs from the preparation settings.")


    # Reuse the input checks without learning from targets or other rows.
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
    # Keep future targets inside their own split rather than borrowing from the next one.
    if not (forecast >= start).all() or not (nominal_target < end).all():
        raise ValueError("A forecast or its target crosses the partition boundary.")
    if not ((nominal_target - forecast) == horizon).all():
        raise ValueError("A forecast has the wrong nominal horizon.")

    # Check every count against its own requested time, not just the current count.
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

    # Clock inputs must describe Melbourne time even though saved timestamps use UTC.
    local = forecast.dt.tz_convert(settings["local_timezone"])
    hours = local.dt.hour + local.dt.minute / 60
    if not np.allclose(data["time_of_day"], hours, rtol=0, atol=1e-10):
        raise ValueError("Time-of-day inputs disagree with Melbourne forecast times.")
    if not np.array_equal(data["day_of_week"].to_numpy(), local.dt.dayofweek.to_numpy()):
        raise ValueError("Weekday inputs disagree with the forecast dates.")


def verify_prediction_alignment(data: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Prevent a model comparison from pairing predictions with the wrong examples.

    Args:
        data (pandas.DataFrame): Prepared examples in forecast time order.
        predictions (pandas.DataFrame): Saved predictions with forecast_time,
            target_time, target_headcount_observed_at, observed_headcount
            and predicted_headcount.

    Returns:
        None: Completes without changing either table when the rows match.

    Raises:
        ValueError: Required prediction columns are absent, tables are empty or
            have different lengths, dates or targets differ, or prediction values
            are missing or infinite. Repeated or unordered forecast times fail.
        KeyError: A required column is absent from the prepared data.

    Notes:
        Rows are compared in their existing order. They are not sorted or dropped
        to make different results appear comparable. Decimal and negative raw
        predictions remain allowed so model limitations stay visible.
    """
    time_columns = ["forecast_time", "target_time", "target_headcount_observed_at"]
    required = time_columns + ["observed_headcount", "predicted_headcount"]
    if any(name not in predictions.columns for name in required):
        raise ValueError("The saved prediction table is missing required columns.")
    if len(data) == 0 or len(data) != len(predictions):
        raise ValueError("Prepared and prediction row counts differ or are empty.")
    # Matching row counts alone cannot prove that the same forecasts were evaluated.
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
    # Do not allow a saved result to substitute different reference headcounts.
    if not np.array_equal(expected, observed):
        raise ValueError("Saved observed headcounts differ from prepared targets.")


def select_validation_model(scores: pd.DataFrame) -> str:
    """Choose the model with the lowest validation MAE using a fixed tie rule.

    Args:
        scores (pandas.DataFrame): One validation row per model with model,
            split, examples, mae_people and rmse_people. Example counts must
            be equal. Matching the actual examples is a separate check in
            verify_prediction_alignment.

    Returns:
        str: The selected model name. Equal MAEs are compared by RMSE, then
            by model name for an exact tie. Runtime is not used for selection.

    Raises:
        ValueError: The table is empty, has missing columns, includes training
            or test rows, repeats a model, or has invalid errors or unequal counts.

    Notes:
        Validation scores guide selection. Test scores are deliberately rejected
        so the final evaluation cannot become another model selection step.
    """
    required = ["model", "split", "examples", "mae_people", "rmse_people"]
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        raise ValueError("A non-empty validation score table is required.")
    if any(name not in scores.columns for name in required):
        raise ValueError("The validation table is missing required columns.")
    # Reserve the test period for assessment rather than model choice.
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
    # Resolve exact ties consistently instead of changing the selection rule after seeing results.
    ranked = scores.sort_values(["mae_people", "rmse_people", "model"])
    return str(ranked.iloc[0]["model"])


def mae_reduction_percent(baseline_mae: float, model_mae: float) -> float:
    """Express the change from baseline MAE without calling it percentage accuracy.

    Args:
        baseline_mae (float): Baseline MAE on the comparison examples.
        model_mae (float): Model MAE on exactly the same examples.
            Both inputs must be finite numbers at least zero.

    Returns:
        float: Baseline MAE minus model MAE, divided by baseline MAE and
            multiplied by 100. Positive means less error; negative means more.
            Returns NaN when the baseline is zero because the ratio is undefined.

    Raises:
        ValueError: Either MAE is negative or infinite, or is NaN.
    """
    baseline = float(baseline_mae)
    model = float(model_mae)
    if not np.isfinite([baseline, model]).all() or baseline < 0 or model < 0:
        raise ValueError("MAEs must be finite and non-negative.")
    # A perfect baseline has no meaningful relative improvement denominator.
    if baseline == 0:
        return float("nan")
    return 100 * (baseline - model) / baseline
