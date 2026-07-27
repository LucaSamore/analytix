"""Loading, chronological splits and forecast accuracy measures."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
M3_FILE = PROJECT_DIR / "M3C_monthly.csv"
M4_FILE = PROJECT_DIR / "M4_monthly_subset.csv"

SEASONAL_PERIOD = 12
FORECAST_HORIZON = 12
VALIDATION_HORIZON = 12
MINIMUM_TRAINING_OBSERVATIONS = 24
DM_EVALUATION_POINTS = 24


# Two series from the four largest M3 categories and one from the other two.
M3_SERIES_IDS = [
    "N1830", "N1842",   # MICRO
    "N1901", "N2185",   # INDUSTRY
    "N2438", "N2219",   # MACRO
    "N2591", "N2662",   # FINANCE
    "N2745",            # DEMOGRAPHIC
    "N2797",            # OTHER
]

# Five long M4 series from five different categories.
M4_SERIES_IDS = ["M15716", "M27126", "M233", "M43445", "M26707"]


def load_series(
    dataset: str,
    series_id: str,
) -> tuple[pd.Series, pd.Series]:
    """Load a series from M3 or M4."""

    if dataset == "M3":
        return _load_m3_series(series_id)

    if dataset == "M4":
        return _load_m4_series(series_id)

    raise ValueError("Dataset must be 'M3' or 'M4'.")


def _load_m3_series(series_id: str) -> tuple[pd.Series, pd.Series]:
    """Load one monthly series from the M3 file."""

    return _load_row(M3_FILE, series_id)


def _load_m4_series(series_id: str) -> tuple[pd.Series, pd.Series]:
    """Load one monthly series from the M4 file."""

    return _load_row(M4_FILE, series_id)


def _load_row(
    data_file: Path,
    series_id: str,
) -> tuple[pd.Series, pd.Series]:
    """Load one row and separate its observations from its metadata."""

    data = pd.read_csv(data_file)
    data["Series"] = data["Series"].str.strip()
    matching_rows = data.loc[data["Series"] == series_id]

    if matching_rows.empty:
        raise ValueError(f"Series {series_id!r} not found in {data_file.name}.")

    row = matching_rows.iloc[0]
    metadata = row.iloc[:6].copy()
    metadata["Category"] = str(metadata["Category"]).strip()

    values = row.iloc[6:].dropna().to_numpy(dtype=float)
    series = pd.Series(values, dtype=float)

    if int(metadata["N"]) != len(series):
        raise ValueError(
            f"Series {series_id!r} declares N={metadata['N']}, "
            f"but {len(series)} values were loaded."
        )

    return series, metadata


def list_series() -> list[tuple[str, str]]:
    """Return the dataset and identifier of every selected series."""

    selected_series = []

    for series_id in M3_SERIES_IDS:
        selected_series.append(("M3", series_id))

    for series_id in M4_SERIES_IDS:
        selected_series.append(("M4", series_id))

    return selected_series


def split_series(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a series chronologically into train, validation and test."""

    minimum_length = (
        MINIMUM_TRAINING_OBSERVATIONS
        + VALIDATION_HORIZON
        + FORECAST_HORIZON
    )

    if len(series) < minimum_length:
        raise ValueError(
            f"The series has {len(series)} observations, "
            f"but at least {minimum_length} are required."
        )

    test_start = len(series) - FORECAST_HORIZON
    validation_start = test_start - VALIDATION_HORIZON

    train = series.iloc[:validation_start].reset_index(drop=True)
    validation = series.iloc[validation_start:test_start].reset_index(drop=True)
    test = series.iloc[test_start:].reset_index(drop=True)

    return train, validation, test


def combine_history(
    train: pd.Series,
    validation: pd.Series,
) -> pd.Series:
    """Join train and validation before the final model fit."""

    return pd.concat([train, validation], ignore_index=True)


def seasonal_naive_forecast(
    history: pd.Series,
    horizon: int,
) -> np.ndarray:
    """Repeat the observations from the same season of the previous year."""

    if len(history) < SEASONAL_PERIOD:
        raise ValueError("Seasonal naive needs at least 12 observations.")

    last_year = history.iloc[-SEASONAL_PERIOD:].to_numpy(dtype=float)
    forecast = []

    for step in range(horizon):
        forecast.append(last_year[step % SEASONAL_PERIOD])

    return np.asarray(forecast, dtype=float)


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    history: pd.Series,
) -> dict[str, float]:
    """Calculate MAE, RMSE, MAPE, sMAPE and MASE."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have the same length.")

    if len(actual) == 0:
        raise ValueError("Metric vectors cannot be empty.")

    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Metric vectors must contain only finite values.")

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    non_zero_actual = actual != 0
    if np.any(non_zero_actual):
        percentage_errors = np.abs(
            errors[non_zero_actual] / actual[non_zero_actual]
        )
        mape = float(np.mean(percentage_errors) * 100)
    else:
        mape = np.nan

    smape_sum = 0.0

    for index in range(len(actual)):
        denominator = (
            abs(actual[index]) + abs(predicted[index])
        ) / 2

        if denominator != 0:
            smape_sum += abs(errors[index]) / denominator

    smape = float(smape_sum / len(actual) * 100)

    scale = _mase_scale(history)
    mase = float(mae / scale) if np.isfinite(scale) else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "mase": mase,
    }


def _mase_scale(history: pd.Series) -> float:
    """Calculate the mean in-sample error of the seasonal naive method."""

    values = history.to_numpy(dtype=float)

    if len(values) <= SEASONAL_PERIOD:
        return np.nan

    seasonal_errors = np.abs(
        values[SEASONAL_PERIOD:] - values[:-SEASONAL_PERIOD]
    )
    scale = float(np.mean(seasonal_errors))

    if scale == 0:
        return np.nan

    return scale