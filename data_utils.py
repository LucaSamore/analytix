"""Shared utilities for the M3/M4 forecasting comparison.

Contains: data loading for the M3 and M4 series used in the experiments,
forecast accuracy metrics, and the Diebold-Mariano test used to check
whether the differences between models are statistically significant.
"""

from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent
M3_FILE = PROJECT_DIR / "M3C_monthly.csv"
M4_FILE = PROJECT_DIR / "M4_monthly_subset.csv"
FORECAST_HORIZON = 12

type DatasetName = Literal["M3", "M4"]
type SeriesData = tuple[pd.Series, pd.Series]
type SeriesLoader = Callable[[str], SeriesData]
type SeriesEntry = tuple[DatasetName, str, SeriesLoader]
type Metrics = dict[str, float]
type ModelResult = dict[str, Any]

# Picked with a fixed random seed (42), stratified by category, so the sample is
# reproducible and not cherry-picked.
M3_SERIES_IDS = [
    "N1830", "N1842",   # MICRO
    "N1901", "N2185",   # INDUSTRY
    "N2438", "N2219",   # MACRO
    "N2591", "N2662",   # FINANCE
    "N2745",            # DEMOGRAPHIC
    "N2797",            # OTHER
]

# 5 M4 monthly series, one per domain
M4_SERIES_IDS = ["M15716", "M27126", "M233", "M43445", "M26707"]


def _load_row(data_file: Path, series_id: str) -> SeriesData:
    data = pd.read_csv(data_file)
    data["Series"] = data["Series"].str.strip()
    row = data.loc[data["Series"] == series_id]

    if row.empty:
        raise ValueError(f"Series {series_id!r} not found in {data_file.name}")

    row = row.iloc[0]
    series_metadata = row.iloc[:6].copy()
    series_metadata["Category"] = str(series_metadata["Category"]).strip()
    series_values = row.iloc[6:].dropna().to_numpy(dtype=float)
    series = pd.Series(series_values)

    return series, series_metadata


def load_m3_series(series_id: str) -> SeriesData:
    return _load_row(M3_FILE, series_id)


def load_m4_series(series_id: str) -> SeriesData:
    return _load_row(M4_FILE, series_id)


def list_series() -> list[SeriesEntry]:
    entries: list[SeriesEntry] = [
        ("M3", series_id, load_m3_series) for series_id in M3_SERIES_IDS
    ]
    entries += [("M4", sid, load_m4_series) for sid in M4_SERIES_IDS]
    return entries


def compute_metrics(actual: ArrayLike, predicted: ArrayLike) -> Metrics:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def diebold_mariano_test(
    actual: ArrayLike, forecast_a: ArrayLike, forecast_b: ArrayLike,
    h: int = 1, power: int = 2,
) -> tuple[float, float]:
    """Diebold-Mariano test comparing the forecast accuracy of two models.

    H0: the two forecasts have equal predictive accuracy (expected loss
    differential = 0). Returns (dm_statistic, p_value); p < 0.05 means the
    accuracy difference is unlikely to be due to chance.

    `h` is the forecast horizon and controls how many autocorrelation lags
    are included in the variance estimate (Harvey, Leybourne & Newbold,
    1997 small-sample correction). With only FORECAST_HORIZON=12 test
    points, h=1 (no autocorrelation correction) is the numerically stable
    default; increase it only if you have longer test windows.

    References: Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
    """
    actual = np.asarray(actual, dtype=float)
    forecast_a = np.asarray(forecast_a, dtype=float)
    forecast_b = np.asarray(forecast_b, dtype=float)

    loss_a = np.abs(actual - forecast_a) ** power
    loss_b = np.abs(actual - forecast_b) ** power
    diff = loss_a - loss_b

    n = len(diff)
    mean_diff = np.mean(diff)

    gamma_0 = np.var(diff, ddof=0)
    variance = gamma_0
    for lag in range(1, h):
        gamma_lag = np.mean((diff[lag:] - mean_diff) * (diff[:-lag] - mean_diff))
        variance += 2 * gamma_lag
    variance /= n

    if variance <= 0:
        return 0.0, 1.0

    dm_stat = mean_diff / np.sqrt(variance)

    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))

    return float(dm_stat), float(p_value)
