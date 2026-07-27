"""Simple reference models used to interpret the main results."""

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from data_utils import (
    FORECAST_HORIZON,
    combine_history,
    compute_metrics,
    seasonal_naive_forecast,
    split_series,
)


def run_seasonal_naive(series: pd.Series) -> dict[str, Any]:
    """Forecast the test set with the previous year's observations."""

    train, validation, test = split_series(series)
    history = combine_history(train, validation)

    forecast = seasonal_naive_forecast(history, FORECAST_HORIZON)
    actual = test.to_numpy(dtype=float)

    return {
        "forecast": forecast,
        "metrics": compute_metrics(actual, forecast, history),
        "configuration": "seasonal period = 12",
    }


def run_autoreg(
    series: pd.Series,
    lag: int = 12,
) -> dict[str, Any]:
    """Forecast the test set with AutoReg on the original series."""

    train, validation, test = split_series(series)
    history = combine_history(train, validation)

    forecast = _autoreg_forecast(history, FORECAST_HORIZON, lag)
    actual = test.to_numpy(dtype=float)

    return {
        "forecast": forecast,
        "metrics": compute_metrics(actual, forecast, history),
        "configuration": f"lag = {lag}",
    }


def _autoreg_forecast(
    history: pd.Series,
    horizon: int,
    lag: int = 12,
) -> np.ndarray:
    """Fit AutoReg to the original series and forecast future values."""

    model = AutoReg(history.to_numpy(dtype=float), lags=lag, old_names=False)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=horizon)

    return np.asarray(forecast, dtype=float)
