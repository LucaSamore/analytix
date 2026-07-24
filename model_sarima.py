"""SARIMA forecasting with automatic order selection."""

from typing import Any

import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.arima import ARIMA

from data_utils import (
    FORECAST_HORIZON,
    SEASONAL_PERIOD,
    combine_history,
    compute_metrics,
    split_series,
)


SarimaStructure = tuple[
    tuple[int, int, int],
    tuple[int, int, int, int],
    bool,
]


def fit_auto_sarima(history: pd.Series) -> ARIMA:
    """Select and fit SARIMA using only observations available at that time."""

    return pm.auto_arima(
        history.to_numpy(dtype=float),
        start_p=1,
        start_q=1,
        test="adf",
        max_p=3,
        max_q=3,
        m=SEASONAL_PERIOD,
        start_P=0,
        seasonal=True,
        d=None,
        D=None,
        seasonal_test="ocsb",
        information_criterion="aicc",
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )


def select_sarima_structure(history: pd.Series) -> SarimaStructure:
    """Select the SARIMA orders once before a rolling experiment."""

    model = fit_auto_sarima(history)

    return model.order, model.seasonal_order, bool(model.with_intercept)


def sarima_forecast_with_structure(
    history: pd.Series,
    horizon: int,
    structure: SarimaStructure,
) -> np.ndarray:
    """Refit a previously selected SARIMA structure and forecast."""

    order, seasonal_order, with_intercept = structure

    model = ARIMA(
        order=order,
        seasonal_order=seasonal_order,
        with_intercept=with_intercept,
        suppress_warnings=True,
    )
    fitted_model = model.fit(history.to_numpy(dtype=float))
    forecast = fitted_model.predict(n_periods=horizon)

    return np.asarray(forecast, dtype=float)


def forecast_sarima_one_step(
    history: pd.Series,
    structure: SarimaStructure,
) -> float:
    """Produce one forecast for the rolling Diebold-Mariano experiment."""

    forecast = sarima_forecast_with_structure(history, 1, structure)
    return float(forecast[0])


def run_sarima(series: pd.Series) -> dict[str, Any]:
    """Fit SARIMA on pre-test observations and forecast the final year."""

    train, validation, test = split_series(series)
    history = combine_history(train, validation)

    model = fit_auto_sarima(history)
    forecast = model.predict(n_periods=FORECAST_HORIZON)

    actual = test.to_numpy(dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    configuration = (
        f"order={model.order}, "
        f"seasonal_order={model.seasonal_order}"
    )

    return {
        "forecast": forecast,
        "metrics": compute_metrics(actual, forecast, history),
        "fitted_model": model,
        "configuration": configuration,
    }
