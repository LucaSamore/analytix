"""Haar decomposition followed by AutoReg on each component."""

from typing import Any

import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.ar_model import AutoReg

from data_utils import (
    FORECAST_HORIZON,
    combine_history,
    compute_metrics,
    split_series,
)


WAVELET_NAME = "haar"
WAVELET_LEVEL = 2
WAVELET_MODE = "periodization"
AUTOREG_LAG = 12
COMPONENT_NAMES = ["A2", "D2", "D1"]


def run_wavelet_autoreg(series: pd.Series) -> dict[str, Any]:
    """Evaluate the fixed Haar level-2 configuration on the final test."""

    train, validation, test = split_series(series)
    history = combine_history(train, validation)
    forecast, components, reconstruction_error = (
        _wavelet_autoreg_forecast(history, FORECAST_HORIZON)
    )

    actual = test.to_numpy(dtype=float)

    configuration = (
        f"{WAVELET_NAME}, level={WAVELET_LEVEL}, "
        f"mode={WAVELET_MODE}, AutoReg lag={AUTOREG_LAG}"
    )

    return {
        "forecast": forecast,
        "metrics": compute_metrics(actual, forecast, history),
        "components": components,
        "reconstruction_error": reconstruction_error,
        "configuration": configuration,
    }


def forecast_wavelet_one_step(history: pd.Series) -> float:
    """Produce one forecast for the rolling Diebold-Mariano experiment."""

    forecast, _, _ = _wavelet_autoreg_forecast(history, 1)

    return float(forecast[0])


def _wavelet_autoreg_forecast(
    history: pd.Series,
    horizon: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Forecast every wavelet component and sum their forecasts."""

    components, reconstruction_error = _decompose_history(history)
    final_forecast = np.zeros(horizon)

    for component_name in COMPONENT_NAMES:
        component_forecast = _forecast_component(
            components[component_name],
            horizon,
        )
        final_forecast += component_forecast

    return final_forecast, components, reconstruction_error


def _decompose_history(
    history: pd.Series,
) -> tuple[dict[str, np.ndarray], float]:
    """Decompose the history into the reconstructed A2, D2 and D1 series."""

    values = history.to_numpy(dtype=float).copy()
    coefficients = pywt.wavedec(
        values,
        WAVELET_NAME,
        mode=WAVELET_MODE,
        level=WAVELET_LEVEL,
    )

    components = {}

    for index, component_name in enumerate(COMPONENT_NAMES):
        components[component_name] = _reconstruct_component(
            coefficients,
            index,
            len(values),
        )

    reconstructed_history = np.zeros(len(values))

    for component_name in COMPONENT_NAMES:
        reconstructed_history += components[component_name]

    reconstruction_error = float(
        np.max(np.abs(reconstructed_history - values))
    )

    return components, reconstruction_error


def _reconstruct_component(
    coefficients: list[np.ndarray],
    selected_index: int,
    original_length: int,
) -> np.ndarray:
    """Reconstruct one component while all other coefficients are zero."""

    selected_coefficients = []

    for index, coefficient in enumerate(coefficients):
        if index == selected_index:
            selected_coefficients.append(coefficient.copy())
        else:
            selected_coefficients.append(np.zeros_like(coefficient))

    component = pywt.waverec(
        selected_coefficients,
        WAVELET_NAME,
        mode=WAVELET_MODE,
    )

    return np.asarray(component[:original_length], dtype=float)


def _forecast_component(
    component: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Forecast one reconstructed component with AutoReg(12)."""

    model = AutoReg(component, lags=AUTOREG_LAG, old_names=False)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=horizon)

    return np.asarray(forecast, dtype=float)
