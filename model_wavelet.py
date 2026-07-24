"""Wavelet decomposition + AutoReg forecasting model."""

import numpy as np
import pandas as pd
import pywt
from numpy.typing import ArrayLike, NDArray
from statsmodels.tsa.ar_model import AutoReg

from data_utils import FORECAST_HORIZON, ModelResult, compute_metrics


def reconstruct_component(
    wavelet_coefficients: list[NDArray[np.float64]],
    selected_index: int, wavelet_name: str, mode: str, original_length: int,
) -> NDArray[np.float64]:
    """Rebuild one component while zeroing every other coefficient array."""
    component_coefficients: list[NDArray[np.float64]] = []

    for index in range(len(wavelet_coefficients)):
        if index == selected_index:
            component_coefficients.append(wavelet_coefficients[index].copy())
        else:
            component_coefficients.append(np.zeros_like(wavelet_coefficients[index]))

    component = pywt.waverec(component_coefficients, wavelet_name, mode=mode)

    return np.asarray(component[:original_length], dtype=float)


def forecast_component_autoreg(
    values: ArrayLike, horizon: int, lag: int
) -> NDArray[np.float64]:
    """Forecast one reconstructed component with an autoregression."""
    model = AutoReg(values, lags=lag, old_names=False)
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)

    return np.asarray(forecast, dtype=float)


def run_wavelet_autoreg(series: pd.Series) -> ModelResult:
    """Decompose the training data and forecast each component."""
    train = series[:-FORECAST_HORIZON]
    test = series[-FORECAST_HORIZON:]

    wavelet_name = "haar"
    requested_level = 2
    mode = "periodization"

    wavelet = pywt.Wavelet(wavelet_name)
    maximum_level = pywt.dwt_max_level(len(train), wavelet.dec_len)
    level = min(requested_level, maximum_level)

    wavelet_coefficients = pywt.wavedec(
        train.to_numpy(copy=True), wavelet_name, mode=mode, level=level
    )

    component_names = [f"A{level}"]
    for detail_level in range(level, 0, -1):
        component_names.append(f"D{detail_level}")

    components: dict[str, NDArray[np.float64]] = {}
    for index, component_name in enumerate(component_names):
        components[component_name] = reconstruct_component(
            wavelet_coefficients, index, wavelet_name, mode, len(train))

    reconstructed_train = np.zeros(len(train))
    for component_name in component_names:
        reconstructed_train += components[component_name]

    reconstruction_error = np.max(np.abs(reconstructed_train - train.values))

    forecast_lag = 12
    component_forecasts: dict[str, NDArray[np.float64]] = {}
    forecast = np.zeros(FORECAST_HORIZON)

    for component_name in component_names:
        component_forecasts[component_name] = forecast_component_autoreg(
            components[component_name], FORECAST_HORIZON, forecast_lag)
        forecast += component_forecasts[component_name]

    actual = test.to_numpy(dtype=float)
    metrics = compute_metrics(actual, forecast)

    return {
        "forecast": np.asarray(forecast, dtype=float),
        "actual": actual,
        "metrics": metrics,
        "components": components,
        "component_forecasts": component_forecasts,
        "component_names": component_names,
        "wavelet_name": wavelet_name,
        "level": level,
        "reconstruction_error": float(reconstruction_error),
    }
