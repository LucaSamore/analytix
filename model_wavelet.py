"""Wavelet decomposition + AutoReg forecasting model."""

import numpy as np
import pywt
from statsmodels.tsa.ar_model import AutoReg

from data_utils import FORECAST_HORIZON, compute_metrics


def reconstruct_component(wavelet_coefficients, selected_index, wavelet_name, mode, original_length):
    component_coefficients = []

    for index in range(len(wavelet_coefficients)):
        if index == selected_index:
            component_coefficients.append(wavelet_coefficients[index].copy())
        else:
            component_coefficients.append(np.zeros_like(wavelet_coefficients[index]))

    component = pywt.waverec(component_coefficients, wavelet_name, mode=mode)

    return component[:original_length]


def forecast_component_autoreg(values, horizon, lag):
    model = AutoReg(values, lags=lag, old_names=False)
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)

    return np.asarray(forecast)


def run_wavelet_autoreg(series, series_name, show_plots=True):
    train = series[:-FORECAST_HORIZON]
    test = series[-FORECAST_HORIZON:]

    print(
        f"\nWavelet split: series={len(series)}, "
        f"train={len(train)}, test={len(test)}"
    )

    wavelet_name = "haar"
    requested_level = 2
    mode = "periodization"

    wavelet = pywt.Wavelet(wavelet_name)
    maximum_level = pywt.dwt_max_level(len(train), wavelet.dec_len)
    level = min(requested_level, maximum_level)

    wavelet_coefficients = pywt.wavedec(
        train.to_numpy(copy=True),
        wavelet_name,
        mode=mode,
        level=level,
    )

    component_names = [f"A{level}"]
    for detail_level in range(level, 0, -1):
        component_names.append(f"D{detail_level}")

    print(f"Wavelet: {wavelet_name}")
    print(f"Requested level: {requested_level}")
    print(f"Used level: {level}")
    print(f"Components: {component_names}")

    components = {}
    for index, component_name in enumerate(component_names):
        components[component_name] = reconstruct_component(
            wavelet_coefficients, index, wavelet_name, mode, len(train)
        )

    reconstructed_train = np.zeros(len(train))
    for component_name in component_names:
        reconstructed_train += components[component_name]

    reconstruction_error = np.max(np.abs(reconstructed_train - train.values))
    print(f"Maximum reconstruction error: {reconstruction_error:.10f}")

    forecast_lag = 12
    component_forecasts = {}
    forecast = np.zeros(FORECAST_HORIZON)

    for component_name in component_names:
        component_forecasts[component_name] = forecast_component_autoreg(
            components[component_name], FORECAST_HORIZON, forecast_lag
        )
        forecast += component_forecasts[component_name]

    forecast_index = np.arange(len(train), len(series))
    actual = test.values
    metrics = compute_metrics(actual, forecast)

    print("\nWavelet + AutoReg forecast:")
    print(forecast)
    print("\nWavelet + AutoReg performance on the test set:")
    print(f"  MAE  = {metrics['mae']:.3f}")
    print(f"  RMSE = {metrics['rmse']:.3f}")
    print(f"  MAPE = {metrics['mape']:.2f}%")

    if show_plots:
        import matplotlib.pyplot as plt

        number_of_plots = len(component_names) + 1
        figure_height = 2.4 * number_of_plots
        _, axes = plt.subplots(number_of_plots, 1, figsize=(12, figure_height), sharex=True)

        axes[0].plot(train.values, color="black")
        axes[0].set_title("Train")
        axes[0].set_ylabel("Value")

        for index, component_name in enumerate(component_names):
            axes[index + 1].plot(components[component_name])
            axes[index + 1].set_title(component_name)
            axes[index + 1].set_ylabel("Value")

        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

        _, axes = plt.subplots(len(component_names), 1, figsize=(12, 2.4 * len(component_names)), sharex=True)
        if len(component_names) == 1:
            axes = [axes]

        for index, component_name in enumerate(component_names):
            axes[index].plot(np.arange(len(train)), components[component_name], label=f"{component_name} train")
            axes[index].plot(forecast_index, component_forecasts[component_name], "-o", label=f"{component_name} forecast")
            axes[index].set_title(component_name)
            axes[index].legend()

        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.plot(series.values, label="Actual series")
        plt.plot(forecast_index, forecast, "-o", label="Wavelet + AutoReg 12-month forecast")
        plt.axvline(len(train) - 1, color="black", linestyle="--", alpha=0.5, label="Train/test split")
        plt.title(f"Wavelet + AutoReg ({wavelet_name}, level={level}) - Series {series_name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    return {
        "forecast": forecast,
        "actual": actual,
        "metrics": metrics,
        "wavelet_name": wavelet_name,
        "level": level,
    }
