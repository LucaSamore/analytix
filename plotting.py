"""Essential plots for the forecasting experiment."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import FORECAST_HORIZON
from model_wavelet import COMPONENT_NAMES


def plot_series(series: pd.Series, series_name: str) -> None:
    """Show the complete observed series."""

    plt.figure(figsize=(12, 4))
    plt.plot(series.to_numpy(dtype=float))
    plt.title(f"Series {series_name}")
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


def plot_wavelet_components(
    series: pd.Series,
    series_name: str,
    result: dict[str, Any],
) -> None:
    """Show the pre-test series and its three Haar components."""

    history = series.iloc[:-FORECAST_HORIZON]
    components = result["components"]

    _, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(history.to_numpy(dtype=float), color="black")
    axes[0].set_title(f"Pre-test history - {series_name}")

    for index, component_name in enumerate(COMPONENT_NAMES):
        axes[index + 1].plot(components[component_name])
        axes[index + 1].set_title(component_name)

    axes[-1].set_xlabel("Observation")
    plt.tight_layout()
    plt.show()


def plot_forecast_comparison(
    series: pd.Series,
    series_name: str,
    results: dict[str, dict[str, Any]],
) -> None:
    """Compare all final forecasts against the same test observations."""

    history_length = len(series) - FORECAST_HORIZON
    forecast_index = np.arange(history_length, len(series))

    plt.figure(figsize=(14, 6))
    plt.plot(
        np.arange(len(series)),
        series.to_numpy(dtype=float),
        color="black",
        label="Actual series",
    )

    for model_name, result in results.items():
        forecast = np.asarray(result["forecast"], dtype=float)
        plt.plot(
            forecast_index,
            forecast,
            marker="o",
            label=model_name,
        )

    plt.axvline(
        history_length - 0.5,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Final test starts",
    )
    plt.title(f"Forecast comparison - Series {series_name}")
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_sarima_diagnostics(result: dict[str, Any]) -> None:
    """Show the standard residual diagnostics of the selected SARIMA."""

    fitted_model = result["fitted_model"]
    figure = fitted_model.plot_diagnostics(figsize=(12, 8))
    residual_axis, density_axis, qq_axis, acf_axis = figure.axes

    figure.suptitle("Diagnostica dei residui SARIMA - Dati precedenti al test")

    residual_axis.set_title("Residui standardizzati nel tempo")
    residual_axis.set_xlabel("Osservazione storica (ordine temporale)")
    residual_axis.set_ylabel("Residuo standardizzato")

    density_axis.set_title("Istogramma e densità stimata")
    density_axis.set_xlabel("Residuo standardizzato")
    density_axis.set_ylabel("Densità")
    density_axis.legend(
        ["Istogramma", "Densità stimata (KDE)", "Normale N(0,1)"]
    )

    qq_axis.set_title("Q-Q rispetto alla distribuzione normale")
    qq_axis.set_xlabel("Quantili teorici della distribuzione normale")
    qq_axis.set_ylabel("Quantili osservati dei residui")

    acf_axis.set_title("Correlogramma dei residui")
    acf_axis.set_xlabel("Ritardo temporale (mesi)")
    acf_axis.set_ylabel("Autocorrelazione dei residui")

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
