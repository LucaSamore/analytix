"""Plots for exploratory analysis and model results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from data_utils import FORECAST_HORIZON, ModelResult


def plot_exploratory_analysis(series: pd.Series, series_name: str) -> None:
    """Plot the series, its distribution, and its seasonal decomposition."""
    sns.set_theme(style="darkgrid", context="talk", palette="deep")

    _, axes = plt.subplots(3, 1, figsize=(8, 16))

    sns.lineplot(x=range(len(series)), y=series, ax=axes[0])
    axes[0].set(title=f"Series {series_name}", xlabel="Observation", ylabel="Value")

    sns.histplot(series, kde=True, ax=axes[1])
    axes[1].set(title="Distribution", xlabel="Value", ylabel="Count")

    sns.boxplot(y=series, ax=axes[2])
    sns.stripplot(y=series, ax=axes[2], color=".25", alpha=0.4, size=3)
    axes[2].set(title="Boxplot", ylabel="Value")

    sns.despine()
    plt.tight_layout()
    plt.show()

    decomposition = seasonal_decompose(series, model="additive", period=12)
    figure = decomposition.plot()
    figure.set_size_inches(8, 6)
    plt.suptitle("Additive decomposition")
    plt.show()


def plot_sarima(series: pd.Series, series_name: str, result: ModelResult) -> None:
    """Plot a SARIMA forecast, confidence interval, and diagnostics."""
    train = series[:-FORECAST_HORIZON]
    test = series[-FORECAST_HORIZON:]
    train_index = np.arange(len(train))
    forecast_index = np.arange(len(train), len(series))

    plt.figure(figsize=(15, 5))
    plt.plot(train_index, train.to_numpy(), label="Train history", color="C0")
    plt.plot(
        forecast_index, test.to_numpy(), label="Actual test values",
        color="C0", linestyle="--",
    )
    plt.plot(
        train_index, result["fitted_values"], label="Fitted in-sample values",
        color="C1", alpha=0.7,
    )
    plt.plot(forecast_index, result["forecast"], label="SARIMA forecast", color="C3")
    plt.fill_between(
        forecast_index, result["confidence_interval"][:, 0],
        result["confidence_interval"][:, 1], color="C3", alpha=0.15,
        label="95% confidence interval",
    )
    plt.title(
        f"SARIMA{result['model_order']}{result['seasonal_order']} - Series {series_name}"
    )
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.legend(loc="upper left", fontsize=11)
    sns.despine()
    plt.tight_layout()
    plt.show()

    result["fitted_model"].plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.show()


def plot_xgboost(series: pd.Series, series_name: str, result: ModelResult) -> None:
    """Plot an XGBoost forecast and its lag-feature importances."""
    forecast_index = range(len(series) - FORECAST_HORIZON, len(series))

    plt.figure(figsize=(12, 5))
    plt.plot(series.to_numpy(), label="Actual series")
    plt.plot(forecast_index, result["forecast"], "-o", label="12-month forecast")
    plt.title(f"XGBoost - Series {series_name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    lag_names = [
        f"lag_{result['look_back'] - index}" for index in range(result["look_back"])
    ]

    plt.figure(figsize=(10, 4))
    plt.bar(lag_names, result["feature_importances"])
    plt.title("XGBoost lag-feature importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_wavelet_autoreg(
    series: pd.Series, series_name: str, result: ModelResult
) -> None:
    """Plot the wavelet decomposition, components, and combined forecast."""
    train = series[:-FORECAST_HORIZON]
    forecast_index = np.arange(len(train), len(series))

    number_of_plots = len(result["component_names"]) + 1
    figure_height = 2.4 * number_of_plots
    _, axes = plt.subplots(
        number_of_plots, 1, figsize=(12, figure_height), sharex=True
    )

    axes[0].plot(train.to_numpy(), color="black")
    axes[0].set_title("Train")
    axes[0].set_ylabel("Value")

    for index, component_name in enumerate(result["component_names"]):
        axes[index + 1].plot(result["components"][component_name])
        axes[index + 1].set_title(component_name)
        axes[index + 1].set_ylabel("Value")

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    _, component_axes = plt.subplots(
        len(result["component_names"]), 1,
        figsize=(12, 2.4 * len(result["component_names"])), sharex=True,
    )
    component_axes = np.atleast_1d(component_axes)

    for index, component_name in enumerate(result["component_names"]):
        component_axes[index].plot(
            np.arange(len(train)), result["components"][component_name],
            label=f"{component_name} train",
        )
        component_axes[index].plot(
            forecast_index, result["component_forecasts"][component_name], "-o",
            label=f"{component_name} forecast",
        )
        component_axes[index].set_title(component_name)
        component_axes[index].legend()

    component_axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(series.to_numpy(), label="Actual series")
    plt.plot(forecast_index, result["forecast"], "-o",
             label="Wavelet + AutoReg 12-month forecast")
    plt.axvline(
        len(train) - 1, color="black", linestyle="--", alpha=0.5,
        label="Train/test split",
    )
    plt.title(
        f"Wavelet + AutoReg ({result['wavelet_name']}, level={result['level']})"
        f" - Series {series_name}"
    )
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
