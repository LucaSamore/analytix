"""Rolling forecasts and Diebold-Mariano comparisons."""

from typing import Any

import numpy as np
import pandas as pd

import data_utils
from dm_test import dm_test
from model_sarima import (
    forecast_sarima_one_step,
    select_sarima_structure,
)
from model_wavelet import forecast_wavelet_one_step
from model_xgboost import (
    forecast_xgboost_one_step,
    select_xgboost_configuration_from_history,
)


WAVELET_MODEL = "Wavelet+AutoReg"
COMPARISON_MODELS = ["SARIMA", "XGBoost"]


def rolling_forecasts(
    series: pd.Series,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Generate 24 one-step forecasts from equal rolling origins."""

    first_forecast = len(series) - data_utils.DM_EVALUATION_POINTS
    initial_history = series.iloc[:first_forecast].reset_index(drop=True)

    sarima_structure = select_sarima_structure(initial_history)
    xgboost_configuration = (
        select_xgboost_configuration_from_history(initial_history)
    )

    actual_values = []
    forecasts = {
        WAVELET_MODEL: [],
        "SARIMA": [],
        "XGBoost": [],
    }

    for position in range(first_forecast, len(series)):
        history = series.iloc[:position].reset_index(drop=True)
        actual_values.append(float(series.iloc[position]))

        forecasts[WAVELET_MODEL].append(
            forecast_wavelet_one_step(history)
        )
        forecasts["SARIMA"].append(
            forecast_sarima_one_step(history, sarima_structure)
        )
        forecasts["XGBoost"].append(
            forecast_xgboost_one_step(
                history,
                xgboost_configuration,
            )
        )

    return np.asarray(actual_values, dtype=float), forecasts


def compare_wavelet_with_models(
    dataset: str,
    series_name: str,
    category: str,
    actual: np.ndarray,
    forecasts: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Compare the wavelet forecasts with SARIMA and XGBoost."""

    rows = []
    wavelet_forecast = np.asarray(forecasts[WAVELET_MODEL], dtype=float)

    for other_model in COMPARISON_MODELS:
        other_forecast = np.asarray(
            forecasts[other_model],
            dtype=float,
        )

        dm_result = dm_test(
            actual.tolist(),
            wavelet_forecast.tolist(),
            other_forecast.tolist(),
            h=1,
            crit="MSE",
        )

        wavelet_mse = float(
            np.mean((actual - wavelet_forecast) ** 2)
        )
        other_mse = float(
            np.mean((actual - other_forecast) ** 2)
        )

        if wavelet_mse < other_mse:
            lower_loss_model = WAVELET_MODEL
        elif other_mse < wavelet_mse:
            lower_loss_model = other_model
        else:
            lower_loss_model = "Tie"

        is_significant = bool(dm_result.p_value < 0.05)

        if is_significant:
            conclusion = (
                f"{lower_loss_model} significantly better"
            )
        else:
            conclusion = "No significant difference"

        rows.append({
            "dataset": dataset,
            "series": series_name,
            "category": category,
            "model_a": WAVELET_MODEL,
            "model_b": other_model,
            "dm_statistic": float(dm_result.DM),
            "p_value": float(dm_result.p_value),
            "significant_5pct": is_significant,
            "lower_average_loss": lower_loss_model,
            "conclusion": conclusion,
        })

    return rows


def run_dm_for_series(
    dataset: str,
    series_name: str,
    category: str,
    series: pd.Series,
) -> list[dict[str, Any]]:
    """Run the complete DM experiment for one series."""

    actual, forecasts = rolling_forecasts(series)

    return compare_wavelet_with_models(
        dataset,
        series_name,
        category,
        actual,
        forecasts,
    )
