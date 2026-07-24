"""XGBoost with lag features, validation and recursive forecasting."""

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from data_utils import (
    FORECAST_HORIZON,
    VALIDATION_HORIZON,
    combine_history,
    compute_metrics,
    split_series,
)


XGBoostConfiguration = dict[str, str | int | float]

XGBOOST_CONFIGURATIONS: list[XGBoostConfiguration] = [
    {
        "name": "100 trees, depth 2",
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.05,
    },
    {
        "name": "200 trees, depth 2",
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.05,
    },
    {
        "name": "200 trees, depth 3",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
    },
]


def create_lagged_dataset(
    values: pd.Series,
    look_back: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a time series into rows of lagged inputs and target values."""

    array = values.to_numpy(dtype=float)
    features = []
    targets = []

    for index in range(len(array) - look_back):
        features.append(array[index:index + look_back])
        targets.append(array[index + look_back])

    if not features:
        raise ValueError("The series is too short for the selected look-back.")

    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
    )


def make_xgboost_model(
    configuration: XGBoostConfiguration,
) -> XGBRegressor:
    """Create one reproducible XGBoost model."""

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(configuration["n_estimators"]),
        max_depth=int(configuration["max_depth"]),
        learning_rate=float(configuration["learning_rate"]),
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def fit_xgboost(
    history: pd.Series,
    configuration: XGBoostConfiguration,
    look_back: int = 12,
) -> XGBRegressor:
    """Fit XGBoost on the lagged rows available in the history."""

    features, targets = create_lagged_dataset(history, look_back)
    model = make_xgboost_model(configuration)
    model.fit(features, targets)

    return model


def recursive_forecast(
    model: XGBRegressor,
    history: pd.Series,
    horizon: int,
    look_back: int = 12,
) -> np.ndarray:
    """Forecast repeatedly, feeding every new prediction back into the input."""

    if len(history) < look_back:
        raise ValueError("Not enough observations for recursive forecasting.")

    input_window = history.iloc[-look_back:].to_numpy(dtype=float)
    forecast = []

    for _ in range(horizon):
        model_input = input_window.reshape(1, look_back)
        next_value = float(model.predict(model_input)[0])
        forecast.append(next_value)

        input_window = np.roll(input_window, -1)
        input_window[-1] = next_value

    return np.asarray(forecast, dtype=float)


def select_xgboost_configuration(
    train: pd.Series,
    validation: pd.Series,
    look_back: int = 12,
) -> XGBoostConfiguration:
    """Choose the configuration with the lowest validation MASE."""

    best_configuration = XGBOOST_CONFIGURATIONS[0]
    best_mase = np.inf

    for configuration in XGBOOST_CONFIGURATIONS:
        model = fit_xgboost(train, configuration, look_back)
        forecast = recursive_forecast(
            model,
            train,
            len(validation),
            look_back,
        )
        metrics = compute_metrics(
            validation.to_numpy(dtype=float),
            forecast,
            train,
        )

        if metrics["mase"] < best_mase:
            best_mase = metrics["mase"]
            best_configuration = configuration

    return best_configuration.copy()


def select_xgboost_configuration_from_history(
    history: pd.Series,
    look_back: int = 12,
) -> XGBoostConfiguration:
    """Select XGBoost using an internal validation at the end of a history."""

    if len(history) <= VALIDATION_HORIZON + look_back:
        raise ValueError("Not enough history to validate XGBoost.")

    train = history.iloc[:-VALIDATION_HORIZON].reset_index(drop=True)
    validation = history.iloc[-VALIDATION_HORIZON:].reset_index(drop=True)
    configuration = select_xgboost_configuration(
        train,
        validation,
        look_back,
    )

    return configuration


def forecast_xgboost_one_step(
    history: pd.Series,
    configuration: XGBoostConfiguration,
    look_back: int = 12,
) -> float:
    """Produce one forecast for the rolling Diebold-Mariano experiment."""

    model = fit_xgboost(history, configuration, look_back)
    forecast = recursive_forecast(model, history, 1, look_back)

    return float(forecast[0])


def run_xgboost(
    series: pd.Series,
    look_back: int = 12,
) -> dict[str, Any]:
    """Select XGBoost on validation and evaluate it on the final test."""

    train, validation, test = split_series(series)
    configuration = select_xgboost_configuration(
        train,
        validation,
        look_back,
    )

    history = combine_history(train, validation)
    model = fit_xgboost(history, configuration, look_back)
    forecast = recursive_forecast(
        model,
        history,
        FORECAST_HORIZON,
        look_back,
    )
    actual = test.to_numpy(dtype=float)

    return {
        "forecast": forecast,
        "metrics": compute_metrics(actual, forecast, history),
        "configuration": str(configuration["name"]),
    }
