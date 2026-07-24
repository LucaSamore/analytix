"""XGBoost forecasting model with lag features and recursive forecasting."""

from pandas.core.common import random_state
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from xgboost import XGBRegressor

from data_utils import FORECAST_HORIZON, ModelResult, compute_metrics


def create_dataset(
    dataset: ArrayLike, look_back: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert a one-dimensional series into lag features and targets."""
    values = np.asarray(dataset, dtype=float)
    data_x: list[NDArray[np.float64]] = []
    data_y: list[float] = []

    for index in range(len(values) - look_back):
        data_x.append(values[index : index + look_back])
        data_y.append(float(values[index + look_back]))

    return np.asarray(data_x, dtype=float), np.asarray(data_y, dtype=float)


def run_xgboost(series: pd.Series, look_back: int = 12) -> ModelResult:
    """Fit XGBoost and recursively forecast the final test window."""
    features, targets = create_dataset(series.to_numpy(dtype=float), look_back)

    x_train = features[:-FORECAST_HORIZON]
    x_test = features[-FORECAST_HORIZON:]
    y_train = targets[:-FORECAST_HORIZON]
    y_test = targets[-FORECAST_HORIZON:]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
    )
    model.fit(x_train, y_train)

    input_window = x_test[0].copy()
    forecast_values: list[float] = []

    for _ in range(FORECAST_HORIZON):
        next_value = float(model.predict(input_window.reshape(1, look_back))[0])
        forecast_values.append(next_value)
        input_window = np.roll(input_window, -1)
        input_window[-1] = next_value

    forecast = np.asarray(forecast_values, dtype=float)
    metrics = compute_metrics(y_test, forecast)

    return {
        "forecast": forecast,
        "actual": np.asarray(y_test, dtype=float),
        "metrics": metrics,
        "feature_importances": np.asarray(model.feature_importances_, dtype=float),
        "look_back": look_back,
    }
