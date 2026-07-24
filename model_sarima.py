"""SARIMA forecasting model."""

import numpy as np
import pmdarima as pm
import pandas as pd

from data_utils import FORECAST_HORIZON, ModelResult, compute_metrics


def run_sarima(series: pd.Series) -> ModelResult:
    """Fit SARIMA and forecast the final test window."""
    train = series[:-FORECAST_HORIZON]
    test = series[-FORECAST_HORIZON:]

    fitted_model = pm.auto_arima(
        train,
        start_p=1,
        start_q=1,
        test="adf",
        max_p=3,
        max_q=3,
        m=12,
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    fitted_model = fitted_model.fit(train)

    fitted_values = np.asarray(fitted_model.predict_in_sample(), dtype=float)
    forecast, confidence_interval = fitted_model.predict(
        n_periods=FORECAST_HORIZON,
        return_conf_int=True,
    )

    actual = test.to_numpy(dtype=float)
    predicted = np.asarray(forecast, dtype=float)
    metrics = compute_metrics(actual, predicted)

    return {
        "forecast": predicted,
        "actual": actual,
        "metrics": metrics,
        "fitted_values": fitted_values,
        "confidence_interval": np.asarray(confidence_interval, dtype=float),
        "model_order": fitted_model.order,
        "seasonal_order": fitted_model.seasonal_order,
        "fitted_model": fitted_model,
    }
