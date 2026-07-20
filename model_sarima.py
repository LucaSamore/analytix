"""SARIMA forecasting model"""

import numpy as np
import pmdarima as pm

from data_utils import FORECAST_HORIZON, compute_metrics


def run_sarima(series, series_name, show_plots=True):
    train = series[:-FORECAST_HORIZON]
    test = series[-FORECAST_HORIZON:]

    print(
        f"\nSARIMA split: series={len(series)}, "
        f"train={len(train)}, test={len(test)}"
    )

    model = pm.auto_arima(
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
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    print(model.summary())
    model_order = model.order
    seasonal_order = model.seasonal_order

    fitted = model.fit(train)
    fitted_values = fitted.predict_in_sample()
    forecast, confidence_interval = fitted.predict(
        n_periods=FORECAST_HORIZON,
        return_conf_int=True,
    )

    forecast_index = np.arange(len(train), len(series))
    actual = test.values
    predicted = np.asarray(forecast)
    metrics = compute_metrics(actual, predicted)

    print("\nSARIMA performance on the test set:")
    print(f"  MAE  = {metrics['mae']:.3f}")
    print(f"  RMSE = {metrics['rmse']:.3f}")
    print(f"  MAPE = {metrics['mape']:.2f}%")

    if show_plots:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(len(train)), train.values, label="Train history", color="C0")
        plt.plot(forecast_index, test.values, label="Actual test values", color="C0", linestyle="--")
        plt.plot(np.arange(len(train)), fitted_values, label="Fitted in-sample values", color="C1", alpha=0.7)
        plt.plot(forecast_index, forecast, label="SARIMA forecast", color="C3")
        plt.fill_between(
            forecast_index,
            confidence_interval[:, 0],
            confidence_interval[:, 1],
            color="C3",
            alpha=0.15,
            label="95% confidence interval",
        )
        plt.title(f"SARIMA{model_order}{seasonal_order} - Series {series_name}")
        plt.xlabel("Observation")
        plt.ylabel("Value")
        plt.legend(loc="upper left", fontsize=11)
        sns.despine()
        plt.tight_layout()
        plt.show()

        fitted.plot_diagnostics(figsize=(12, 8))
        plt.tight_layout()
        plt.show()

    return {
        "forecast": predicted,
        "actual": actual,
        "metrics": metrics,
        "model_order": model_order,
        "seasonal_order": seasonal_order,
    }
