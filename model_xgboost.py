"""XGBoost forecasting model with lag features and recursive forecasting"""

import numpy as np
from xgboost import XGBRegressor

from data_utils import FORECAST_HORIZON, compute_metrics


def create_dataset(dataset, look_back=1):
    """Convert a one-dimensional series into lag features and targets."""
    data_x = []
    data_y = []

    for index in range(len(dataset) - look_back):
        data_x.append(dataset[index : index + look_back])
        data_y.append(dataset[index + look_back])

    return np.array(data_x), np.array(data_y)


def run_xgboost(series, series_name, show_plots=True):
    look_back = 12
    features, targets = create_dataset(series.values, look_back)

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
    forecast = []

    for _ in range(FORECAST_HORIZON):
        forecast.append(model.predict(input_window.reshape(1, look_back))[0])
        input_window = np.roll(input_window, -1)
        input_window[-1] = forecast[-1]

    forecast = np.array(forecast)
    metrics = compute_metrics(y_test, forecast)

    print("\nXGBoost performance on the test set:")
    print(f"  MAE  = {metrics['mae']:.3f}")
    print(f"  RMSE = {metrics['rmse']:.3f}")
    print(f"  MAPE = {metrics['mape']:.2f}%")

    if show_plots:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        plt.plot(series.values, label="Actual series")
        plt.plot(
            range(len(series) - FORECAST_HORIZON, len(series)),
            forecast,
            "-o",
            label="12-month forecast",
        )
        plt.title(f"XGBoost - Series {series_name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        importances = model.feature_importances_
        lag_names = [f"lag_{look_back - index}" for index in range(look_back)]

        plt.figure(figsize=(10, 4))
        plt.bar(lag_names, importances)
        plt.title("XGBoost lag-feature importance")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return {
        "forecast": forecast,
        "actual": y_test,
        "metrics": metrics,
    }
