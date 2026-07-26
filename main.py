#!/usr/bin/env python3
"""Run the forecasting comparison in demo or batch mode."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import data_utils
import dm_analysis
import plotting
from model_baselines import run_autoreg, run_seasonal_naive
from model_sarima import run_sarima
from model_wavelet import run_wavelet_autoreg
from model_xgboost import run_xgboost


OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def run_all_models(
    series: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Run the two references and the three requested models."""

    return {
        "Seasonal naive": run_seasonal_naive(series),
        "AutoReg": run_autoreg(series),
        "SARIMA": run_sarima(series),
        "XGBoost": run_xgboost(series),
        "Wavelet+AutoReg": run_wavelet_autoreg(series),
    }


def print_metrics(model_name: str, result: dict[str, Any]) -> None:
    """Print one model's final-test metrics."""

    metrics = result["metrics"]

    print(f"\n{model_name}:")
    print(f"  MAE   = {metrics['mae']:.3f}")
    print(f"  RMSE  = {metrics['rmse']:.3f}")
    print(f"  MAPE  = {metrics['mape']:.2f}%")
    print(f"  sMAPE = {metrics['smape']:.2f}%")
    print(f"  MASE  = {metrics['mase']:.3f}")
    print(f"  Configuration: {result['configuration']}")


def run_demo(dataset: str, series_id: str) -> None:
    """Run and plot every model for one selected series."""

    series, metadata = data_utils.load_series(dataset, series_id)
    series_name = str(metadata["Series"]).strip()
    train, validation, test = data_utils.split_series(series)

    print(
        f"Running {dataset} series {series_name} "
        f"({len(series)} observations)"
    )
    print(
        f"Split: train={len(train)}, validation={len(validation)}, "
        f"test={len(test)}"
    )

    results = run_all_models(series)

    for model_name, result in results.items():
        print_metrics(model_name, result)

    plotting.plot_series(series, series_name)
    plotting.plot_wavelet_components(
        series,
        series_name,
        results["Wavelet+AutoReg"],
    )
    plotting.plot_forecast_comparison(series, series_name, results)
    plotting.plot_sarima_diagnostics(results["SARIMA"])


def print_batch_summary(metrics: pd.DataFrame) -> None:
    """Summarise the scale-free metrics across all series."""

    print("\nAverage scale-free metrics across the 15 series:")
    print(
        metrics.groupby("model")[["mape", "smape", "mase"]]
        .mean()
        .sort_values("mase")
        .to_string()
    )

    winner_indices = metrics.groupby(
        ["dataset", "series"]
    )["mase"].idxmin()

    print("\nNumber of series won according to MASE:")
    print(
        metrics.loc[winner_indices]
        .groupby("model")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )


def run_batch(run_dm: bool = False) -> None:
    """Run every selected series and save the result tables."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    dm_rows = []

    for dataset, series_id in data_utils.list_series():
        series, metadata = data_utils.load_series(dataset, series_id)
        series_name = str(metadata["Series"]).strip()
        category = str(metadata["Category"]).strip()

        print(
            f"\n=== {dataset} / {series_name} "
            f"({category}, {len(series)} observations) ==="
        )

        results = run_all_models(series)

        for model_name, result in results.items():
            print_metrics(model_name, result)

            row = {
                "dataset": dataset,
                "series": series_name,
                "category": category,
                "model": model_name,
                "configuration": result["configuration"],
            }
            row.update(result["metrics"])
            metrics_rows.append(row)

        if run_dm:
            print(
                f"\nRunning {data_utils.DM_EVALUATION_POINTS} "
                "rolling one-step forecasts for DM..."
            )
            dm_rows.extend(
                dm_analysis.run_dm_for_series(
                    dataset,
                    series_name,
                    category,
                    series,
                )
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = OUTPUT_DIR / "results_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print(f"\nSaved metrics to {metrics_path}")
    print_batch_summary(metrics)

    if run_dm:
        dm_results = pd.DataFrame(dm_rows)
        dm_path = OUTPUT_DIR / "results_dm_test.csv"
        dm_results.to_csv(dm_path, index=False)

        print(f"\nSaved Diebold-Mariano results to {dm_path}")
        print("\nDM conclusions:")
        print(dm_results["conclusion"].value_counts().to_string())
    else:
        print(
            "\nDiebold-Mariano was not run. "
            "Use --run-dm to include it."
        )


def main() -> None:
    """Read command-line arguments and start the selected execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["demo", "batch"],
        default="batch",
    )
    parser.add_argument(
        "--dataset",
        choices=["M3", "M4"],
        default="M3",
    )
    parser.add_argument(
        "--series",
        default="N1830",
        help="Series identifier used in demo mode.",
    )
    parser.add_argument(
        "--run-dm",
        action="store_true",
        help="Run the slower rolling Diebold-Mariano experiment.",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args.dataset, args.series)
    else:
        run_batch(run_dm=args.run_dm)


if __name__ == "__main__":
    main()
