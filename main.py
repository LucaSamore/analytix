#!/usr/bin/env python3
"""Entry point: runs SARIMA, XGBoost and Wavelet+AutoReg sequentially.

Two modes:

  --mode demo   Full step-by-step run (every plot) on ONE series.
                Example: python main.py --mode demo --dataset M3 --series N1892

  --mode batch  Runs all 10 M3 + 5 M4 series through the three models
                with no plot pop-ups, then saves two CSV files:
                  results/results_metrics.csv   (MAE/RMSE/MAPE per series/model)
                  results/results_dm_test.csv   (Diebold-Mariano test per pair)
                Example: python main.py --mode batch
"""

import argparse
from pathlib import Path

import pandas as pd

import data_utils
from model_sarima import run_sarima
from model_xgboost import run_xgboost
from model_wavelet import run_wavelet_autoreg

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

MODELS = {
    "SARIMA": run_sarima,
    "XGBoost": run_xgboost,
    "Wavelet+AutoReg": run_wavelet_autoreg,
}


def run_demo(dataset, series_id):
    loader = data_utils.load_m3_series if dataset == "M3" else data_utils.load_m4_series
    series, metadata = loader(series_id)
    series_name = metadata.iloc[0]

    print(f"Running full demo on {dataset} series {series_name} "
          f"({len(series)} observations)")

    data_utils.show_exploratory_analysis(series, series_name)

    for model_name, run_model in MODELS.items():
        print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")
        result = run_model(series, series_name, show_plots=True)
        print(f"\n{model_name} metrics: {result['metrics']}")


def run_batch():
    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_rows = []
    dm_rows = []

    for dataset, series_id, loader in data_utils.list_series():
        series, metadata = loader(series_id)
        series_name = metadata.iloc[0]
        category = str(metadata.iloc[3]).strip()

        print(f"\n=== {dataset} / {series_name} ({category}, "
              f"{len(series)} observations) ===")

        results = {}
        for model_name, run_model in MODELS.items():
            results[model_name] = run_model(series, series_name, show_plots=False)

            row = {"dataset": dataset, "series": series_name, "category": category, "model": model_name}
            row.update(results[model_name]["metrics"])
            metrics_rows.append(row)

        model_names = list(results.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a, name_b = model_names[i], model_names[j]
                dm_stat, p_value = data_utils.diebold_mariano_test(
                    results[name_a]["actual"],
                    results[name_a]["forecast"],
                    results[name_b]["forecast"],
                )
                dm_rows.append({
                    "dataset": dataset,
                    "series": series_name,
                    "category": category,
                    "model_a": name_a,
                    "model_b": name_b,
                    "dm_statistic": dm_stat,
                    "p_value": p_value,
                    "significant_5pct": p_value < 0.05,
                })

    metrics_df = pd.DataFrame(metrics_rows)
    dm_df = pd.DataFrame(dm_rows)

    metrics_path = OUTPUT_DIR / "results_metrics.csv"
    dm_path = OUTPUT_DIR / "results_dm_test.csv"
    metrics_df.to_csv(metrics_path, index=False)
    dm_df.to_csv(dm_path, index=False)

    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved Diebold-Mariano results to {dm_path}")

    print("\nAverage metrics per model (across all 15 series):")
    print(metrics_df.groupby("model")[["mae", "rmse", "mape"]].mean().to_string())

    print("\nDiebold-Mariano: share of series with a significant "
          "difference (p<0.05) per model pair:")
    print(dm_df.groupby(["model_a", "model_b"])["significant_5pct"].mean().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["demo", "batch"], default="batch")
    parser.add_argument("--dataset", choices=["M3", "M4"], default="M3", help="Dataset for demo mode")
    parser.add_argument("--series", default="N1892", help="Series id for demo mode")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args.dataset, args.series)
    else:
        run_batch()


if __name__ == "__main__":
    main()
