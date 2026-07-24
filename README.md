# Operational Analytics Project

This project compares three forecasting approaches on multiple time series from the M3 and M4 competitions:

- SARIMA with automatic order selection
- XGBoost with lag features and recursive forecasting
- Wavelet decomposition followed by AutoReg forecasting

## Project files

```text
m3_forecast_project/
├── data_utils.py          # data loading, metrics, Diebold-Mariano test
├── model_sarima.py        # SARIMA computation
├── model_xgboost.py       # XGBoost computation
├── model_wavelet.py       # Wavelet + AutoReg computation
├── plotting.py            # EDA and model-result plots
├── main.py                # entry point and orchestration
├── M3C_monthly.csv        # M3 monthly dataset
├── M4_monthly_subset.csv  # 5 longer M4 monthly series (see below)
├── README.md
└── requirements.txt
```

The model modules only fit and forecast. `plotting.py` handles figures,
while `main.py` coordinates demo and batch execution.

## Data

- **M3**: `M3C_monthly.csv`, the original M3 monthly dataset. Ten series are used for testing, stratified across the six M3 domains (2 from MICRO, INDUSTRY, MACRO and FINANCE, 1 from DEMOGRAPHIC and 1 from OTHER), selected with a fixed random seed (42) — see `M3_SERIES_IDS` in `data_utils.py`.
- **M4**: `M4_monthly_subset.csv`, five longer monthly series, one per domain. See `M4_SERIES_IDS` in `data_utils.py`.

Both datasets are read through the same `_load_row` loader in `data_utils.py`, so the three models don't need to know which dataset a series came from.

## Requirements

- 64-bit Python 3.12 or newer
- `pip`

## Setup on Windows

XGBoost requires the Microsoft Visual C++ runtime. If it is not already installed, install the current x64 [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

Open PowerShell in the project directory and run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell does not allow activation scripts, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## Setup on macOS

```bash
brew install libomp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `libomp` package is the OpenMP runtime required by XGBoost on macOS.

## Setup on Linux

```bash
sudo apt update
sudo apt install libgomp1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the project

With the virtual environment active, `main.py` supports two modes:

### Demo mode (one series, full plots)

For the live code walkthrough: runs exploratory analysis + all three
models on a single series, showing every plot (as the original script
did).

```bash
python main.py --mode demo --dataset M3 --series N1892
python main.py --mode demo --dataset M4 --series M15716
```

### Batch mode (all series, metrics only)

Runs all 10 M3 + 5 M4 series through the three models with no plot
pop-ups, then writes:

- `results/results_metrics.csv` — MAE/RMSE/MAPE for every
  series x model combination
- `results/results_dm_test.csv` — pairwise Diebold-Mariano test
  (statistic, p-value, significance at 5%) for every series, comparing
  each pair of models

```bash
python main.py --mode batch
```

This can take several minutes (SARIMA's stepwise search is the slowest step, especially on the longer M4 series). Progress and per-series metrics are printed to the terminal as it runs; a summary (average metrics per model, share of series with a significant DM difference per model pair) is printed at the end.

## Authors
