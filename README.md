# Operational Analytics Project

This project compares forecasting methods for monthly time series from the
M3 and M4 datasets.

The three main methods used in this project are:

- SARIMA;
- XGBoost with 12 lagged values;
- Haar wavelet decomposition followed by AutoReg(12).

Two simple reference models are also included:

- seasonal naive;
- AutoReg(12) applied directly to the original series.

The seasonal naive model shows whether a method improves on simply repeating
the previous year's observations. The direct AutoReg model shows whether the
wavelet decomposition provides an actual benefit over using the same
forecasting model on the original series.

## Project structure

```text
analytix/
├── data_utils.py
├── dm_analysis.py
├── model_baselines.py
├── model_sarima.py
├── model_xgboost.py
├── model_wavelet.py
├── plotting.py
├── main.py
├── M3C_monthly.csv
├── M4_monthly_subset.csv
├── requirements.txt
├── README.md
└── results/
    ├── results_metrics.csv
    └── results_dm_test.csv
```

The two dataset files must remain in the same directory as `data_utils.py`.
The `results` directory is created automatically when a batch experiment is
run.

## Data

The experiment uses ten monthly M3 series:

- 2 MICRO series;
- 2 INDUSTRY series;
- 2 MACRO series;
- 2 FINANCE series;
- 1 DEMOGRAPHIC series;
- 1 OTHER series.

It also uses five longer monthly M4 series from the MICRO, INDUSTRY, MACRO,
FINANCE and DEMOGRAPHIC categories.

A total of 15 series are analysed.

## Chronological data split

Every series is divided without changing its temporal order:

```text
|--------------- training ---------------|-- validation --|---- test ----|
                                                    12 months      12 months
```

The final 12 observations are used only as the test set. They are not used
to select model settings.

The 12 validation observations are used to select one of three small XGBoost
configurations. The wavelet configuration is fixed before the experiment.
SARIMA selects its orders using only the history available before the test
set.

After model selection, the training and validation portions are combined.
Each model is fitted again before forecasting the 12 test observations.

The 12-month horizon is an experimental choice made for this project. The
code does not attempt to reproduce the complete official M3 or M4 competition
protocol.

## Models

### Seasonal naive

For each forecasted month, this model repeats the value observed in the same
month of the previous year.

### AutoReg

AutoReg uses the last 12 observations of the original series. It is both a
simple baseline and a control for the wavelet method.

### SARIMA

`pmdarima.auto_arima` selects the non-seasonal and seasonal orders. The
seasonal period is 12 and the selection criterion is AICc, which includes an
additional correction for smaller samples.

### XGBoost

The model inputs are the previous 12 observations. Only three configurations
are compared:

```text
100 trees, maximum depth 2, learning rate 0.05
200 trees, maximum depth 2, learning rate 0.05
200 trees, maximum depth 3, learning rate 0.05
```

The configuration with the lowest validation MASE is selected.

Multi-step forecasts are recursive: each prediction is added to the history
used to predict the following month. This keeps the implementation simple,
although forecasting errors may accumulate over the 12 steps.

### Haar wavelet and AutoReg

The wavelet configuration is intentionally fixed and simple:

```text
wavelet = haar
level = 2
boundary mode = periodization
component model = AutoReg(12)
```

The observed history is reconstructed as:

```text
history = A2 + D2 + D1
```

AutoReg(12) forecasts A2, D2 and D1 separately. The component forecasts are
then added together:

```text
final forecast = forecast(A2) + forecast(D2) + forecast(D1)
```

The wavelet transform uses only the observations available at the time of
the forecast. Future test values are never included in the decomposition.

## Accuracy measures

The following measures are calculated for every model:

- MAE;
- RMSE;
- MAPE;
- sMAPE;
- MASE.

MAE and RMSE depend on the scale of each series. sMAPE and MASE are therefore
the main measures used to compare average performance across different
series.

A MASE below 1 means that the model improves, on average, on the in-sample
error of the seasonal naive reference.

## Diebold-Mariano test

The Diebold-Mariano test is not implemented manually. The project uses the
`dm_test` function from the `dieboldmariano` package.

The test uses:

- squared loss;
- a two-sided alternative;
- forecast horizon `h=1`;
- the Harvey-Leybourne-Newbold correction for small samples.

The test requires a sequence of comparable forecast errors. The 12 errors
from one multi-step forecast are not used directly because they refer to
different forecast horizons.

Instead, the project generates 24 rolling one-step forecasts for each series:

```text
fit through month t     -> forecast month t+1
fit through month t+1   -> forecast month t+2
fit through month t+2   -> forecast month t+3
...
```

The SARIMA structure and XGBoost configuration are selected using only the
history available before the first rolling origin. Their structures then
remain fixed while their parameters are estimated again as new observations
become available. The wavelet configuration is fixed throughout the
experiment.

Wavelet+AutoReg is compared with:

- SARIMA;
- XGBoost.

This produces 30 comparisons: 15 series multiplied by 2 reference models.
The resulting p-values are adjusted together with the Holm method.

The final 12-month accuracy measures and the rolling one-step DM test answer
different questions. The first evaluates a complete one-year forecast. The
second evaluates whether differences in next-month forecast loss are
consistent across rolling origins.

Even with 24 origins, the statistical sample is small. DM results should
therefore be interpreted as exploratory evidence rather than definitive
proof.

In `results_dm_test.csv`:

- a negative DM statistic favours Wavelet+AutoReg;
- a positive DM statistic favours the model in the `model_b` column;
- a non-significant result does not prove that the models are identical.

## Requirements

Python 3.11 or a newer compatible version is recommended.

Before starting, open a terminal in the project directory and confirm that
the two CSV files listed in the project structure are present.

The examples below create an isolated virtual environment named `.venv`.

## Installation on Windows

### PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, allow it only for the current
PowerShell session and activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Command Prompt

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the Python launcher is not available, replace `py -3.11` with `python`.

## Installation on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If XGBoost reports that the OpenMP runtime is missing, install it with
Homebrew and repeat the dependency installation:

```bash
brew install libomp
python -m pip install -r requirements.txt
```

## Installation on Linux

On Ubuntu or Debian, first make sure that Python virtual environments are
available:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

Then create the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If XGBoost reports a missing OpenMP runtime on Ubuntu or Debian:

```bash
sudo apt install libgomp1
```

On Fedora, the corresponding package can be installed with:

```bash
sudo dnf install libgomp
```

## Running the project

Run all commands from the project directory with the virtual environment
activated.

### Demonstration on one series

```bash
python main.py --mode demo --dataset M3 --series N1830
```

The demo prints all model configurations and metrics, then displays:

- the complete observed series;
- the Haar wavelet components;
- the five forecast paths;
- the SARIMA diagnostic plots.

An M4 series can be selected in the same way:

```bash
python main.py --mode demo --dataset M4 --series M15716
```

### Complete batch experiment

```bash
python main.py --mode batch
```

This runs the five models on all 15 series and writes:

```text
results/results_metrics.csv
```

### Complete batch experiment with the DM test

```bash
python main.py --mode batch --run-dm
```

This also performs the rolling one-step forecasts and writes:

```text
results/results_metrics.csv
results/results_dm_test.csv
```

The DM experiment is considerably slower because SARIMA, XGBoost and the
wavelet model must be fitted repeatedly at every rolling origin.

### Command-line help

```bash
python main.py --help
```

### Leaving the virtual environment

This command is the same on Windows, macOS and Linux:

```bash
deactivate
```

## Current experiment results

Average scale-free performance across the 15 selected series:

| Model | Mean sMAPE | Mean MASE |
|---|---:|---:|
| AutoReg(12) | 9.225 | 0.818 |
| SARIMA | 10.366 | 0.882 |
| Wavelet+AutoReg | 10.155 | 0.893 |
| XGBoost | 9.902 | 1.049 |
| Seasonal naive | 11.961 | 1.198 |

AutoReg(12) has the best mean MASE. SARIMA and Wavelet+AutoReg are close,
while XGBoost is more variable across individual series.

Wavelet+AutoReg has a mean MASE below 1 and wins one of the 15 series. The
result suggests that the fixed Haar decomposition is useful for some series
but does not provide a systematic improvement over the simpler models.

After applying the Holm correction to all 30 DM comparisons, only the
Wavelet+AutoReg versus XGBoost comparison for M3 series N2797 is
statistically significant. It favours XGBoost. The other 29 comparisons do
not provide sufficient evidence of a difference in expected forecast
accuracy.

## Interpretation

The project does not assume that the wavelet method must always win.

A correct conclusion is that the simple Haar configuration is competitive
on some series, but its benefit depends on the structure of the data. Wavelet
decomposition is a way to represent movements at different scales; it is not
an automatic guarantee of higher forecast accuracy.

## Authors

- Lucia Castellucci — [lucia.castellucci2@studio.unibo.it](mailto:lucia.castellucci2@studio.unibo.it)
- Luca Samorè — [luca.samore@studio.unibo.it](mailto:luca.samore@studio.unibo.it)