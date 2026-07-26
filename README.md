# Operational Analytics Project

This project compares forecasting methods for monthly time series from the
M3 and M4 datasets.

The three main methods requested for the project are:

- SARIMA
- XGBoost
- Wavelet decomposition followed by AutoReg

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
├── dm_test.py
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

- 2 MICRO series
- 2 INDUSTRY series
- 2 MACRO series
- 2 FINANCE series
- 1 DEMOGRAPHIC series
- 1 OTHER series

It also uses five longer monthly M4 series from the MICRO, INDUSTRY, MACRO,
FINANCE and DEMOGRAPHIC categories.

A total of 15 series are analysed.

## Chronological data split

Every series is divided without changing its temporal order:

```text
|--------------- training ---------------|-- validation --|---- test ----|
                                              12 months       12 months
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

- MAE
- RMSE
- MAPE
- sMAPE
- MASE

MAE and RMSE depend on the scale of each series. sMAPE and MASE are therefore
the main measures used to compare average performance across different
series.

A MASE below 1 means that the model improves, on average, on the in-sample
error of the seasonal naive reference.

## Diebold-Mariano test

The project uses the local `dm_test.py` script supplied with the course
materials instead of the external `dieboldmariano` package. The script is
included unchanged.

The test uses:

- squared loss;
- a two-sided alternative;
- forecast horizon `h=1`;
- the Harvey-Leybourne-Newbold correction already implemented inside the
  supplied script.

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

- SARIMA
- XGBoost

This produces 30 comparisons: 15 series multiplied by 2 reference models.
Each p-value is interpreted individually at the 5% significance level. No
additional multiple-comparison correction, such as Holm, is applied.

The final 12-month accuracy measures and the rolling one-step DM test answer
different questions. The first evaluates a complete one-year forecast. The
second evaluates whether differences in next-month forecast loss are
consistent across rolling origins.

Even with 24 origins, the statistical sample is small. DM results should
therefore be interpreted as exploratory evidence rather than definitive
proof. Moreover, interpreting 30 unadjusted tests increases the probability
of finding at least one apparently significant result by chance.

In `results_dm_test.csv`:

- a negative DM statistic favours Wavelet+AutoReg
- a positive DM statistic favours the model in the `model_b` column
- `p_value` is the unadjusted p-value returned by the supplied script
- `significant_5pct` reports whether that p-value is below 0.05
- a non-significant result does not prove that the models are identical

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

All executions start from a terminal opened in the project directory. The
virtual environment created during installation must be active.

On Windows PowerShell:

```powershell
cd "C:\path\to\analytix"
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
cd /path/to/analytix
source .venv/bin/activate
```

After activation, the commands are the same on every operating system and
have this general form:

```text
python main.py [--mode {demo,batch}] [--dataset {M3,M4}]
               [--series SERIES_ID] [--run-dm]
```

The available arguments are:

| Argument | Meaning | Default |
|---|---|---|
| `--mode demo` | Run all models and show plots for one series | |
| `--mode batch` | Run all models on the 15 selected series | `batch` |
| `--dataset M3` | Select the M3 dataset in demo mode | `M3` |
| `--dataset M4` | Select the M4 dataset in demo mode | |
| `--series ID` | Select the series identifier in demo mode | `N1830` |
| `--run-dm` | Add the rolling DM experiment to batch mode | disabled |

`--dataset` and `--series` are used only in demo mode. Batch mode always runs
the 10 predefined M3 series and the 5 predefined M4 series.

### Available series for demo mode

The identifiers are case-sensitive and must be written exactly as shown.

| Dataset | Category | Series identifiers |
|---|---|---|
| M3 | Micro | `N1830`, `N1842` |
| M3 | Industry | `N1901`, `N2185` |
| M3 | Macro | `N2438`, `N2219` |
| M3 | Finance | `N2591`, `N2662` |
| M3 | Demographic | `N2745` |
| M3 | Other | `N2797` |
| M4 | Micro | `M15716` |
| M4 | Industry | `M27126` |
| M4 | Macro | `M233` |
| M4 | Finance | `M43445` |
| M4 | Demographic | `M26707` |

### General quick start

Running the program without arguments starts the standard batch experiment:

```bash
python main.py
```

This is equivalent to:

```bash
python main.py --mode batch
```

It runs the five models on all 15 series and calculates the final 12-month
forecast metrics. It does not run the slower Diebold-Mariano experiment.

### Demo examples with M3 series

Run the default M3 Micro series:

```bash
python main.py --mode demo --dataset M3 --series N1830
```

Run an M3 Industry series:

```bash
python main.py --mode demo --dataset M3 --series N1901
```

Run an M3 Macro series:

```bash
python main.py --mode demo --dataset M3 --series N2438
```

Run an M3 Finance series:

```bash
python main.py --mode demo --dataset M3 --series N2591
```

Run the M3 Demographic series:

```bash
python main.py --mode demo --dataset M3 --series N2745
```

Run the M3 Other series:

```bash
python main.py --mode demo --dataset M3 --series N2797
```

The demo prints all model configurations and metrics, then displays:

- the complete observed series
- the Haar wavelet components
- the five forecast paths
- the SARIMA diagnostic plots

Close the plot windows to allow the program to finish.

### Demo examples with M4 series

Run an M4 Micro series:

```bash
python main.py --mode demo --dataset M4 --series M15716
```

Run an M4 Macro series:

```bash
python main.py --mode demo --dataset M4 --series M233
```

Run an M4 Demographic series:

```bash
python main.py --mode demo --dataset M4 --series M26707
```

Run an M4 Industry series:

```bash
python main.py --mode demo --dataset M4 --series M27126
```

Run an M4 Finance series:

```bash
python main.py --mode demo --dataset M4 --series M43445
```

### Complete batch experiment without the DM test

```bash
python main.py --mode batch
```

This runs:

- 5 forecasting models;
- 10 M3 series;
- 5 M4 series;
- 75 final model evaluations in total.

The metrics are printed in the terminal and written to:

```text
results/results_metrics.csv
```

This mode is useful when the forecasting models or metrics have changed but
the statistical comparison does not need to be recalculated.

### Complete batch experiment with the DM test

```bash
python main.py --mode batch --run-dm
```

This first runs the standard batch experiment and then generates 24 rolling
one-step forecasts for each series. Wavelet+AutoReg is compared with SARIMA
and XGBoost, producing 30 DM comparisons.

The command writes:

```text
results/results_metrics.csv
results/results_dm_test.csv
```

The DM experiment is considerably slower because SARIMA, XGBoost and the
wavelet model must be fitted repeatedly at every rolling origin.

Running a batch command again replaces the corresponding CSV results with
the results of the new execution.

### Running without graphical output

The `batch` modes do not open plot windows, so they are the appropriate
choice for a remote machine or a server without a graphical desktop:

```bash
python main.py --mode batch
```

The `demo` mode is intended for a computer with a graphical desktop because
it displays the Matplotlib figures interactively.

### Command-line help

```bash
python main.py --help
```

Use this command to see the accepted arguments directly from the program.

### Common execution problems

- `ModuleNotFoundError`: activate `.venv` and run
  `python -m pip install -r requirements.txt`.
- `FileNotFoundError` for an M3 or M4 CSV: place both dataset files in the
  same directory as `data_utils.py`.
- `Series ... not found`: check the dataset and use one of the exact
  identifiers listed above.
- No plots appear: use demo mode on a machine with a graphical desktop.
- DM execution takes a long time: this is expected because the models are
  fitted repeatedly for the rolling forecasts.

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

Using the 30 unadjusted p-values individually, 11 comparisons are significant
at the nominal 5% level. Ten favour the competing model and one favours
Wavelet+AutoReg. These results must be treated as exploratory because no
multiple-comparison correction is applied.

## Interpretation

The project does not assume that the wavelet method must always win.

A correct conclusion is that the simple Haar configuration is competitive
on some series, but its benefit depends on the structure of the data. Wavelet
decomposition is a way to represent movements at different scales; it is not
an automatic guarantee of higher forecast accuracy.

## Authors

- Lucia Castellucci — [lucia.castellucci2@studio.unibo.it](mailto:lucia.castellucci2@studio.unibo.it)
- Luca Samorè — [luca.samore@studio.unibo.it](mailto:luca.samore@studio.unibo.it)
