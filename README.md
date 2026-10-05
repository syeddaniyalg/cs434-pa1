# CS434 Assignment 1: Structured Time Series Forecasting

This repository contains the implementation and experimental results for Assignment 1 of CS434, Deep Learning for Space, Time and Graphs. The assignment studies how temporal inductive biases affect forecasting in two settings: a controlled multi-sensor vibration process and an anonymized real-world time series.

The submitted report is available in [`paper/main.tex`](paper/main.tex). It uses the NeurIPS format and discusses the experimental design, quantitative results, model-selection decisions, and final forecasts for both tasks.

## Task 1: Forecasting Across Heterogeneous Sensors

Task 1 uses a simulated production line with four vibration sensors and three operating products. Models receive 96 observations from one station and predict the next 48 observations. Stations 1--3 are used for fitting, while Station 4 is held out to evaluate transfer to an unseen sensor.

The notebook develops and compares:

- shared and sensor-specific ridge forecasters;
- a period-routed ridge forecaster;
- Raw Attention;
- Attention with moving-average decomposition;
- a raw temporal-delay mixer; and
- an Autoformer-inspired model combining decomposition and delay mixing.

The three required implementation components are completed in [`task1/task1_notebook.ipynb`](task1/task1_notebook.ipynb): the Raw Attention forward pass, centered moving-average decomposition, and weighted delay aggregation. The notebook was executed with the full preset. Its checks pass, and Outputs 1.1--4.3 are stored with the notebook and under `task1/results/design/`.

The Autoformer-inspired model is the strongest neural model in this controlled experiment. Period-routed ridge is selected for deployment on both established and held-out stations because its physical assumption closely matches the simulated process and gives the best validation result at low fitting cost.

## Task 2: Autoformer Forecasting

Task 2 forecasts the next 168 observations of an anonymized, non-negative target series. The implementation in [`task2/autoformer/autoformer.py`](task2/autoformer/autoformer.py) is a compact encoder-decoder Autoformer built from scratch. It includes progressive moving-average decomposition and FFT-based auto-correlation with time-delay aggregation.

The experimental pipeline in [`task2/task2_notebook.ipynb`](task2/task2_notebook.ipynb) includes:

- chronological training and validation splits;
- target and external-variable scaling fitted only on the training region;
- smoke and one-batch overfitting checks;
- a multi-seed moving-average-width ablation;
- a three-seed external-data ablation;
- final training on all observed target values; and
- generation of the required 168-step forecast.

Validation selects moving-average width 13 and the use of all ten external variables. The final model has 42,561 trainable parameters and is trained for three epochs. Its forecast is saved in [`task2/outputs/predictions.txt`](task2/outputs/predictions.txt). On the hidden leaderboard continuation, it obtains MAE 62.3154, RMSE 84.5748, sMAPE 75.99%, and a final score of 84.5764.

## Repository Structure

```text
.
|-- DL4STG-PA1.pdf             # Assignment handout
|-- paper/                     # NeurIPS report source and referenced figures
|-- task1/
|   |-- task1_notebook.ipynb   # Executed Task 1 notebook
|   |-- harness/               # Supplied data, models, checks, and reporting tools
|   |-- checkpoints/           # Trained Task 1 checkpoints
|   `-- results/design/        # Numbered tables and figures
|-- task2/
|   |-- task2_notebook.ipynb   # Executed Task 2 notebook
|   |-- autoformer/            # Autoformer implementation
|   |-- data/                  # Target, test indices, and external variables
|   |-- checkpoints/           # Trained Task 2 checkpoints
|   `-- outputs/               # Forecast, metrics, and figures
`-- requirements.txt
```

## Environment Setup

Python 3.11 or later is recommended. From the repository root, create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, install the CPU wheel before the remaining requirements:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Running the Notebooks

Launch Jupyter Lab from the task directory whose notebook you want to run. For Task 1:

```bash
cd task1
jupyter lab
```

For Task 2, return to the repository root and start Jupyter from `task2/`:

```bash
cd task2
jupyter lab
```

The notebooks use task-relative paths, so the working directory must be the directory that contains the selected notebook.

Task 1 supports two presets through `PA1_PRESET`. Use `smoke` while checking an implementation and `full` for report results. The submitted notebook explicitly uses the full preset. CUDA is selected when available; `PA1_DEVICE=cpu` can be set before importing the harness to force CPU execution.

Task 2 automatically selects CUDA when it is available and otherwise uses CPU. Its saved checkpoints allow completed experiments to be inspected without repeating every training run. Set the notebook's `RESUME` variable only when intentionally continuing an interrupted run.

## Report

The report source is self-contained apart from its local NeurIPS style file and the assets under `paper/figures/`. To compile it on Overleaf, upload the contents of `paper/` and set `main.tex` as the main document.

