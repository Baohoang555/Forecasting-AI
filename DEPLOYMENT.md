# Deployment and Operations

## Requirements

- Windows with PowerShell.
- Python 3.11 (the full pipeline was last run with Python 3.11.17).
- Source files:
  - `data/raw/MealHistory_UTF8.csv`
  - `data/raw/hr.xlsx`
  - `data/raw/relax_day.xlsx`
- The packages listed in `requirements.txt`. PyTorch ran on CPU in the latest
  completed pipeline run.

## Environment setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell does not allow activation, use the virtual environment interpreter
directly:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --check
```

## Run and verify

Check inputs and installed dependencies before starting:

```powershell
python run_pipeline.py --check
```

Run all pipeline steps:

```powershell
python run_pipeline.py
```

The pipeline executes these steps:

1. Clean eCMS meal logs.
2. Aggregate employee attendance by month.
3. Generate forecasting features.
4. Evaluate baselines.
5. Evaluate machine-learning models.
6. Evaluate deep-learning models.
7. Evaluate model ensembles.
8. Generate company, segment, and outlier evaluation reports.
9. Forecast the next month and filter against the HR roster.
10. Compare prior forecasts with actuals when complete actual data is available.

To resume from a later step or run selected steps:

```powershell
python run_pipeline.py --from-step 5
python run_pipeline.py --only 1 2 10
```

Use `--no-hr-filter` only when intentionally forecasting without the HR roster.
The `--check` and `--dry-run` options do not execute model training or generate a
new forecast.

## Outputs

- `data/processed/daily_grid.parquet` — cleaned employee-day data.
- `data/processed/monthly_dataset.parquet` — employee-month history.
- `data/processed/features_monthly.parquet` — model features.
- `results/metrics/` — model metrics, budget evaluation, segment analysis, and
  outliers.
- `results/forecast/` — employee-level, department-level, and HR-excluded
  forecast CSV files.
- `results/logs/pipeline_<timestamp>.log` — console output and status for each run.

Review the latest pipeline log and its final step summary after each run. A forecast
is not verified against actuals until step 10 can find complete actual data for its
month. Do not use forecasts to calculate employee bills or deductions.

## Latest verified run (October 19, 2026)

The latest full run used Python 3.11.17. Steps 1–9 succeeded; step 10 was skipped
for the September 2026 forecast because complete actual data was available only
through August 2026.

- The cleaned data contained 353,668 log rows; 56,217 rows were removed and 1,455
  valid no-attendance rows were reinstated.
- The monthly dataset contained 20,417 employee-month rows across 1,751 employees,
  covering August 2025 through August 2026.
- Feature engineering produced 15,245 rows.
- The next-month forecast was for September 2026. After HR filtering, it covered
  1,694 employees and totaled 1,236,612,000 VND. Another 57 employees were excluded,
  representing 17,066,000 VND of forecast cost.
- On the June–August 2026 backtest, the MA 3M baseline recorded 129,230 VND cost
  MAE and 17.32% cost WAPE. CatBoost recorded 137,480 VND cost MAE and 18.78% cost
  WAPE. These are backtest metrics, not guarantees of future accuracy.

## Streamlit dashboard

Configure `DASHBOARD_PASSWORD` in the deployment environment or in a local `.env`
file beside `app.py`. Do not commit the real password or include it in logs or
documentation. Start the dashboard from the project root:

```powershell
streamlit run app.py
```

### Docker

The Docker image contains the dashboard runtime, not the full model-training
environment. It deliberately excludes `data/` and `results/`; mount these folders
at runtime so private employee data and forecasts are not baked into the image.
The mounted folders are read-only in the container.

Create a `.env` file in the project root with a `DASHBOARD_PASSWORD` entry, then
start the dashboard from PowerShell:

```powershell
Set-Content -Path .env -Value 'DASHBOARD_PASSWORD=your-private-password'
docker compose up --build -d
```

Do not commit `.env`. Compose reads the password from this file and passes it
into the container.

The dashboard is available at <http://localhost:8501>. Stop it with
`docker compose down`.

### Continuous integration

GitHub Actions runs `python -m unittest discover -s tests -v` with Python 3.11,
then builds the Docker image and checks its Streamlit health endpoint. Tests that
need the private `data/hr.xlsx` or `data/relax_day.xlsx` fixtures are skipped when
those files are not present in CI.

## Reconcile September actuals

After the September eCMS log is complete, rerun cleaning, monthly aggregation,
and forecast monitoring:

```powershell
python run_pipeline.py --only 1 2 10
```

Review the new log and the updated comparison report before treating the September
forecast as evaluated.
