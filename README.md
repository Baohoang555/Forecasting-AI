## Project map

| Path | Purpose |
|---|---|
| `src/` | Cleaning, aggregation, feature engineering, models, evaluation, and forecasting |
| `data/raw/` | Source meal logs, HR roster, and holiday calendar |
| `data/processed/` | Generated daily, monthly, and feature datasets |
| `results/metrics/` | Backtest, budget, segment, and outlier reports |
| `results/forecast/` | Employee- and department-level forecast CSVs |
| `results/figures/` | Charts for the model, budget, and forecast reports |
| `results/logs/` | Pipeline execution logs |
| `app.py` | Streamlit dashboard |
| `run_pipeline.py` | Pipeline entry point |

See [DEPLOYMENT.md](./DEPLOYMENT.md) for environment setup and operational
instructions.
