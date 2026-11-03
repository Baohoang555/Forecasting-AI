"""Backtest daily company-wide meal demand with ARIMA, MLP, and Prophet."""

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

SOURCE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


PRICE_44 = 44_000
PRICE_70 = 70_000
ARIMA_ORDER = (6, 0, 1)
LAGS = (1, 7, 14, 28)
MODEL_COLUMNS = {
    "ARIMA(6,0,1)": ("arima_p44", "arima_p70"),
    "MLP (daily)": ("mlp_p44", "mlp_p70"),
    "Facebook Prophet": ("prophet_p44", "prophet_p70"),
}


def _normalize_complete_flags(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("is_complete_month có giá trị thiếu.")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        raise ValueError("is_complete_month chỉ được chứa bool hoặc giá trị 0/1.")
    return parsed.astype(bool)


def load_daily_demand(
    daily_grid_path: str | Path,
    monthly_path: str | Path,
) -> tuple[pd.DataFrame, list[pd.Period]]:
    grid = pd.read_parquet(
        daily_grid_path,
        columns=["date", "n_viet", "n_chay", "n_au"],
    )
    required_monthly = pd.read_parquet(
        monthly_path,
        columns=["month", "is_complete_month"],
    )
    if grid.empty or required_monthly.empty:
        raise ValueError("Daily grid hoặc monthly dataset rỗng.")

    grid["date"] = pd.to_datetime(grid["date"], errors="coerce").dt.normalize()
    if grid["date"].isna().any():
        raise ValueError("Daily grid có ngày thiếu hoặc không hợp lệ.")
    for column in ("n_viet", "n_chay", "n_au"):
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
        if grid[column].isna().any() or not np.isfinite(grid[column]).all():
            raise ValueError(f"Daily grid.{column} có giá trị không hợp lệ.")

    daily = (
        grid.assign(N44=grid["n_viet"] + grid["n_chay"], N70=grid["n_au"])
        .groupby("date", as_index=True)[["N44", "N70"]]
        .sum()
        .sort_index()
    )
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0.0)
    daily.index.name = "date"

    months = required_monthly.copy()
    months["month"] = pd.PeriodIndex(months["month"], freq="M")
    months["is_complete_month"] = _normalize_complete_flags(
        months["is_complete_month"]
    )
    complete_months = (
        months.groupby("month")["is_complete_month"]
        .all()
        .loc[lambda flags: flags]
        .index.sort_values()
    )
    test_months = list(complete_months[-3:])
    if not test_months:
        raise ValueError("Không có tháng hoàn chỉnh để backtest.")

    for month in test_months:
        dates = pd.date_range(month.start_time, month.end_time, freq="D")
        if dates.min() < daily.index.min() or dates.max() > daily.index.max():
            raise ValueError(
                f"Daily grid không phủ đủ ngày cho tháng hoàn chỉnh {month}."
            )
    return daily, test_months


def _holiday_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    from calendar_utils import load_public_holidays

    holidays = load_public_holidays()
    dates = pd.DatetimeIndex(dates).normalize()
    holiday_dates = dates[dates.isin(holidays)]
    if holiday_dates.empty:
        return pd.DataFrame(columns=["holiday", "ds"])
    return pd.DataFrame({"holiday": "company_holiday", "ds": holiday_dates})


def _date_features(dates: pd.DatetimeIndex) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    day_of_week = dates.dayofweek.to_numpy(dtype=float)
    day_of_year = dates.dayofyear.to_numpy(dtype=float)
    return np.column_stack(
        [
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
            (day_of_week >= 5).astype(float),
            dates.is_month_start.astype(float),
            dates.is_month_end.astype(float),
            dates.day.to_numpy(dtype=float) / 31,
        ]
    )


def forecast_mlp(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Fit a pooled daily MLP and recursively forecast both meal types."""
    values = history[["N44", "N70"]].to_numpy(dtype=float)
    if len(values) <= max(LAGS):
        raise ValueError("MLP cần tối thiểu 29 ngày lịch sử.")

    training_features = []
    training_targets = []
    for index in range(max(LAGS), len(values)):
        features = np.concatenate(
            [
                values[index - np.asarray(LAGS), 0],
                values[index - np.asarray(LAGS), 1],
                _date_features(pd.DatetimeIndex([history.index[index]]))[0],
            ]
        )
        training_features.append(features)
        training_targets.append(values[index])

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(np.asarray(training_features))
    y_train = y_scaler.fit_transform(np.asarray(training_targets))
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(x_train, y_train)

    forecast_history = [row.copy() for row in values]
    predictions = []
    for date in future_dates:
        lagged = np.asarray(
            [
                [
                    *[forecast_history[-lag][target] for lag in LAGS],
                    *_date_features(pd.DatetimeIndex([date]))[0],
                ]
                for target in (0, 1)
            ]
        )
        features = np.concatenate([lagged[0][:4], lagged[1][:4], lagged[0][4:]])
        predicted = y_scaler.inverse_transform(
            model.predict(x_scaler.transform(features.reshape(1, -1)))
        )[0]
        predicted = np.clip(predicted, 0, None)
        predictions.append(predicted)
        forecast_history.append(predicted)
    return np.asarray(predictions, dtype=float)


def forecast_arima(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
) -> np.ndarray:
    if len(history) < 20:
        raise ValueError("ARIMA cần tối thiểu 20 ngày lịch sử.")
    predictions = []
    for target in ("N44", "N70"):
        transformed = np.log1p(history[target].to_numpy(dtype=float))
        model = ARIMA(
            transformed,
            order=ARIMA_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit()
        forecast = np.asarray(fitted.forecast(steps=len(future_dates)), dtype=float)
        predictions.append(np.expm1(np.clip(forecast, -20, 20)))
    return np.clip(np.column_stack(predictions), 0, None)


def forecast_prophet(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
) -> np.ndarray:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Cần cài package 'prophet' để chạy benchmark Facebook Prophet."
        ) from exc

    holiday_dates = pd.DatetimeIndex(history.index).append(
        pd.DatetimeIndex(future_dates)
    )
    holidays = _holiday_frame(holiday_dates)
    predictions = []
    for target in ("N44", "N70"):
        train = pd.DataFrame(
            {"ds": history.index, "y": history[target].to_numpy(dtype=float)}
        )
        model = Prophet(
            holidays=holidays if not holidays.empty else None,
            weekly_seasonality=True,
            yearly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        model.fit(train)
        future = pd.DataFrame({"ds": future_dates})
        prediction = model.predict(future)["yhat"].to_numpy(dtype=float)
        predictions.append(np.clip(prediction, 0, None))
    return np.column_stack(predictions)


def calculate_daily_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError("Actual và prediction phải có cùng dạng (ngày, 2).")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual hoặc prediction có NaN/Inf.")

    predicted = np.clip(predicted, 0, None)
    actual_cost = actual[:, 0] * PRICE_44 + actual[:, 1] * PRICE_70
    predicted_cost = predicted[:, 0] * PRICE_44 + predicted[:, 1] * PRICE_70
    absolute_cost_error = np.abs(actual_cost - predicted_cost)
    total_cost = actual_cost.sum()

    metrics = {
        "MAE_Cost": float(absolute_cost_error.mean()),
        "RMSE_Cost": float(np.sqrt(np.mean((actual_cost - predicted_cost) ** 2))),
        "WAPE_Cost_%": float(
            absolute_cost_error.sum() / total_cost * 100 if total_cost else np.nan
        ),
    }
    for index, target in enumerate(("N44", "N70")):
        errors = actual[:, index] - predicted[:, index]
        positive_actual = actual[:, index] > 0
        metrics[f"MSE_{target}"] = float(np.mean(errors**2))
        metrics[f"MAPE_{target}_positive_%"] = float(
            np.mean(np.abs(errors[positive_actual] / actual[positive_actual, index])) * 100
            if positive_actual.any()
            else np.nan
        )
        metrics[f"MAE_{target}"] = float(np.mean(np.abs(errors)))
    return metrics


def run_time_series_backtest(
    daily_grid_path: str | Path = "data/processed/daily_grid.parquet",
    monthly_path: str | Path = "data/processed/monthly_dataset.parquet",
    output_dir: str | Path = "results/metrics",
    prediction_path: str | Path = "data/processed/time_series_test_predictions.parquet",
) -> pd.DataFrame:
    daily, test_months = load_daily_demand(daily_grid_path, monthly_path)
    predictors = {
        "ARIMA(6,0,1)": forecast_arima,
        "MLP (daily)": forecast_mlp,
        "Facebook Prophet": forecast_prophet,
    }
    fold_metrics = []
    prediction_frames = []

    for month in test_months:
        history = daily.loc[daily.index < month.start_time]
        dates = pd.date_range(month.start_time, month.end_time, freq="D")
        actual = daily.loc[dates, ["N44", "N70"]].to_numpy(dtype=float)
        fold = pd.DataFrame({"date": dates, "N44": actual[:, 0], "N70": actual[:, 1]})

        for model_name, predictor in predictors.items():
            predicted = predictor(history, dates)
            if predicted.shape != actual.shape:
                raise ValueError(
                    f"{model_name} trả về {predicted.shape}; cần {actual.shape}."
                )
            columns = MODEL_COLUMNS[model_name]
            fold[columns[0]], fold[columns[1]] = predicted[:, 0], predicted[:, 1]
            metrics = calculate_daily_metrics(actual, predicted)
            metrics.update({"Model": model_name, "Test_Month": str(month)})
            fold_metrics.append(metrics)

        prediction_frames.append(fold)

    by_month = pd.DataFrame(fold_metrics)
    summary = (
        by_month.groupby("Model", as_index=False)
        .mean(numeric_only=True)
        .sort_values("MAE_Cost")
        .reset_index(drop=True)
    )
    summary.insert(1, "Evaluation_Level", "Company daily total")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "time_series_models_summary.csv", index=False)
    by_month.to_csv(out_dir / "time_series_models_by_month.csv", index=False)

    predictions_path = Path(prediction_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        predictions_path, index=False
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark daily ARIMA, MLP, and Facebook Prophet models."
    )
    parser.add_argument("--daily-grid", default="data/processed/daily_grid.parquet")
    parser.add_argument("--monthly", default="data/processed/monthly_dataset.parquet")
    parser.add_argument("--output-dir", default="results/metrics")
    parser.add_argument(
        "--predictions",
        default="data/processed/time_series_test_predictions.parquet",
    )
    args = parser.parse_args()

    summary = run_time_series_backtest(
        args.daily_grid, args.monthly, args.output_dir, args.predictions
    )
    print("\n===== DAILY COMPANY-WIDE TIME-SERIES BACKTEST =====")
    print(summary.to_string(index=False))
    print(f"Đã lưu benchmark: {Path(args.output_dir) / 'time_series_models_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
