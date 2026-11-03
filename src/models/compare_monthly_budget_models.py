"""Compare aggregate monthly time-series forecasts with existing ML/DL models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

PRICE_44 = 44_000
PRICE_70 = 70_000
TEST_MONTHS = 3
ARIMA_ORDER = (6, 0, 1)
MLP_LAGS = 3


def _month_index(values: pd.Series) -> pd.PeriodIndex:
    try:
        return pd.PeriodIndex(values, freq="M")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Có giá trị month không hợp lệ.") from exc


def _daily_values(monthly_totals: pd.DataFrame, column: str) -> np.ndarray:
    return monthly_totals[column].to_numpy(dtype=float)


def _forecast_arima(history: pd.DataFrame, steps: int) -> np.ndarray:
    if len(history) < ARIMA_ORDER[0] + 2:
        raise ValueError("ARIMA(6,0,1) cần ít nhất 8 tháng lịch sử.")
    forecasts = []
    for column in ("N44", "N70"):
        transformed = np.log1p(_daily_values(history, column))
        fitted = ARIMA(
            transformed,
            order=ARIMA_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        forecasts.append(
            np.expm1(
                np.clip(np.asarray(fitted.forecast(steps=steps), dtype=float), -20, 20)
            )
        )
    return np.clip(np.column_stack(forecasts), 0, None)


def _forecast_mlp(history: pd.DataFrame) -> np.ndarray:
    if len(history) <= MLP_LAGS:
        raise ValueError("MLP cần nhiều hơn 3 tháng lịch sử.")
    values = history[["N44", "N70"]].to_numpy(dtype=float)
    x = np.asarray(
        [values[index - MLP_LAGS:index].reshape(-1) for index in range(MLP_LAGS, len(values))]
    )
    y = values[MLP_LAGS:]
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)
    model = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=1.0,
        max_iter=1000,
        random_state=42,
    )
    model.fit(x_scaled, y_scaled)
    prediction = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(values[-MLP_LAGS:].reshape(1, -1))).reshape(1, -1)
    )[0]
    return np.clip(prediction, 0, None).reshape(1, 2)


def _forecast_prophet(
    history: pd.DataFrame,
    forecast_month: pd.Period,
) -> np.ndarray:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Cần cài package 'prophet' để chạy benchmark Prophet."
        ) from exc

    forecasts = []
    for column in ("N44", "N70"):
        train = pd.DataFrame(
            {
                "ds": history.index.to_timestamp(),
                "y": history[column].to_numpy(dtype=float),
            }
        )
        model = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        model.fit(train)
        future = pd.DataFrame(
            {"ds": [forecast_month.to_timestamp()]}
        )
        forecasts.append(
            float(np.clip(model.predict(future)["yhat"].iloc[0], 0, None))
        )
    return np.asarray(forecasts, dtype=float).reshape(1, 2)


def _budget_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
) -> dict[str, float]:
    joined = actual[["month", "N44", "N70"]].merge(
        predicted[["month", "N44", "N70"]],
        on="month",
        how="inner",
        suffixes=("_actual", "_predicted"),
        validate="one_to_one",
    )
    if len(joined) != len(actual):
        raise ValueError("Actual và prediction không có cùng các tháng đánh giá.")
    actual_cost = (
        joined["N44_actual"] * PRICE_44
        + joined["N70_actual"] * PRICE_70
    )
    predicted_cost = (
        joined["N44_predicted"] * PRICE_44
        + joined["N70_predicted"] * PRICE_70
    )
    error = predicted_cost - actual_cost
    return {
        "MAE_Budget": float(np.abs(error).mean()),
        "RMSE_Budget": float(np.sqrt(np.mean(error**2))),
        "WAPE_Budget_%": float(
            np.abs(error).sum() / actual_cost.sum() * 100
            if actual_cost.sum()
            else np.nan
        ),
        "Mean_Bias_Budget": float(error.mean()),
    }


def _validate_model_predictions(
    predictions: pd.DataFrame,
    actual: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    required = {"employee_id", "month", "N44", "N70"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{name} thiếu cột: {sorted(missing)}.")
    result = predictions[["employee_id", "month", "N44", "N70"]].copy()
    result["month"] = _month_index(result["month"])
    if result.duplicated(["employee_id", "month"]).any():
        raise ValueError(f"{name} có employee-month trùng.")
    for column in ("N44", "N70"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not np.isfinite(result[column]).all():
            raise ValueError(f"{name}.{column} có giá trị không hợp lệ.")

    expected = actual[["employee_id", "month"]].sort_values(
        ["month", "employee_id"], key=lambda col: col.astype(str)
    ).reset_index(drop=True)
    found = result[["employee_id", "month"]].sort_values(
        ["month", "employee_id"], key=lambda col: col.astype(str)
    ).reset_index(drop=True)
    if not expected.equals(found):
        raise ValueError(
            f"{name} không dự báo đúng cùng cohort employee-month với actual."
        )
    return result


def run_comparison(
    monthly_path: str | Path = "data/processed/monthly_dataset.parquet",
    ml_predictions_path: str | Path = "data/processed/ml_test_predictions.parquet",
    dl_predictions_path: str | Path = "data/processed/dl_test_predictions.parquet",
    ensemble_predictions_path: str | Path = "results/metrics/final_predictions_evaluated.parquet",
    output_dir: str | Path = "results/metrics",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_parquet(monthly_path)
    required = {"employee_id", "month", "N44", "N70", "is_complete_month"}
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f"Monthly dataset thiếu cột: {sorted(missing)}.")
    monthly = monthly.copy()
    monthly["month"] = _month_index(monthly["month"])
    complete = monthly["is_complete_month"]
    if complete.isna().any():
        raise ValueError("is_complete_month có giá trị thiếu.")
    if not pd.api.types.is_bool_dtype(complete):
        normalized = complete.astype(str).str.strip().str.lower()
        mapping = {"true": True, "false": False, "1": True, "0": False}
        complete = normalized.map(mapping)
        if complete.isna().any():
            raise ValueError("is_complete_month chỉ được chứa bool hoặc 0/1.")
    monthly = monthly.loc[complete.astype(bool)].copy()
    test_months = sorted(monthly["month"].unique())[-TEST_MONTHS:]
    if len(test_months) != TEST_MONTHS:
        raise ValueError(f"Cần đúng {TEST_MONTHS} tháng hoàn chỉnh để so sánh.")

    actual_rows = monthly[monthly["month"].isin(test_months)].copy()
    actual_totals = (
        actual_rows.groupby("month", as_index=False)[["N44", "N70"]]
        .sum()
        .sort_values("month")
    )
    model_predictions: dict[str, pd.DataFrame] = {}

    ml = pd.read_parquet(ml_predictions_path)
    ml["month"] = _month_index(ml["month"])
    for name, n44_col, n70_col in (
        ("M1 RandomForest", "m1_p44", "m1_p70"),
        ("M2 LightGBM", "m2_p44", "m2_p70"),
        ("M3 CatBoost", "m3_p44", "m3_p70"),
    ):
        prediction = ml[["employee_id", "month", n44_col, n70_col]].rename(
            columns={n44_col: "N44", n70_col: "N70"}
        )
        checked = _validate_model_predictions(prediction, actual_rows, name)
        model_predictions[name] = checked

    dl = pd.read_parquet(dl_predictions_path)
    dl["month"] = _month_index(dl["month"])
    for name, n44_col, n70_col in (
        ("M4 GRU", "m4_p44", "m4_p70"),
        ("M5 MLP (monthly)", "m5_p44", "m5_p70"),
    ):
        prediction = dl[["employee_id", "month", n44_col, n70_col]].rename(
            columns={n44_col: "N44", n70_col: "N70"}
        )
        checked = _validate_model_predictions(prediction, actual_rows, name)
        model_predictions[name] = checked

    ensemble = pd.read_parquet(ensemble_predictions_path)
    ensemble["month"] = _month_index(ensemble["month"])
    s2 = ensemble[
        ["employee_id", "month", "pred_n44_s2", "pred_n70_s2"]
    ].rename(columns={"pred_n44_s2": "N44", "pred_n70_s2": "N70"})
    model_predictions["S2 Ensemble (5 models)"] = _validate_model_predictions(
        s2, actual_rows, "S2 Ensemble"
    )

    budget_predictions = {
        name: prediction.groupby("month", as_index=False)[["N44", "N70"]]
        .sum()
        for name, prediction in model_predictions.items()
    }

    time_series_models = {
        "ARIMA(6,0,1) (monthly total)": _forecast_arima,
        "MLP (monthly total)": None,
        "Facebook Prophet (monthly total)": _forecast_prophet,
    }
    for month in test_months:
        history_rows = monthly.loc[monthly["month"] < month]
        history = (
            history_rows.groupby("month")[["N44", "N70"]]
            .sum()
            .sort_index()
        )
        history.index = pd.PeriodIndex(history.index, freq="M")
        if history.index[-1] + 1 != month:
            raise ValueError(
                f"Thiếu tháng lịch sử liền kề trước tháng test {month}."
            )
        for name, predictor in time_series_models.items():
            if name == "MLP (monthly total)":
                predicted_values = _forecast_mlp(history)
            elif name == "Facebook Prophet (monthly total)":
                predicted_values = predictor(history, month)
            else:
                predicted_values = predictor(history, steps=1)
            forecast = pd.DataFrame(
                {
                    "month": [month],
                    "N44": predicted_values[0, 0],
                    "N70": predicted_values[0, 1],
                }
            )
            budget_predictions.setdefault(name, pd.DataFrame())
            budget_predictions[name] = pd.concat(
                [budget_predictions[name], forecast],
                ignore_index=True,
            )

    monthly_rows = []
    summary_rows = []
    for model_name, predictions in budget_predictions.items():
        predictions = predictions.sort_values("month").reset_index(drop=True)
        metrics = _budget_metrics(actual_totals, predictions)
        summary_rows.append({"Model": model_name, **metrics})

        comparison = actual_totals.merge(
            predictions,
            on="month",
            suffixes=("_actual", "_predicted"),
            validate="one_to_one",
        )
        for _, row in comparison.iterrows():
            actual_cost = row["N44_actual"] * PRICE_44 + row["N70_actual"] * PRICE_70
            predicted_cost = (
                row["N44_predicted"] * PRICE_44
                + row["N70_predicted"] * PRICE_70
            )
            monthly_rows.append(
                {
                    "Model": model_name,
                    "month": str(row["month"]),
                    "Actual_N44": row["N44_actual"],
                    "Predicted_N44": row["N44_predicted"],
                    "Actual_N70": row["N70_actual"],
                    "Predicted_N70": row["N70_predicted"],
                    "Actual_Budget": actual_cost,
                    "Predicted_Budget": predicted_cost,
                    "Budget_Error": predicted_cost - actual_cost,
                }
            )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("WAPE_Budget_%")
        .reset_index(drop=True)
    )
    details = pd.DataFrame(monthly_rows).sort_values(["month", "Model"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "same_split_budget_model_comparison.csv", index=False)
    details.to_csv(output / "same_split_budget_model_comparison_by_month.csv", index=False)
    return summary, details


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare monthly company-budget forecasts on the same test months."
    )
    parser.add_argument("--monthly", default="data/processed/monthly_dataset.parquet")
    parser.add_argument(
        "--ml-predictions", default="data/processed/ml_test_predictions.parquet"
    )
    parser.add_argument(
        "--dl-predictions", default="data/processed/dl_test_predictions.parquet"
    )
    parser.add_argument(
        "--ensemble-predictions",
        default="results/metrics/final_predictions_evaluated.parquet",
    )
    parser.add_argument("--output-dir", default="results/metrics")
    args = parser.parse_args()

    summary, _ = run_comparison(
        args.monthly,
        args.ml_predictions,
        args.dl_predictions,
        args.ensemble_predictions,
        args.output_dir,
    )
    print("\n===== SAME DATASET / SAME TEST MONTHS: COMPANY BUDGET =====")
    print(summary.to_string(index=False))
    print(
        "Lưu bảng tổng hợp: "
        f"{Path(args.output_dir) / 'same_split_budget_model_comparison.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
