"""Compare a monthly forecast with actuals without silently dropping unmatched employees.

Employee-level error metrics are calculated on matched employee IDs only. Company-level
budget deviation uses the full forecast cohort versus the full actual cohort, and coverage
counts make any cohort mismatch explicit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_44 = 44_000
PRICE_70 = 70_000


def _require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{label} thiếu cột bắt buộc: {sorted(missing)}")


def _cost_from_actual(df: pd.DataFrame) -> pd.Series:
    if "total_amount" in df.columns:
        cost = pd.to_numeric(df["total_amount"], errors="coerce")
        # Only use total_amount when it is populated; otherwise reconstruct from meal counts.
        if cost.notna().all():
            return cost.astype(float)
    _require_columns(df, {"N44", "N70"}, "Actual monthly data")
    n44 = pd.to_numeric(df["N44"], errors="coerce")
    n70 = pd.to_numeric(df["N70"], errors="coerce")
    if n44.isna().any() or n70.isna().any():
        raise ValueError("Actual N44/N70 hoặc total_amount chứa giá trị thiếu/không phải số.")
    return n44 * PRICE_44 + n70 * PRICE_70


def run_monitor(
    forecast_path="results/forecast/forecast_YYYY-MM_employee.csv",
    monthly_path="data/processed/monthly_dataset.parquet",
    wape_threshold=15.0,
    output_dir="results/metrics",
):
    forecast_path = Path(forecast_path)
    monthly_path = Path(monthly_path)
    if not forecast_path.exists():
        raise FileNotFoundError(f"Không tìm thấy forecast: {forecast_path}")
    if not monthly_path.exists():
        raise FileNotFoundError(f"Không tìm thấy monthly dataset: {monthly_path}")

    forecast = pd.read_csv(forecast_path)
    monthly = pd.read_parquet(monthly_path)
    _require_columns(forecast, {"employee_id", "forecast_month", "pred_cost"}, "Forecast")
    _require_columns(monthly, {"employee_id", "month"}, "Monthly dataset")
    if forecast.empty:
        raise ValueError("Forecast file rỗng.")
    if forecast["employee_id"].isna().any() or monthly["employee_id"].isna().any():
        raise ValueError("employee_id bị thiếu trong forecast hoặc monthly dataset.")
    if forecast["forecast_month"].isna().any():
        raise ValueError("forecast_month bị thiếu trong forecast.")

    forecast_months = pd.PeriodIndex(forecast["forecast_month"].astype(str), freq="M").unique()
    if len(forecast_months) != 1:
        raise ValueError(f"Forecast phải chứa đúng một tháng; tìm thấy: {[str(m) for m in forecast_months]}")
    forecast_month = forecast_months[0]
    forecast = forecast.copy()
    monthly = monthly.copy()
    forecast["employee_id"] = forecast["employee_id"].astype(str)
    monthly["employee_id"] = monthly["employee_id"].astype(str)
    forecast["pred_cost"] = pd.to_numeric(forecast["pred_cost"], errors="coerce")
    if forecast["pred_cost"].isna().any() or not np.isfinite(forecast["pred_cost"]).all():
        raise ValueError("pred_cost có giá trị thiếu hoặc không hữu hạn.")
    if (forecast["pred_cost"] < 0).any():
        raise ValueError("pred_cost không được âm.")
    if forecast["employee_id"].duplicated().any():
        dup = forecast.loc[forecast["employee_id"].duplicated(False), "employee_id"].head(10).tolist()
        raise ValueError(f"Forecast có employee_id trùng; không tự ý drop. Ví dụ: {dup}")

    monthly["month"] = pd.PeriodIndex(monthly["month"], freq="M")
    actual = monthly.loc[monthly["month"] == forecast_month].copy()
    if actual.empty:
        raise ValueError(
            f"Chưa có actual cho tháng {forecast_month}; monitor chưa thể đánh giá forecast tháng tương lai."
        )
    if actual["employee_id"].duplicated().any():
        dup = actual.loc[actual["employee_id"].duplicated(False), "employee_id"].head(10).tolist()
        raise ValueError(f"Monthly actual có employee_id trùng trong {forecast_month}: {dup}")

    actual["actual_cost"] = _cost_from_actual(actual)
    actual = actual[["employee_id", "actual_cost"]]
    forecast = forecast[["employee_id", "pred_cost"]]
    comparison = forecast.merge(actual, on="employee_id", how="outer", validate="one_to_one", indicator=True)
    comparison["status"] = comparison["_merge"].map({
        "both": "matched", "left_only": "forecast_only", "right_only": "actual_only"
    })
    comparison["abs_error"] = (comparison["pred_cost"] - comparison["actual_cost"]).abs()
    comparison["signed_error"] = comparison["pred_cost"] - comparison["actual_cost"]

    matched = comparison.loc[comparison["_merge"] == "both"].copy()
    forecast_only = comparison.loc[comparison["_merge"] == "left_only"]
    actual_only = comparison.loc[comparison["_merge"] == "right_only"]
    n_forecast, n_actual, n_matched = len(forecast), len(actual), len(matched)
    forecast_coverage = n_matched / n_forecast * 100 if n_forecast else 0.0
    actual_coverage = n_matched / n_actual * 100 if n_actual else 0.0

    mae = float(matched["abs_error"].mean()) if n_matched else float("nan")
    matched_actual_sum = float(matched["actual_cost"].sum()) if n_matched else 0.0
    individual_wape = (
        float(matched["abs_error"].sum() / matched_actual_sum * 100)
        if matched_actual_sum > 0 else float("nan")
    )

    # Company-level comparison intentionally uses every row on each side, including unmatched IDs.
    total_pred = float(forecast["pred_cost"].sum())
    total_actual = float(actual["actual_cost"].sum())
    company_budget_deviation = (
        abs(total_pred - total_actual) / total_actual * 100 if total_actual > 0 else float("nan")
    )
    signed_budget_deviation = total_pred - total_actual

    print("\n" + "=" * 72)
    print(f"MONITOR DỰ BÁO — THÁNG {forecast_month}")
    print("=" * 72)
    print(f"Cohort forecast / actual / matched : {n_forecast:,} / {n_actual:,} / {n_matched:,}")
    print(f"Coverage forecast / actual        : {forecast_coverage:.2f}% / {actual_coverage:.2f}%")
    print(f"Không khớp chỉ có forecast / actual: {len(forecast_only):,} / {len(actual_only):,}")
    print(f"MAE chi phí trên cohort matched    : {mae:,.0f} đ" if np.isfinite(mae) else "MAE: N/A (không có employee matched)")
    print(f"WAPE cá nhân trên cohort matched   : {individual_wape:.2f}%" if np.isfinite(individual_wape) else "WAPE cá nhân: N/A")
    print(f"Tổng forecast toàn cohort          : {total_pred:,.0f} đ")
    print(f"Tổng actual toàn cohort            : {total_actual:,.0f} đ")
    print(f"Sai lệch ngân sách có dấu           : {signed_budget_deviation:,.0f} đ")
    print(f"Sai lệch tổng ngân sách (%)         : {company_budget_deviation:.2f}%" if np.isfinite(company_budget_deviation) else "Sai lệch tổng ngân sách: N/A")

    if len(forecast_only) or len(actual_only):
        print("[CẢNH BÁO] Cohort forecast và actual không khớp; xem file employee comparison để điều tra.")
    if np.isfinite(individual_wape) and individual_wape > float(wape_threshold):
        print(f"[CẢNH BÁO] WAPE cá nhân {individual_wape:.2f}% vượt ngưỡng {wape_threshold:.2f}%.")
    if np.isfinite(company_budget_deviation) and company_budget_deviation > float(wape_threshold):
        print(f"[CẢNH BÁO] Sai lệch tổng ngân sách {company_budget_deviation:.2f}% vượt ngưỡng {wape_threshold:.2f}%.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"forecast_accuracy_employees_{forecast_month}.csv"
    comparison.drop(columns=["_merge"]).to_csv(comparison_path, index=False)

    row = {
        "forecast_month": str(forecast_month),
        "n_forecast": n_forecast,
        "n_actual": n_actual,
        "n_matched": n_matched,
        "n_forecast_only": len(forecast_only),
        "n_actual_only": len(actual_only),
        "forecast_coverage_pct": forecast_coverage,
        "actual_coverage_pct": actual_coverage,
        "mae_cost_matched": mae,
        "individual_wape_matched_pct": individual_wape,
        "forecast_total_cost_all": total_pred,
        "actual_total_cost_all": total_actual,
        "company_budget_signed_error": signed_budget_deviation,
        "company_budget_deviation_pct": company_budget_deviation,
        "wape_threshold_pct": float(wape_threshold),
    }
    log_path = output_dir / "forecast_accuracy_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    if not log.empty and "forecast_month" in log.columns:
        log = log[log["forecast_month"].astype(str) != str(forecast_month)]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(log_path, index=False)
    print(f"Đã lưu đối chiếu nhân viên: {comparison_path}")
    print(f"Đã cập nhật log metrics: {log_path}")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", required=True, help="CSV forecast_<YYYY-MM>_employee.csv")
    parser.add_argument("--monthly", default="data/processed/monthly_dataset.parquet")
    parser.add_argument("--wape-threshold", type=float, default=15.0)
    parser.add_argument("--output-dir", default="results/metrics")
    args = parser.parse_args()
    run_monitor(args.forecast, args.monthly, args.wape_threshold, args.output_dir)
