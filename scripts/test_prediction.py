"""# Dự đoán tiền ăn hàng tháng của nhân viên

Test nhanh kết quả dự báo đã sinh từ pipeline ML/DL/ensemble.

Chạy từ thư mục gốc:
    python scripts/test_prediction.py

Script này không huấn luyện lại mô hình. Nó đọc prediction artifact, chọn các
nhân viên ở tháng mới nhất và tính lại tổng tiền dự báo:

    tiền = N44 * 44.000 + N70 * 70.000
"""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRICE_44 = 44_000
PRICE_70 = 70_000


def main() -> int:
    predictions_path = ROOT / "results" / "metrics" / "final_predictions_evaluated.parquet"
    if not predictions_path.exists():
        predictions_path = ROOT / "data" / "processed" / "ml_test_predictions.parquet"

    if not predictions_path.exists():
        raise FileNotFoundError(
            "Chưa có prediction artifact. Hãy chạy ML/DL, ensemble và evaluate trước."
        )

    df = pd.read_parquet(predictions_path)
    required = {"employee_id", "month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    latest_month = df["month"].max()
    monthly_path = ROOT / "data" / "processed" / "monthly_dataset.parquet"
    if not monthly_path.exists():
        raise FileNotFoundError(f"Thiếu monthly dataset: {monthly_path}")
    monthly = pd.read_parquet(monthly_path)
    if "is_complete_month" not in monthly.columns:
        raise ValueError("monthly dataset thiếu cờ hoàn chỉnh; chạy lại pipeline.")
    complete_months = set(
        monthly.loc[monthly["is_complete_month"].astype(bool), "month"].astype(str)
    )
    if str(latest_month) not in complete_months:
        raise ValueError(
            f"Prediction tháng {latest_month} chưa được xác nhận hoàn chỉnh "
            "hoặc là artifact cũ. Hãy chạy lại pipeline ML/DL và evaluation."
        )
    result = df[df["month"] == latest_month].copy()

    if {"pred_n44_s2", "pred_n70_s2"} <= set(result.columns):
        result["pred_n44"] = result["pred_n44_s2"].clip(lower=0)
        result["pred_n70"] = result["pred_n70_s2"].clip(lower=0)
        prediction_name = "S2 ensemble"
    elif {"m3_p44", "m3_p70"} <= set(result.columns):
        result["pred_n44"] = result["m3_p44"].clip(lower=0)
        result["pred_n70"] = result["m3_p70"].clip(lower=0)
        prediction_name = "M3 CatBoost"
    elif {"m1_p44", "m1_p70"} <= set(result.columns):
        result["pred_n44"] = result["m1_p44"].clip(lower=0)
        result["pred_n70"] = result["m1_p70"].clip(lower=0)
        prediction_name = "M1 Random Forest"
    else:
        raise ValueError("Không tìm thấy cặp cột dự báo N44/N70.")

    result["predicted_cost"] = (
        result["pred_n44"] * PRICE_44 + result["pred_n70"] * PRICE_70
    )

    print("# Dự đoán tiền ăn hàng tháng của nhân viên")
    print(f"Tháng kiểm tra: {latest_month}")
    print(f"Mô hình: {prediction_name}")
    print(f"Số nhân viên: {len(result):,}")
    print(f"Tổng tiền dự báo: {result['predicted_cost'].sum():,.0f} đ")
    print(f"Trung bình mỗi nhân viên: {result['predicted_cost'].mean():,.0f} đ")
    print()
    print("10 nhân viên đầu tiên:")
    columns = ["employee_id", "month", "pred_n44", "pred_n70", "predicted_cost"]
    print(
        result.sort_values("predicted_cost", ascending=False)[columns]
        .head(10)
        .to_string(index=False, formatters={
            "pred_n44": "{:.2f}".format,
            "pred_n70": "{:.2f}".format,
            "predicted_cost": "{:,.0f} đ".format,
        })
    )

    if {"N44", "N70"} <= set(result.columns):
        result["actual_cost"] = result["N44"] * PRICE_44 + result["N70"] * PRICE_70
        result["error_cost"] = result["predicted_cost"] - result["actual_cost"]
        total_actual = result["actual_cost"].sum()
        total_error = result["error_cost"].sum()
        print()
        print(f"Tổng tiền thực tế: {total_actual:,.0f} đ")
        print(f"Sai lệch tổng: {total_error:+,.0f} đ")
        if total_actual > 0:
            print(f"Sai lệch phần trăm: {total_error / total_actual * 100:+.3f}%")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
