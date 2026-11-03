
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_N44 = 44_000
PRICE_N70 = 70_000

REQUIRED_COLUMNS = {
    "employee_id",
    "date",
    "department",
    "n_viet",
    "n_chay",
    "n_au",
    "n_meals",
    "total_amount",
    "has_checkin",
}


def _validate_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Validate daily-grid columns and normalize dates/months."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Daily grid thiếu các cột bắt buộc: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Daily grid không có dữ liệu.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Cột date có giá trị thiếu hoặc không hợp lệ.")

    if df["employee_id"].isna().any():
        raise ValueError("Cột employee_id có giá trị thiếu.")

    # Daily grid cần tối đa một dòng cho mỗi nhân viên-ngày.
    duplicated = df.duplicated(["employee_id", "date"], keep=False)
    if duplicated.any():
        examples = df.loc[
            duplicated, ["employee_id", "date"]
        ].head(10).to_dict("records")
        raise ValueError(
            "Daily grid có employee_id-date bị trùng. "
            f"Ví dụ: {examples}"
        )

    numeric_cols = [
        "n_viet",
        "n_chay",
        "n_au",
        "n_meals",
        "total_amount",
        "has_checkin",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        bad_cols = df[numeric_cols].columns[
            df[numeric_cols].isna().any()
        ].tolist()
        raise ValueError(
            f"Các cột số có giá trị thiếu/không hợp lệ: {bad_cols}"
        )

    if (df[numeric_cols] < 0).any().any():
        raise ValueError("Daily grid có giá trị âm trong các cột số.")

    df["month"] = df["date"].dt.to_period("M")
    return df.sort_values(["employee_id", "date"]).reset_index(drop=True)


def create_monthly_dataset(grid_path, output_path, first_complete_month=None):
    """first_complete_month: (tuỳ chọn) 'YYYY-MM' - bỏ mọi tháng trước tháng này (ví dụ tháng
    khởi động hệ thống). Nếu không truyền, chỉ CẢNH BÁO khi tháng đầu có vẻ bị cụt."""
    print("Đọc daily grid...")
    df = pd.read_parquet(grid_path)
    df = _validate_daily_grid(df)

    # ------------------------------------------------
    # 0. KIỂM TRA TRÁI: THÁNG ĐẦU CÓ BỊ CỤT KHÔNG? (đối xứng với kiểm tra tháng cuối)
    # ------------------------------------------------
    if first_complete_month is not None:
        cutoff = pd.Period(first_complete_month, freq="M")
        n_before = int((df["month"] < cutoff).sum())
        if n_before:
            print(f"[THÔNG BÁO] Bỏ {n_before:,} dòng thuộc các tháng trước {cutoff} "
                  f"(--first-complete-month).")
            df = df.loc[df["month"] >= cutoff].copy()
        if df.empty:
            raise ValueError("Không còn dữ liệu sau khi bỏ các tháng trước first_complete_month.")
    first_date = df["date"].min()
    first_month_period = first_date.to_period("M")
    if first_date.normalize() > first_month_period.start_time.normalize() + pd.Timedelta(days=3):
        print(
            f"[CẢNH BÁO] Tháng đầu {first_month_period} có thể KHÔNG đầy đủ: dữ liệu bắt đầu "
            f"từ {first_date.date()} (không phải đầu tháng). Nếu đây là giai đoạn khởi động hệ "
            f"thống, hãy chạy lại với --start-date ở bước làm sạch hoặc "
            f"--first-complete-month ở bước này."
        )

    # ------------------------------------------------
    # 1. XÁC ĐỊNH THÁNG HOÀN CHỈNH
    # ------------------------------------------------
    # Quy tắc này giả định dữ liệu được cập nhật liên tục đến ngày
    # lớn nhất trong daily grid. Nếu pipeline có ngày chốt dữ liệu
    # riêng, nên dùng ngày chốt đó thay cho df["date"].max().
    last_date = df["date"].max()
    last_month = last_date.to_period("M")
    last_day_of_month = last_month.end_time.normalize()

    is_last_month_complete = last_date.normalize() >= last_day_of_month

    if not is_last_month_complete:
        print(
            f"[CẢNH BÁO] Tháng {last_month} chưa hoàn chỉnh: "
            f"dữ liệu mới nhất đến {last_date.date()}, "
            f"cuối tháng là {last_day_of_month.date()}. "
            "Loại tháng này khỏi monthly dataset."
        )
        df = df.loc[df["month"] != last_month].copy()

    if df.empty:
        raise ValueError(
            "Không còn dữ liệu sau khi loại tháng chưa hoàn chỉnh."
        )

    # Đánh dấu theo tháng, nhất quán trên tất cả nhân viên.
    df["is_complete_month"] = True

    # ------------------------------------------------
    # 2. AGGREGATE EMPLOYEE-MONTH
    # ------------------------------------------------
    df["n44"] = df["n_viet"] + df["n_chay"]
    df["n70"] = df["n_au"]

    monthly = (
        df.groupby(["employee_id", "month"], as_index=False, sort=True)
        .agg(
            department=("department", "last"),
            N44=("n44", "sum"),
            N70=("n70", "sum"),
            total_meals=("n_meals", "sum"),
            total_amount=("total_amount", "sum"),
            days_with_meal=("has_checkin", "sum"),
            calendar_days=("date", "nunique"),
            is_complete_month=("is_complete_month", "all"),
        )
    )

    if monthly.duplicated(["employee_id", "month"]).any():
        raise ValueError(
            "Lỗi sau aggregation: employee_id-month bị trùng."
        )

    # ------------------------------------------------
    # 3. CALENDAR SPINE TRONG KHOẢNG QUAN SÁT CỦA TỪNG NHÂN VIÊN
    # ------------------------------------------------
    # Không dùng last department để gán ngược về quá khứ.
    # Không kéo nhân viên đến tháng cuối hệ thống nếu họ không còn
    # xuất hiện trong dữ liệu.
    #
    # Giới hạn quan trọng:
    # Không có HR roster/ngày vào-nghỉ việc nên first/last observed
    # month chỉ là xấp xỉ cohort, không chứng minh ngày làm việc thực tế.
    emp_meta = (
        monthly.groupby("employee_id", as_index=False)
        .agg(
            first_month=("month", "min"),
            last_month=("month", "max"),
        )
    )

    spine_parts = []
    for row in emp_meta.itertuples(index=False):
        month_range = pd.period_range(
            start=row.first_month,
            end=row.last_month,
            freq="M",
        )
        spine_parts.append(
            pd.DataFrame(
                {
                    "employee_id": row.employee_id,
                    "month": month_range,
                }
            )
        )

    full_spine = pd.concat(spine_parts, ignore_index=True)

    monthly = full_spine.merge(
        monthly,
        on=["employee_id", "month"],
        how="left",
        validate="one_to_one",
    )

    # Chỉ điền 0 cho tháng nằm trong khoảng quan sát của nhân viên.
    # Giả định daily grid đầu vào là nguồn dữ liệu đầy đủ và một tháng
    # không có dòng employee-month trong khoảng này thực sự là không ăn.
    fill_zero_cols = [
        "N44",
        "N70",
        "total_meals",
        "total_amount",
        "days_with_meal",
        "calendar_days",
    ]
    monthly[fill_zero_cols] = monthly[fill_zero_cols].fillna(0)

    # ------------------------------------------------
    # 4. PHÒNG BAN: CHỈ FORWARD-FILL TỪ QUÁ KHỨ
    # ------------------------------------------------
    monthly = monthly.sort_values(
        ["employee_id", "month"]
    ).reset_index(drop=True)

    monthly["department"] = (
        monthly.groupby("employee_id", sort=False)["department"].ffill()
    )

    # Không bfill: không dùng department của tương lai điền ngược.
    # Nếu không có phòng ban nào được biết từ trước, giữ NaN.
    if monthly["department"].isna().any():
        missing_count = int(monthly["department"].isna().sum())
        print(
            f"[CẢNH BÁO] Có {missing_count:,} dòng thiếu department "
            "vì không có thông tin phòng ban đã biết trước đó."
        )

    # Các tháng được tạo trong spine nằm trong khoảng tháng đã được
    # giữ lại ở bước 1, nên đều thuộc thời gian quan sát hoàn chỉnh.
    monthly["is_complete_month"] = monthly["is_complete_month"].fillna(
        True
    ).astype(bool)

    int_cols = [
        "N44",
        "N70",
        "total_meals",
        "days_with_meal",
        "calendar_days",
    ]
    monthly[int_cols] = monthly[int_cols].round().astype("int64")

    if monthly.duplicated(["employee_id", "month"]).any():
        raise ValueError(
            "Monthly dataset có employee_id-month bị trùng."
        )

    # Kiểm tra tính hợp lý giữa tổng suất ăn và hai nhóm suất.
    expected_meals = monthly["N44"] + monthly["N70"]
    if not np.array_equal(
        expected_meals.to_numpy(),
        monthly["total_meals"].to_numpy(),
    ):
        print(
            "[CẢNH BÁO] N44 + N70 không bằng total_meals ở một số dòng. "
            "Kiểm tra định nghĩa các cột nguồn trước khi sử dụng."
        )

    # ------------------------------------------------
    # 5. ĐỐI SOÁT CHI PHÍ
    # ------------------------------------------------
    monthly["expected_amount"] = (
        monthly["N44"] * PRICE_N44
        + monthly["N70"] * PRICE_N70
    )

    monthly["amount_diff"] = (
        monthly["total_amount"] - monthly["expected_amount"]
    )

    print("\n=== MONTHLY DATASET ===")
    print(f"Employee-month: {len(monthly):,}")
    print(f"Employees: {monthly['employee_id'].nunique():,}")
    print(
        f"Months: {monthly['month'].min()} -> "
        f"{monthly['month'].max()}"
    )
    print(
        "Duplicate employee-month:",
        int(monthly.duplicated(["employee_id", "month"]).sum()),
    )

    print("\n=== TARGET CHECK ===")
    mismatch = monthly[monthly["amount_diff"].abs() > 0.01]
    print(f"Amount mismatch: {len(mismatch):,}")
    if not mismatch.empty:
        print(mismatch.head(10))

    # ------------------------------------------------
    # 6. CALENDAR LAGS
    # ------------------------------------------------
    monthly = monthly.sort_values(
        ["employee_id", "month"]
    ).reset_index(drop=True)

    for col in ("N44", "N70"):
        grouped = monthly.groupby("employee_id", sort=False)[col]
        for lag in (1, 2, 3):
            monthly[f"{col}_lag{lag}"] = grouped.shift(lag)

    # ------------------------------------------------
    # 7. ROLLING FEATURES TỪ CÁC THÁNG TRƯỚC
    # ------------------------------------------------
    for col in ("N44", "N70"):
        monthly[f"{col}_mean_3m"] = (
            monthly.groupby("employee_id", sort=False)[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
        )

    # ------------------------------------------------
    # 8. AU SHARE
    # ------------------------------------------------
    monthly["au_share"] = np.divide(
        monthly["N70"],
        monthly["total_meals"],
        out=np.zeros(len(monthly), dtype=float),
        where=monthly["total_meals"].to_numpy() > 0,
    )

    monthly["au_share_3m"] = (
        monthly.groupby("employee_id", sort=False)["au_share"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    )

    # ------------------------------------------------
    # 9. TENURE THEO THÁNG QUAN SÁT
    # ------------------------------------------------
    first_month = monthly.groupby("employee_id")["month"].transform("min")
    monthly["tenure_months"] = (
        (monthly["month"].dt.year - first_month.dt.year) * 12
        + (monthly["month"].dt.month - first_month.dt.month)
    )

    # ------------------------------------------------
    # 10. KIỂM TRA VÀ LƯU
    # ------------------------------------------------
    monthly = monthly.sort_values(
        ["employee_id", "month"]
    ).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(output_path, index=False)

    print(f"\nĐã lưu: {output_path}")
    return monthly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        default="data/processed/daily_grid.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/processed/monthly_dataset.parquet",
    )
    parser.add_argument(
        "--first-complete-month",
        default=None,
        help="YYYY-MM: bỏ mọi tháng trước tháng này (ví dụ tháng khởi động hệ thống).",
    )
    args = parser.parse_args()

    create_monthly_dataset(args.grid, args.output, args.first_complete_month)
    