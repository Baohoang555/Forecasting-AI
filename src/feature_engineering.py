
import numpy as np
import pandas as pd

from data_cleaning import PUBLIC_HOLIDAYS


LAG_SOURCE_COLS = [
    "N44",
    "N70",
    "total_meals",
    "days_with_meal",
    "total_amount",
]


def get_working_days_in_month(period: pd.Period) -> int:
    """Số ngày THỨ HAI-THỨ SÁU trong tháng, trừ ngày nghỉ đã cấu hình (data/relax_day.xlsx).

    Lưu ý ngữ nghĩa: KHÔNG gồm thứ Bảy làm việc (lịch thứ Bảy chưa đưa vào mô hình theo quyết
    định hiện tại của công ty), nên đây là proxy lịch chứ không phải "số ngày công ty thực sự
    vận hành". Không dùng giá trị này làm giới hạn trên của số suất ăn.
    """
    period = pd.Period(period, freq="M")
    start_date = period.start_time.normalize()
    end_date = period.end_time.normalize()

    bus_days = pd.bdate_range(start=start_date, end=end_date)
    holidays = pd.DatetimeIndex(PUBLIC_HOLIDAYS).normalize()

    return int((~bus_days.normalize().isin(holidays)).sum())


def _normalize_month(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tháng và kiểm tra khóa employee-month."""
    df = df.copy()

    if df["employee_id"].isna().any():
        raise ValueError("monthly_dataset có employee_id bị thiếu.")

    try:
        df["month"] = pd.PeriodIndex(df["month"], freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError("Cột month không hợp lệ.") from exc

    if df["month"].isna().any():
        raise ValueError("monthly_dataset có month bị thiếu.")

    duplicated = df.duplicated(["employee_id", "month"], keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["employee_id", "month"]]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            f"Có employee_id-month bị trùng; ví dụ: {examples}"
        )

    return df.sort_values(
        ["employee_id", "month"]
    ).reset_index(drop=True)


def _build_calendar_lags(
    df: pd.DataFrame,
    source_cols: list[str],
    max_lag: int = 3,
) -> pd.DataFrame:
    """Tạo lag theo tháng lịch chính xác, không nhảy qua tháng thiếu."""
    result = df.copy()
    month_num = result["month"].map(lambda p: p.year * 12 + p.month)

    grouped = result.groupby("employee_id", sort=False)

    for lag in range(1, max_lag + 1):
        previous_month_num = grouped["month"].shift(lag).map(
            lambda p: p.year * 12 + p.month if pd.notna(p) else np.nan
        )

        valid_lag = (month_num - previous_month_num).eq(lag)

        for col in source_cols:
            shifted = grouped[col].shift(lag)
            result[f"{col}_lag_{lag}"] = shifted.where(valid_lag, np.nan)

    return result


def _build_department_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo target encoding theo thời gian.

    Feature của tháng t chỉ sử dụng target từ các tháng nhỏ hơn t.
    Department được lấy từ department_lag_1 để tránh dùng phòng ban
    của tháng hiện tại làm thông tin lịch sử.
    """
    result = df.copy()

    if "department" not in result.columns:
        result["department_lag_1"] = np.nan
        result["dept_enc_n44"] = np.nan
        result["dept_enc_n70"] = np.nan
        return result

    grouped = result.groupby("employee_id", sort=False)
    department_lag = grouped["department"].shift(1)

    month_num = result["month"].map(lambda p: p.year * 12 + p.month)
    previous_month_num = grouped["month"].shift(1).map(
        lambda p: p.year * 12 + p.month if pd.notna(p) else np.nan
    )
    valid_lag1 = (month_num - previous_month_num).eq(1)

    result["department_lag_1"] = department_lag.where(
        valid_lag1, np.nan
    )

    result["dept_enc_n44"] = np.nan
    result["dept_enc_n70"] = np.nan

    # Tính riêng từng tháng, chỉ lấy target từ các tháng trước.
    for current_month in sorted(result["month"].unique()):
        current_mask = result["month"].eq(current_month)
        history = result.loc[~(result["month"] >= current_month)].copy()

        if history.empty:
            # Không có lịch sử: không lấy trung bình từ tương lai.
            result.loc[current_mask, "dept_enc_n44"] = 0.0
            result.loc[current_mask, "dept_enc_n70"] = 0.0
            continue

        global_n44 = history["N44"].mean()
        global_n70 = history["N70"].mean()

        if pd.isna(global_n44):
            global_n44 = 0.0
        if pd.isna(global_n70):
            global_n70 = 0.0

        # Encoding phòng ban sử dụng department được biết trước đó.
        history = history.dropna(subset=["department_lag_1"])

        if history.empty:
            result.loc[current_mask, "dept_enc_n44"] = global_n44
            result.loc[current_mask, "dept_enc_n70"] = global_n70
            continue

        dept_means = history.groupby("department_lag_1")[
            ["N44", "N70"]
        ].mean()

        current_dept = result.loc[
            current_mask, "department_lag_1"
        ]

        result.loc[current_mask, "dept_enc_n44"] = (
            current_dept.map(dept_means["N44"]).fillna(global_n44)
        )
        result.loc[current_mask, "dept_enc_n70"] = (
            current_dept.map(dept_means["N70"]).fillna(global_n70)
        )

    return result


def build_features(
    input_path="data/processed/monthly_dataset.parquet",
    output_path="data/processed/features_monthly.parquet",
):
    print("Đang đọc dữ liệu monthly...")

    df = pd.read_parquet(input_path)

    required_cols = {
        "employee_id",
        "month",
        "N44",
        "N70",
        "total_meals",
        "days_with_meal",
        "total_amount",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"monthly_dataset thiếu các cột bắt buộc: {sorted(missing)}"
        )

    df = _normalize_month(df)

    # Kiểm tra target và các cột số bắt buộc.
    numeric_cols = [
        "N44",
        "N70",
        "total_meals",
        "days_with_meal",
        "total_amount",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].isna().any():
            raise ValueError(
                f"Cột {col} có giá trị thiếu hoặc không phải số."
            )

        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Cột {col} có giá trị vô cực.")

    # Giữ lại cờ tháng hoàn chỉnh nếu upstream đã cung cấp.
    if "is_complete_month" in df.columns:
        if df["is_complete_month"].isna().any():
            raise ValueError("is_complete_month có giá trị thiếu.")

        if pd.api.types.is_bool_dtype(df["is_complete_month"]):
            df["is_complete_month"] = df["is_complete_month"].astype(bool)
        else:
            normalized = (
                df["is_complete_month"].astype(str).str.strip().str.lower()
            )
            mapping = {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }

            if not normalized.isin(mapping).all():
                raise ValueError(
                    "is_complete_month chỉ được chứa bool hoặc giá trị 0/1."
                )

            df["is_complete_month"] = normalized.map(mapping).astype(bool)

    # 1. Feature lịch đã biết trước.
    months = sorted(df["month"].unique())
    all_needed_months = set(months)

    for month in months:
        all_needed_months.add(month - 1)

    work_days_map = {
        month: get_working_days_in_month(month)
        for month in all_needed_months
    }

    df["working_days"] = df["month"].map(work_days_map).astype(float)

    # 2. Lag theo tháng lịch.
    df = _build_calendar_lags(
        df,
        source_cols=LAG_SOURCE_COLS,
        max_lag=3,
    )

    # 3. Rolling features chỉ lấy dữ liệu các tháng trước.
    for col in ("N44", "N70", "total_meals"):
        lag_frame = df[
            [f"{col}_lag_1", f"{col}_lag_2", f"{col}_lag_3"]
        ]

        df[f"{col}_mean_3m"] = lag_frame.mean(axis=1)
        df[f"{col}_std_3m"] = lag_frame.std(
            axis=1, ddof=1
        ).fillna(0.0)

    lag1 = df["total_meals_lag_1"]
    lag2 = df["total_meals_lag_2"]

    df["meal_momentum_1_2"] = (
        (lag1 + 1.0) / (lag2 + 1.0)
    ).replace([np.inf, -np.inf], np.nan)

    prev_months = df["month"].map(lambda month: month - 1)
    prev_working_days = prev_months.map(work_days_map).astype(float)
    prev_working_days = prev_working_days.replace(0, np.nan)

    df["attendance_rate_lag1"] = (
        df["days_with_meal_lag_1"] / prev_working_days
    ).clip(0.0, 1.0).fillna(0.0)

    df["au_ratio_lag1"] = (
        df["N70_lag_1"] / (df["total_meals_lag_1"] + 1e-5)
    ).clip(0.0, 1.0).fillna(0.0)

    # 4. Target encoding theo thời gian, không dùng target tương lai.
    df = _build_department_encodings(df)

    # Bảo đảm feature chi phí không được suy ra từ target tháng hiện tại.
    # Các cột target hiện tại vẫn được giữ làm nhãn train/backtest.
    cols_to_drop = [
        "au_share",
        "expected_amount",
        "amount_diff",
        "N44_lag1",
        "N44_lag2",
        "N44_lag3",
        "N70_lag1",
        "N70_lag2",
        "N70_lag3",
        "au_share_3m",
    ]

    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 5. Chỉ giữ các dòng có đủ ba tháng lịch sử liên tiếp.
    required_history = [
        f"{col}_lag_{lag}"
        for col in LAG_SOURCE_COLS
        for lag in (1, 2, 3)
    ]

    df_features = df.loc[
        df[required_history].notna().all(axis=1)
    ].copy()

    # Chỉ dùng các tháng hoàn chỉnh khi cờ này có trong dữ liệu.
    if "is_complete_month" in df_features.columns:
        df_features = df_features.loc[
            df_features["is_complete_month"]
        ].copy()

    df_features = df_features.reset_index(drop=True)

    if df_features.empty:
        raise ValueError(
            "Không có dòng nào đủ 3 tháng lịch sử liên tiếp "
            "và thuộc tháng hoàn chỉnh; kiểm tra monthly_dataset."
        )

    if df_features.duplicated(["employee_id", "month"]).any():
        raise ValueError("Feature dataset có employee-month bị trùng.")

    # Lưu ý: N44 và N70 của tháng hiện tại là nhãn, không phải feature.
    print(
        f"Bảng đặc trưng hoàn chỉnh: "
        f"{df_features.shape[0]:,} dòng × {df_features.shape[1]} cột"
    )

    output_dir = pd.io.common.stringify_path(output_path)
    from pathlib import Path

    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)

    df_features.to_parquet(output_dir, index=False)
    print(f"Đã lưu tại: {output_dir}")

    return df_features


if __name__ == "__main__":
    build_features()