
"""
Production inference: dự báo số bữa ăn và ngân sách tháng tiếp theo.

- CatBoost cho nhân viên có đủ 3 tháng lịch sử liên tiếp.
- Group-Mean fallback cho nhân viên thiếu lịch sử.
- Chỉ dùng dữ liệu thuộc các tháng hoàn chỉnh.
- Không dùng dự báo để tính hóa đơn hoặc trừ tiền nhân viên.
- Số suất mỗi nhân viên chỉ bị chặn bởi giới hạn vật lý (1 suất/ngày dương lịch). KHÔNG còn
  chặn theo số ngày Thứ Hai-Thứ Sáu vì công ty có làm thứ Bảy (trần cũ ép 48% nhân viên về
  đúng 20 suất ở tháng 9/2026).
- Nhân viên không còn trong data/hr.xlsx (danh sách đang làm việc) bị loại khỏi dự báo và
  được ghi riêng ở file *_excluded_not_in_hr.csv. Tắt bằng --no-hr-filter.

Cách chạy:
    python src/predict_next_month.py \
        --monthly data/processed/monthly_dataset.parquet \
        --features data/processed/features_monthly.parquet \
        --output-dir results/forecast \
        --hr data/hr.xlsx
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from data_cleaning import PUBLIC_HOLIDAYS
from feature_engineering import get_working_days_in_month

warnings.filterwarnings("ignore")

PRICE_44 = 44_000
PRICE_70 = 70_000

TARGET_COLUMNS = ["N44", "N70"]
LAG_SOURCE_COLS = [
    "N44",
    "N70",
    "total_meals",
    "days_with_meal",
    "total_amount",
]

MODEL_FEATURES = [
    "working_days",
    "N44_lag_1",
    "N44_lag_2",
    "N44_lag_3",
    "N70_lag_1",
    "N70_lag_2",
    "N70_lag_3",
    "total_meals_lag_1",
    "total_meals_lag_2",
    "total_meals_lag_3",
    "days_with_meal_lag_1",
    "days_with_meal_lag_2",
    "days_with_meal_lag_3",
    "total_amount_lag_1",
    "total_amount_lag_2",
    "total_amount_lag_3",
    "N44_mean_3m",
    "N44_std_3m",
    "N70_mean_3m",
    "N70_std_3m",
    "total_meals_mean_3m",
    "total_meals_std_3m",
    "meal_momentum_1_2",
    "attendance_rate_lag1",
    "au_ratio_lag1",
    "tenure_months",
]


def normalize_complete_flag(series: pd.Series) -> pd.Series:
    """Parse completion flags without treating string 'False' as True."""
    if series.isna().any():
        raise ValueError("is_complete_month có giá trị thiếu.")

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }

    normalized = series.astype(str).str.strip().str.lower()
    parsed = normalized.map(mapping)

    if parsed.isna().any():
        bad = normalized[parsed.isna()].unique().tolist()[:10]
        raise ValueError(f"Giá trị is_complete_month không hợp lệ: {bad}")

    return parsed.astype(bool)


def normalize_months(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a validated Period[M] month column."""
    if "month" not in df.columns:
        raise ValueError("Dataset thiếu cột month.")

    result = df.copy()

    try:
        result["month"] = pd.PeriodIndex(result["month"], freq="M")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Cột month không thể chuyển sang định dạng Period[M]."
        ) from exc

    if result["month"].isna().any():
        raise ValueError("Cột month có giá trị thiếu.")

    return result


def check_holiday_coverage(forecast_month: pd.Period) -> None:
    holidays = pd.DatetimeIndex(PUBLIC_HOLIDAYS).normalize()

    # Lịch hết hạn: tháng dự báo thuộc năm chưa có trong lịch -> dừng, tránh im lặng dự báo
    # thiếu ngày nghỉ (ví dụ Tết năm sau).
    last_covered_year = int(holidays.year.max())
    if forecast_month.year > last_covered_year:
        raise ValueError(
            f"Lịch nghỉ lễ (data/relax_day.xlsx) chỉ phủ đến năm {last_covered_year}, "
            f"chưa có năm {forecast_month.year} của tháng dự báo {forecast_month}. "
            "Hãy bổ sung lịch nghỉ rồi chạy lại."
        )

    start = forecast_month.start_time.normalize()
    end = forecast_month.end_time.normalize()
    in_month = holidays[(holidays >= start) & (holidays <= end)]

    if len(in_month) == 0:
        print(
            f"Không có ngày lễ trong {forecast_month} "
            f"(bình thường nếu tháng đó không có kỳ nghỉ; lịch phủ đến năm {last_covered_year})."
        )
    else:
        print(
            f"Ngày lễ cấu hình trong {forecast_month}: "
            f"{[d.date().isoformat() for d in in_month]}"
        )


def get_last_complete_month(monthly: pd.DataFrame) -> pd.Period:
    required = {"employee_id", "month", "is_complete_month"}
    missing = required - set(monthly.columns)

    if missing:
        raise ValueError(
            f"Monthly dataset thiếu cột bắt buộc: {sorted(missing)}."
        )

    if monthly.empty:
        raise ValueError("Monthly dataset rỗng.")

    data = normalize_months(monthly)
    data["is_complete_month"] = normalize_complete_flag(
        data["is_complete_month"]
    )

    completeness = data.groupby("month")["is_complete_month"].all()
    complete_months = completeness[completeness].index.tolist()

    if not complete_months:
        raise ValueError("Không có tháng hoàn chỉnh được xác nhận.")

    return max(complete_months)


def get_latest_consecutive_complete_months(
    monthly: pd.DataFrame,
    last_complete_month: pd.Period,
    n_months: int = 3,
) -> list[pd.Period]:
    data = normalize_months(monthly)
    data["is_complete_month"] = normalize_complete_flag(
        data["is_complete_month"]
    )

    completeness = data.groupby("month")["is_complete_month"].all()
    result = []

    for offset in range(n_months - 1, -1, -1):
        month = last_complete_month - offset
        if month not in completeness.index or not bool(completeness.loc[month]):
            raise ValueError(
                f"Không có đủ {n_months} tháng hoàn chỉnh liên tiếp gần nhất. "
                f"Thiếu tháng {month}."
            )
        result.append(month)

    return result


def prepare_monthly_history(
    monthly: pd.DataFrame,
    last_complete_month: pd.Period,
) -> pd.DataFrame:
    required = {
        "employee_id",
        "month",
        "department",
        "N44",
        "N70",
        "total_meals",
        "days_with_meal",
        "total_amount",
        "is_complete_month",
    }
    missing = required - set(monthly.columns)

    if missing:
        raise ValueError(
            f"Monthly dataset thiếu cột bắt buộc: {sorted(missing)}."
        )

    data = normalize_months(monthly)
    data["is_complete_month"] = normalize_complete_flag(
        data["is_complete_month"]
    )

    data = data.loc[
        (data["month"] <= last_complete_month)
        & data["is_complete_month"]
    ].copy()

    if data.empty:
        raise ValueError("Không còn lịch sử tháng hoàn chỉnh.")

    if data["employee_id"].isna().any():
        raise ValueError("Monthly dataset có employee_id bị thiếu.")

    duplicated = data.duplicated(["employee_id", "month"], keep=False)
    if duplicated.any():
        examples = data.loc[
            duplicated, ["employee_id", "month"]
        ].head(10).to_dict("records")
        raise ValueError(f"Trùng employee-month: {examples}")

    for col in LAG_SOURCE_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)

    return data.sort_values(["employee_id", "month"]).reset_index(drop=True)


def build_forecast_feature_row(
    monthly: pd.DataFrame,
    forecast_month: pd.Period,
    last_complete_month: pd.Period | None = None,
) -> pd.DataFrame:
    """Build one inference row per employee in the active cohort."""
    if last_complete_month is None:
        last_complete_month = get_last_complete_month(monthly)

    if forecast_month != last_complete_month + 1:
        raise ValueError(
            "forecast_month phải là tháng ngay sau last_complete_month."
        )

    history = prepare_monthly_history(monthly, last_complete_month)

    recent_months = get_latest_consecutive_complete_months(
        history,
        last_complete_month,
        n_months=3,
    )

    active_rows = history[history["month"].isin(recent_months)]
    active_ids = active_rows["employee_id"].drop_duplicates().tolist()
    active_ids = sorted(active_ids, key=str)

    if not active_ids:
        raise ValueError("Không có nhân viên trong cohort dự báo.")

    previous_working_days = get_working_days_in_month(forecast_month - 1)
    current_working_days = get_working_days_in_month(forecast_month)

    rows = []

    for employee_id in active_ids:
        emp_history = history.loc[
            history["employee_id"] == employee_id
        ].sort_values("month")

        department_values = emp_history.loc[
            emp_history["department"].notna(), "department"
        ]

        department = (
            str(department_values.iloc[-1])
            if not department_values.empty
            else "unknown"
        )

        first_month = emp_history["month"].min()
        tenure_months = (
            (forecast_month.year - first_month.year) * 12
            + forecast_month.month
            - first_month.month
        )

        row = {
            "employee_id": employee_id,
            "month": forecast_month,
            "department": department,
            "working_days": current_working_days,
            "tenure_months": int(tenure_months),
        }

        for lag in (1, 2, 3):
            target_month = forecast_month - lag
            matched = emp_history.loc[
                emp_history["month"] == target_month
            ]

            for col in LAG_SOURCE_COLS:
                key = f"{col}_lag_{lag}"
                row[key] = (
                    matched.iloc[0][col]
                    if not matched.empty
                    else np.nan
                )

        for col in ("N44", "N70", "total_meals"):
            values = [
                row[f"{col}_lag_{lag}"]
                for lag in (1, 2, 3)
            ]
            valid_values = np.asarray(values, dtype=float)
            valid_values = valid_values[np.isfinite(valid_values)]

            row[f"{col}_mean_3m"] = (
                float(valid_values.mean())
                if len(valid_values)
                else np.nan
            )
            row[f"{col}_std_3m"] = (
                float(valid_values.std(ddof=1))
                if len(valid_values) >= 2
                else 0.0
            )

        lag1 = row["total_meals_lag_1"]
        lag2 = row["total_meals_lag_2"]

        row["meal_momentum_1_2"] = (
            (lag1 + 1) / (lag2 + 1)
            if pd.notna(lag1) and pd.notna(lag2)
            else np.nan
        )

        days_with_meal_lag1 = row["days_with_meal_lag_1"]
        row["attendance_rate_lag1"] = (
            float(
                np.clip(
                    days_with_meal_lag1 / max(previous_working_days, 1),
                    0,
                    1,
                )
            )
            if pd.notna(days_with_meal_lag1)
            else 0.0
        )

        n70_lag1 = row["N70_lag_1"]
        meals_lag1 = row["total_meals_lag_1"]
        row["au_ratio_lag1"] = (
            float(np.clip(n70_lag1 / (meals_lag1 + 1e-5), 0, 1))
            if pd.notna(n70_lag1) and pd.notna(meals_lag1)
            else 0.0
        )

        rows.append(row)

    forecast_df = pd.DataFrame(rows)

    # Department mean fallback chỉ dùng lịch sử hoàn chỉnh đã biết đến
    # last_complete_month, không dùng dữ liệu tháng dự báo hoặc tương lai.
    dept_means = history.groupby("department", dropna=True)[
        TARGET_COLUMNS
    ].mean()
    global_means = history[TARGET_COLUMNS].mean()

    for col, output_col in [
        ("N44", "dept_enc_n44"),
        ("N70", "dept_enc_n70"),
    ]:
        forecast_df[output_col] = forecast_df["department"].map(
            dept_means[col]
        ).fillna(global_means[col])

    return forecast_df


def max_meals_in_month(forecast_month: pd.Period) -> int:
    """Giới hạn vật lý của số suất/nhân viên/tháng: 1 suất mỗi ngày dương lịch.

    Thay cho trần theo working_days (Thứ Hai-Thứ Sáu) trước đây - trần đó sai vì công ty có
    làm thứ Bảy (27% nhân viên-tháng trong lịch sử ăn vượt working_days).
    """
    return int(pd.Period(forecast_month, freq="M").days_in_month)


def split_by_hr_roster(
    forecast_df: pd.DataFrame,
    hr_path,
    max_excluded_share: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tách (giữ lại, bị loại): bị loại = nhân viên không có trong danh sách đang làm việc.

    hr_path: file hr.xlsx, cột 'Mã NV' là mã nhân viên (trùng 'Mã chấm công' của eCMS).
    Chốt an toàn: nếu quá max_excluded_share nhân viên bị loại thì dừng, vì nhiều khả năng
    file HR sai/cũ/khác hệ mã chứ không phải nghỉ việc hàng loạt.
    """
    hr = pd.read_excel(hr_path)
    id_col = "Mã NV" if "Mã NV" in hr.columns else hr.columns[0]
    current_ids = set(
        pd.to_numeric(hr[id_col], errors="coerce").dropna().astype("int64").tolist()
    )
    if not current_ids:
        raise ValueError(f"Không đọc được mã nhân viên từ cột '{id_col}' của {hr_path}.")

    numeric_ids = pd.to_numeric(forecast_df["employee_id"], errors="coerce")
    if numeric_ids.isna().all():
        raise ValueError("employee_id của dự báo không phải mã số, không đối chiếu được với HR.")

    in_hr = numeric_ids.isin(current_ids)
    excluded_share = float((~in_hr).mean())
    if excluded_share > max_excluded_share:
        raise ValueError(
            f"{excluded_share:.1%} nhân viên trong dự báo không có trong {hr_path} "
            f"(vượt ngưỡng an toàn {max_excluded_share:.0%}). Kiểm tra file HR có đúng/mới không, "
            "hoặc chạy với --no-hr-filter."
        )
    return forecast_df[in_hr.to_numpy()].copy(), forecast_df[~in_hr.to_numpy()].copy()


def cap_and_round_meals(
    pred_n44,
    pred_n70,
    working_days,
) -> tuple[np.ndarray, np.ndarray]:
    """Round non-negative predictions and cap their sum at the given per-row cap.

    Tên tham số `working_days` được giữ để tương thích; nó là giới hạn trên tuỳ ý của người
    gọi. Production truyền max_meals_in_month() (số ngày dương lịch), không phải Thứ Hai-Thứ Sáu.
    """
    a = np.asarray(pred_n44, dtype=float)
    b = np.asarray(pred_n70, dtype=float)
    caps = np.asarray(working_days, dtype=int)

    if a.ndim != 1 or b.ndim != 1 or caps.ndim != 1:
        raise ValueError("Đầu vào cap_and_round_meals phải là mảng 1 chiều.")

    if not (len(a) == len(b) == len(caps)):
        raise ValueError("Số phần tử N44, N70 và working_days không khớp.")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Dự đoán có NaN hoặc vô cực.")

    if np.any(caps < 0):
        raise ValueError("working_days không được âm.")

    a = np.clip(a, 0, None)
    b = np.clip(b, 0, None)

    out_a = np.zeros(len(a), dtype=int)
    out_b = np.zeros(len(b), dtype=int)

    for i, (x, y, cap) in enumerate(zip(a, b, caps)):
        if cap == 0 or x + y <= 0:
            continue

        if x + y <= cap:
            base_a = int(np.floor(x + 0.5))
            base_b = int(np.floor(y + 0.5))

            while base_a + base_b > cap:
                frac_a = x - np.floor(x)
                frac_b = y - np.floor(y)

                if base_a > 0 and (base_b == 0 or frac_a <= frac_b):
                    base_a -= 1
                elif base_b > 0:
                    base_b -= 1
                else:
                    break
        else:
            scaled_a = x * cap / (x + y)
            scaled_b = y * cap / (x + y)

            base_a = int(np.floor(scaled_a))
            base_b = int(np.floor(scaled_b))
            remaining = cap - base_a - base_b

            fractions = sorted(
                [
                    (scaled_a - base_a, 0),
                    (scaled_b - base_b, 1),
                ],
                reverse=True,
            )

            for _, which in fractions[:remaining]:
                if which == 0:
                    base_a += 1
                else:
                    base_b += 1

        out_a[i] = base_a
        out_b[i] = base_b

    if np.any(out_a + out_b > caps):
        raise AssertionError("Meal cap invariant violated.")

    return out_a, out_b


def train_production_catboost(
    features_path: str,
    last_complete_month: pd.Period,
):
    """
    Train hai CatBoost model riêng cho N44 và N70.

    Các feature phải được tạo bằng feature_engineering.py theo quy tắc
    chỉ sử dụng thông tin có sẵn trước tháng của từng dòng.
    """
    df = pd.read_parquet(features_path)

    required = {"employee_id", "month", "N44", "N70", "department"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Feature dataset thiếu cột bắt buộc: {sorted(missing)}."
        )

    df = normalize_months(df)

    if "is_complete_month" in df.columns:
        df["is_complete_month"] = normalize_complete_flag(
            df["is_complete_month"]
        )
        df = df[df["is_complete_month"]].copy()

    df = df[df["month"] <= last_complete_month].copy()

    if df.empty:
        raise ValueError(
            f"Không có feature row hợp lệ đến {last_complete_month}."
        )

    if df.duplicated(["employee_id", "month"]).any():
        raise ValueError("Feature dataset có employee-month bị trùng.")

    for target_col in TARGET_COLUMNS:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df[TARGET_COLUMNS] = df[TARGET_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna(subset=TARGET_COLUMNS).copy()

    if df.empty:
        raise ValueError("Không có nhãn N44/N70 hợp lệ để train.")

    # Chỉ sử dụng danh sách feature đã định nghĩa rõ ràng.
    # Không đưa target tháng hiện tại hoặc số tiền tháng hiện tại vào X.
    missing_features = set(MODEL_FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(
            "Feature dataset thiếu các đặc trưng cần thiết: "
            f"{sorted(missing_features)}. "
            "Hãy chạy lại feature_engineering.py."
        )

    cat_cols = MODEL_FEATURES + ["department"]

    X = df[cat_cols].copy()
    X["department"] = X["department"].fillna("unknown").astype(str)

    numeric_cols = [col for col in MODEL_FEATURES]
    X[numeric_cols] = X[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    X[numeric_cols] = X[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    cat_idx = [cat_cols.index("department")]

    model_44 = CatBoostRegressor(
        iterations=250,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=42,
        verbose=0,
        thread_count=-1,
        cat_features=cat_idx,
        allow_writing_files=False,
    )

    model_70 = CatBoostRegressor(
        iterations=250,
        learning_rate=0.03,
        depth=5,
        loss_function="MAE",
        random_seed=42,
        verbose=0,
        thread_count=-1,
        cat_features=cat_idx,
        allow_writing_files=False,
    )

    model_44.fit(X, df["N44"])
    model_70.fit(X, df["N70"])

    print(
        f"Đã train CatBoost trên {len(df):,} dòng "
        f"đến tháng {last_complete_month}."
    )

    return model_44, model_70, MODEL_FEATURES.copy(), cat_cols


def run_predict_next_month(
    monthly_path="data/processed/monthly_dataset.parquet",
    features_path="data/processed/features_monthly.parquet",
    output_dir="results/forecast",
    hr_path="data/hr.xlsx",
):
    """hr_path: danh sách nhân viên đang làm việc; None = không lọc."""
    monthly = pd.read_parquet(monthly_path)

    if monthly.empty:
        raise ValueError(f"Monthly dataset rỗng: {monthly_path}")

    monthly = normalize_months(monthly)

    last_complete_month = get_last_complete_month(monthly)
    forecast_month = last_complete_month + 1

    print(f"Tháng lịch sử hoàn chỉnh gần nhất: {last_complete_month}")
    print(f"=> Dự báo cho tháng: {forecast_month}")

    check_holiday_coverage(forecast_month)

    forecast_df = build_forecast_feature_row(
        monthly,
        forecast_month,
        last_complete_month,
    )

    if forecast_df.empty:
        raise ValueError("Cohort dự báo rỗng.")

    if (
        forecast_df["employee_id"].isna().any()
        or forecast_df["employee_id"].duplicated().any()
    ):
        raise ValueError("Cohort có employee_id thiếu hoặc trùng.")

    lag_cols = [
        f"{col}_lag_{lag}"
        for col in LAG_SOURCE_COLS
        for lag in (1, 2, 3)
    ]
    has_full_history = forecast_df[lag_cols].notna().all(axis=1)

    print(
        f"Cohort: {len(forecast_df):,} nhân viên; "
        f"đủ 3 tháng liên tiếp: {int(has_full_history.sum()):,}; "
        f"cold-start: {int((~has_full_history).sum()):,}"
    )

    model_44, model_70, numeric_features, cat_cols = (
        train_production_catboost(
            features_path,
            last_complete_month,
        )
    )

    X_pred = forecast_df[cat_cols].copy()
    X_pred["department"] = X_pred["department"].fillna(
        "unknown"
    ).astype(str)

    X_pred[numeric_features] = X_pred[numeric_features].apply(
        pd.to_numeric, errors="coerce"
    )
    X_pred[numeric_features] = X_pred[numeric_features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    pred_n44 = np.asarray(
        model_44.predict(X_pred),
        dtype=np.float64,
    )
    pred_n70 = np.asarray(
        model_70.predict(X_pred),
        dtype=np.float64,
    )

    # Fallback cho nhân viên thiếu lịch sử liên tiếp.
    pred_n44 = np.where(
        has_full_history,
        pred_n44,
        forecast_df["dept_enc_n44"].to_numpy(dtype=float),
    )
    pred_n70 = np.where(
        has_full_history,
        pred_n70,
        forecast_df["dept_enc_n70"].to_numpy(dtype=float),
    )

    meal_cap = max_meals_in_month(forecast_month)
    pred_n44, pred_n70 = cap_and_round_meals(
        pred_n44,
        pred_n70,
        np.full(len(forecast_df), meal_cap, dtype=int),
    )

    forecast_df["pred_N44"] = pred_n44
    forecast_df["pred_N70"] = pred_n70
    forecast_df["pred_cost"] = (
        pred_n44 * PRICE_44 + pred_n70 * PRICE_70
    )
    forecast_df["source"] = np.where(
        has_full_history,
        "CatBoost",
        "Group-Mean (cold-start)",
    )

    if (
        forecast_df["pred_N44"].lt(0).any()
        or forecast_df["pred_N70"].lt(0).any()
    ):
        raise AssertionError("Số bữa ăn dự báo không được âm.")

    if (
        forecast_df["pred_N44"] + forecast_df["pred_N70"] > meal_cap
    ).any():
        raise AssertionError(
            "Tổng số bữa ăn dự báo vượt số ngày dương lịch của tháng."
        )

    # Loại nhân viên không còn trong danh sách đang làm việc (hr.xlsx).
    excluded_df = forecast_df.iloc[0:0]
    if hr_path is not None:
        if Path(hr_path).exists():
            forecast_df, excluded_df = split_by_hr_roster(forecast_df, hr_path)
            print(
                f"Lọc theo {hr_path}: giữ {len(forecast_df):,}, "
                f"loại {len(excluded_df):,} nhân viên không còn trong danh sách "
                f"(dự báo {int(excluded_df['pred_cost'].sum()):,} đ bị bỏ)."
            )
        else:
            print(f"[CẢNH BÁO] Không thấy {hr_path}: bỏ qua bước lọc nhân viên đã nghỉ.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    month_tag = str(forecast_month)

    employee_out = forecast_df[
        [
            "employee_id",
            "department",
            "month",
            "pred_N44",
            "pred_N70",
            "pred_cost",
            "source",
        ]
    ].rename(columns={"month": "forecast_month"})

    employee_path = output_dir / f"forecast_{month_tag}_employee.csv"
    employee_out.to_csv(employee_path, index=False)

    excluded_path = output_dir / f"forecast_{month_tag}_excluded_not_in_hr.csv"
    if len(excluded_df) > 0:
        excluded_df[
            ["employee_id", "department", "month", "pred_N44", "pred_N70", "pred_cost", "source"]
        ].rename(columns={"month": "forecast_month"}).to_csv(excluded_path, index=False)
    elif excluded_path.exists():
        excluded_path.unlink()  # tránh để lại file cũ gây hiểu nhầm

    department_summary = (
        forecast_df.groupby("department", dropna=False)
        .agg(
            so_nhan_vien=("employee_id", "nunique"),
            tong_du_bao=("pred_cost", "sum"),
        )
        .reset_index()
        .sort_values("tong_du_bao", ascending=False)
    )

    department_path = (
        output_dir / f"forecast_{month_tag}_department.csv"
    )
    department_summary.to_csv(department_path, index=False)

    company_total = int(forecast_df["pred_cost"].sum())

    print("\n" + "=" * 70)
    print(f"DỰ BÁO NGÂN SÁCH CĂN-TIN — THÁNG {forecast_month}")
    print("=" * 70)
    print(f"Tổng dự báo toàn công ty : {company_total:,.0f} đ")
    print(f"Số nhân viên dự báo      : {len(forecast_df):,}")
    print("Top 5 phòng ban theo dự báo chi phí:")
    print(
        department_summary.head(5).to_string(
            index=False,
            formatters={
                "tong_du_bao": "{:,.0f} đ".format,
            },
        )
    )
    print(f"Đã lưu chi tiết nhân viên: {employee_path}")
    print(f"Đã lưu tổng hợp phòng ban: {department_path}")
    print(
        "\n⚠️ Dự báo chỉ phục vụ lập ngân sách/chuẩn bị nguyên liệu, "
        "không dùng để trừ tiền nhân viên."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly",
        default="data/processed/monthly_dataset.parquet",
    )
    parser.add_argument(
        "--features",
        default="data/processed/features_monthly.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="results/forecast",
    )
    parser.add_argument(
        "--hr",
        default="data/hr.xlsx",
        help="Danh sách nhân viên đang làm việc (cột 'Mã NV'). Người không có trong file bị loại.",
    )
    parser.add_argument(
        "--no-hr-filter",
        action="store_true",
        help="Không lọc theo danh sách HR.",
    )
    args = parser.parse_args()

    run_predict_next_month(
        args.monthly,
        args.features,
        args.output_dir,
        None if args.no_hr_filter else args.hr,
    )