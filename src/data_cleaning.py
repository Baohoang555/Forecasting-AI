import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_utils import load_public_holidays

# ----------------------------------------------------------------------
# 1. CẤU HÌNH
# ----------------------------------------------------------------------

COLUMN_MAP = {
    "employee_id": "Mã chấm công",
    "department": "Phòng ban",
    "meal_type_raw": "Khẩu phần ăn",
    "bonus_meal_raw": "Khẩu phần ăn bồi dưỡng",
    "shift_raw": "Ca ăn",
    "amount": "Thành tiền",
    "cumulative_raw": "Tổng tích lũy",
    "checkin_time": "Thời gian ăn",
    "status": "Trạng thái",
    "reason": "Lý do",
}
VALID_STATUS = "Hợp lệ"

MEAL_TYPE_MAP = {
    "Vietnamese Meal": "Việt",
    "Vegetarian Meal": "Chay",
    "Western Meal": "Âu",
}
PRICE_MAP = {"Việt": 44_000, "Chay": 44_000, "Âu": 70_000}

SHIFT_MAP = {
    "Shift 1":           {"start": "05:20", "end": "14:20"},
    "Office Hour Shift": {"start": "07:00", "end": "17:00"},
    "Shift 2":           {"start": "14:40", "end": "23:40"},
    "Service Shift":     {"start": None, "end": None},
}

# Lịch nghỉ lễ được đọc từ data/relax_day.xlsx (+ data/company_days_off.csv nếu có).
# Xem src/calendar_utils.py. Không còn hard-code danh sách ngày lễ ở đây.
PUBLIC_HOLIDAYS = load_public_holidays()

# Hệ thống eCMS vận hành từ 07/2025 nhưng thực tế chỉ chạy đủ công suất từ ~21-22/07/2025
# (1-7/7: ~1 suất/ngày; 8-18/7: 4%-35% mức bình thường). Giai đoạn khởi động này
# KHÔNG đại diện cho nhu cầu thật nên loại khỏi dữ liệu phân tích.
# Dùng --start-date none để tắt, hoặc --start-date YYYY-MM-DD để đổi.
ANALYSIS_START_DATE = "2025-08-01"

# Bản ghi bị eCMS gắn "không hợp lệ" chỉ vì CHƯA CÓ LOG ĐIỂM DANH từ ePad (lỗi đồng bộ,
# điển hình 2025-09-11 và 2026-09-07): nhân viên thực sự đã ăn (có chọn món và thành tiền)
# nên được tính là suất hợp lệ. Mẫu regex so khớp không phân biệt hoa/thường và chịu được
# lỗi mất dấu tiếng Việt trong file xuất ("Ch?a có log ?i?m danh").
REINSTATE_REASON_PATTERNS = [r"log.*danh"]
_PAREN_RE = re.compile(r"\(([^)]+)\)")


# ----------------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU THÔ
# ----------------------------------------------------------------------

def load_raw_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # Tránh lỗi khoảng trắng thừa ở header CSV
    df.columns = df.columns.str.strip()

    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Không tìm thấy các cột {missing} trong file input. "
            f"Kiểm tra lại COLUMN_MAP cho khớp với file thật."
        )

    inv_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=inv_map)
    return df[list(COLUMN_MAP.keys())].copy()


# ----------------------------------------------------------------------
# 3. TRÍCH NHÃN TIẾNG ANH TRONG NGOẶC (dùng chung cho loại món & ca ăn)
# ----------------------------------------------------------------------

def _extract_paren_label(text) -> str | float:
    """Lấy nội dung trong dấu ngoặc đơn ĐẦU TIÊN của chuỗi.
    Trả về NaN nếu không tìm thấy hoặc input rỗng."""
    if pd.isna(text):
        return np.nan
    m = _PAREN_RE.search(str(text))
    return m.group(1).strip() if m else np.nan


def standardize_meal_type(series: pd.Series) -> pd.Series:
    labels = series.map(_extract_paren_label)
    return labels.map(MEAL_TYPE_MAP)  # không khớp -> NaN, để lộ ra khi QA


def standardize_shift(series: pd.Series) -> pd.Series:
    labels = series.map(_extract_paren_label)
    return labels.where(labels.isin(SHIFT_MAP.keys()), np.nan)


# ----------------------------------------------------------------------
# 4. LÀM SẠCH
# ----------------------------------------------------------------------

def clean_log(
    raw: pd.DataFrame,
    start_date: str | None = ANALYSIS_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (log_sach, log_bi_loai).

    start_date: bỏ các bản ghi trước ngày này (giai đoạn khởi động hệ thống). None = giữ tất cả.
    """
    df = raw.copy()
    removed_reasons = []

    # Sửa 1: Chuẩn hóa khoảng trắng và chữ hoa/thường tránh drop nhầm bản ghi hợp lệ
    status_clean = df["status"].fillna("").astype(str).str.strip()
    status_invalid = status_clean.str.lower() != VALID_STATUS.lower()

    # Khôi phục các bản ghi "không hợp lệ" chỉ vì chưa có log điểm danh (lỗi đồng bộ ePad).
    reason_clean = df["reason"].fillna("").astype(str)
    reinstate_mask = pd.Series(False, index=df.index)
    for pattern in REINSTATE_REASON_PATTERNS:
        reinstate_mask |= reason_clean.str.contains(pattern, case=False, regex=True)
    reinstate_mask &= status_invalid
    df["reinstated_attendance_sync"] = reinstate_mask

    invalid_mask = status_invalid & ~reinstate_mask
    removed_reasons.append(
        df[invalid_mask].assign(drop_reason="trang_thai_khong_hop_le")
    )
    df = df[~invalid_mask]

    df["employee_id"] = pd.to_numeric(df["employee_id"], errors="coerce").astype("Int64")
    bad_emp = df["employee_id"].isna()
    removed_reasons.append(df[bad_emp].assign(drop_reason="thieu_ma_nhan_vien"))
    df = df[~bad_emp]

    # Sửa 2: Parse thời gian linh hoạt (mixed format) tránh NaT khi gặp định dạng ngày khác nhau
    df["checkin_time"] = pd.to_datetime(df["checkin_time"], errors="coerce", format="mixed")
    bad_time = df["checkin_time"].isna()
    removed_reasons.append(df[bad_time].assign(drop_reason="khong_parse_duoc_thoi_gian"))
    df = df[~bad_time]

    if start_date is not None:
        before_start = df["checkin_time"] < pd.Timestamp(start_date)
        removed_reasons.append(df[before_start].assign(drop_reason="giai_doan_khoi_dong_he_thong"))
        df = df[~before_start]

    # Khi chưa có log điểm danh, nhân viên thường quẹt lại thẻ nhiều lần (cách nhau ~1 phút)
    # nhưng chỉ ăn 1 suất. Với các dòng được khôi phục: chỉ giữ lần quẹt đầu tiên mỗi nhân
    # viên-ngày, và bỏ hẳn nếu cùng ngày nhân viên đó đã có một suất hợp lệ thật.
    df["_day"] = df["checkin_time"].dt.normalize()
    is_reinstated = df["reinstated_attendance_sync"]
    if is_reinstated.any():
        genuine_days = set(zip(df.loc[~is_reinstated, "employee_id"], df.loc[~is_reinstated, "_day"]))
        re_rows = df[is_reinstated]
        has_genuine = pd.Series(
            [(e, d) in genuine_days for e, d in zip(re_rows["employee_id"], re_rows["_day"])],
            index=re_rows.index,
        )
        keep_candidates = re_rows[~has_genuine].sort_values("checkin_time")
        retry_dups = keep_candidates[keep_candidates.duplicated(["employee_id", "_day"], keep="first")]
        drop_reinstated = df.index.isin(has_genuine[has_genuine].index) | df.index.isin(retry_dups.index)
        removed_reasons.append(df[drop_reinstated].assign(drop_reason="quet_lai_khi_thieu_log_diem_danh"))
        df = df[~drop_reinstated]
    df = df.drop(columns="_day")

    df["meal_type"] = standardize_meal_type(df["meal_type_raw"])
    bad_meal = df["meal_type"].isna()
    removed_reasons.append(df[bad_meal].assign(drop_reason="khong_nhan_dien_duoc_loai_mon"))
    df = df[~bad_meal]

    df["shift"] = standardize_shift(df["shift_raw"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad_amount = df["amount"].isna() | (df["amount"] < 0)
    removed_reasons.append(df[bad_amount].assign(drop_reason="thanh_tien_khong_hop_le"))
    df = df[~bad_amount]

    # QA check đơn giá nghiệp vụ, giữ nguyên amount vì eCMS là ground truth
    expected_price = df["meal_type"].map(PRICE_MAP)
    df["price_mismatch"] = (df["amount"] != expected_price)
    n_mismatch = df["price_mismatch"].sum()
    if n_mismatch > 0:
        print(f"[CẢNH BÁO] {n_mismatch} dòng có Thành tiền KHÁC với bảng giá "
              f"44.000/70.000 đã biết — giữ nguyên amount theo Ground Truth của eCMS.")

    # Sửa 3: Khử trùng lặp an toàn bằng cách bổ sung 'amount' vào khóa định danh giao dịch
    dup_mask = df.duplicated(
        subset=["employee_id", "checkin_time", "meal_type", "shift", "amount"], keep="first"
    )
    removed_reasons.append(df[dup_mask].assign(drop_reason="trung_lap_ban_ghi"))
    df = df[~dup_mask]

    removed_df = pd.concat(removed_reasons, ignore_index=True) if removed_reasons else pd.DataFrame()
    return df.reset_index(drop=True), removed_df


# ----------------------------------------------------------------------
# 5. GỘP NHIỀU BỮA ĂN TRONG CÙNG 1 NGÀY -> BẢNG NGÀY
# ----------------------------------------------------------------------

def aggregate_daily(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Một nhân viên có thể ăn nhiều hơn 1 lần hợp lệ trong ngày (nhiều ca).
    Gộp lại thành 1 dòng/nhân viên/ngày: tổng tiền + số bữa từng loại."""
    df = clean_df.copy()
    df["date"] = df["checkin_time"].dt.normalize()

    daily = (
        df.groupby(["employee_id", "date"])
        .agg(
            department=("department", "last"),
            total_amount=("amount", "sum"),
            n_meals=("amount", "size"),
            n_viet=("meal_type", lambda s: (s == "Việt").sum()),
            n_chay=("meal_type", lambda s: (s == "Chay").sum()),
            n_au=("meal_type", lambda s: (s == "Âu").sum()),
        )
        .reset_index()
    )
    return daily


# ----------------------------------------------------------------------
# 6. DỰNG LƯỚI NHÂN VIÊN x NGÀY
# ----------------------------------------------------------------------

def build_employee_day_grid(daily: pd.DataFrame) -> pd.DataFrame:
    """Dựng lưới đầy đủ từ ngày xuất hiện đầu tiên (first_seen) đến mốc chốt
    dữ liệu của toàn hệ thống (global_max) theo Phương án A: nhân viên không có
    check-in vẫn là nhân viên công ty và được điền 0 suất ăn."""
    global_max = daily["date"].max()

    emp_info = (
        daily.sort_values("date")
        .groupby("employee_id")
        .agg(department=("department", "last"), first_seen=("date", "min"))
        .reset_index()
    )

    grids = []
    for _, row in emp_info.iterrows():
        start = row["first_seen"]
        # Sửa 4: Kéo dài lưới đến tận global_max thay vì ngắt ở last_seen của nhân viên
        end = global_max
        dates = pd.date_range(start, end, freq="D")
        grids.append(pd.DataFrame({
            "employee_id": row["employee_id"],
            "department": row["department"],
            "date": dates,
        }))
    grid = pd.concat(grids, ignore_index=True)

    grid = grid.merge(
        daily[["employee_id", "date", "total_amount", "n_meals", "n_viet", "n_chay", "n_au"]],
        on=["employee_id", "date"], how="left",
    )
    for col in ["total_amount", "n_meals", "n_viet", "n_chay", "n_au"]:
        grid[col] = grid[col].fillna(0)
    grid["has_checkin"] = (grid["n_meals"] > 0).astype(int)

    return grid


def add_calendar_features(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid.copy()
    grid["is_weekend"] = grid["date"].dt.dayofweek.isin([5, 6])
    grid["is_public_holiday"] = grid["date"].isin(PUBLIC_HOLIDAYS)
    grid["is_workday"] = ~(grid["is_weekend"] | grid["is_public_holiday"])
    return grid


# ----------------------------------------------------------------------
# 7. HÀM CHẠY CHÍNH
# ----------------------------------------------------------------------

def run(
    input_path: str,
    output_path: str,
    removed_log_path: str | None = None,
    start_date: str | None = ANALYSIS_START_DATE,
    reinstated_log_path: str | None = None,
) -> pd.DataFrame:
    raw = load_raw_log(input_path)
    clean, removed = clean_log(raw, start_date=start_date)
    daily = aggregate_daily(clean)
    grid = build_employee_day_grid(daily)
    grid = add_calendar_features(grid)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith(".parquet"):
        grid.to_parquet(output_path, index=False)
    else:
        grid.to_csv(output_path, index=False)

    if removed_log_path and len(removed) > 0:
        Path(removed_log_path).parent.mkdir(parents=True, exist_ok=True)
        removed.to_csv(removed_log_path, index=False)

    reinstated = clean[clean["reinstated_attendance_sync"]]
    if reinstated_log_path and len(reinstated) > 0:
        Path(reinstated_log_path).parent.mkdir(parents=True, exist_ok=True)
        reinstated[["employee_id", "checkin_time", "meal_type", "amount", "reason"]].to_csv(
            reinstated_log_path, index=False
        )

    print("=== TÓM TẮT LÀM SẠCH DỮ LIỆU ===")
    print(f"Ngày bắt đầu phân tích   : {start_date or 'không giới hạn'}")
    print(f"Lịch nghỉ lễ             : {len(PUBLIC_HOLIDAYS)} ngày "
          f"({PUBLIC_HOLIDAYS.min().date()} -> {PUBLIC_HOLIDAYS.max().date()})")
    print(f"Số dòng log thô          : {len(raw):,}")
    print(f"Số dòng log sau làm sạch : {len(clean):,}")
    print(f"Số dòng bị loại          : {len(removed):,}")
    if len(removed) > 0:
        print(removed["drop_reason"].value_counts().to_string())
    print(f"Khôi phục 'chưa có log điểm danh' (tính là suất hợp lệ): {len(reinstated):,} dòng")
    if len(reinstated) > 0:
        by_month = reinstated.groupby(reinstated["checkin_time"].dt.to_period("M")).size()
        print("   theo tháng: " + ", ".join(f"{m}: {n:,}" for m, n in by_month.items()))
    print()
    print("=== TÓM TẮT SAU KHI GỘP THEO NGÀY ===")
    print(f"Số nhân viên          : {daily['employee_id'].nunique():,}")
    print(f"Số dòng (NV x ngày ăn): {len(daily):,}")
    print(f"Số ngày có >1 bữa     : {(daily['n_meals'] > 1).sum():,} "
          f"({(daily['n_meals'] > 1).mean():.1%})")
    print()
    print("=== TÓM TẮT LƯỚI NHÂN VIÊN x NGÀY ===")
    print(f"Số dòng lưới  : {len(grid):,}")
    print(f"Tỷ lệ có ăn   : {grid['has_checkin'].mean():.1%}")
    print(f"Khoảng ngày   : {grid['date'].min().date()} -> {grid['date'].max().date()}")

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--removed-log", default=None)
    parser.add_argument("--reinstated-log", default=None,
                        help="CSV ghi các dòng 'chưa có log điểm danh' được khôi phục")
    parser.add_argument("--start-date", default=ANALYSIS_START_DATE,
                        help="Bỏ bản ghi trước ngày này (YYYY-MM-DD). 'none' = không giới hạn.")
    args = parser.parse_args()

    start = None if str(args.start_date).lower() == "none" else args.start_date
    run(args.input, args.output, args.removed_log, start, args.reinstated_log)