"""
Lịch nghỉ lễ của công ty - nguồn dữ liệu duy nhất cho toàn pipeline.

Nguồn:
  1. data/relax_day.xlsx        : lịch nghỉ lễ/Tết (HR/Hành chính cung cấp).
       Cột (theo thứ tự): Năm | Dịp lễ/Tết | Thời gian nghỉ | Số ngày nghỉ
       Ô "Thời gian nghỉ" có thể là:
         - chuỗi  "dd/mm/yyyy - dd/mm/yyyy"  (một khoảng ngày)
         - chuỗi  "dd/mm/yyyy"               (một ngày)
         - ô ngày của Excel                  (một ngày)
  2. data/company_days_off.csv  : (tuỳ chọn) ngày nghỉ/giảm hoạt động RIÊNG của công ty
       không nằm trong lịch lễ (ví dụ tuần trước Tết). Cột: from_date,to_date,name
       Dòng bắt đầu bằng '#' bị bỏ qua.

Cách dùng trong code khác:
    from calendar_utils import load_public_holidays
    PUBLIC_HOLIDAYS = load_public_holidays()      # pd.DatetimeIndex

Lưu ý ngữ nghĩa: "ngày nghỉ" ở đây chỉ dùng để tính `working_days` = số ngày
thứ Hai-thứ Sáu KHÔNG nghỉ. Thứ Bảy làm việc chưa được tính (quyết định hiện tại
của công ty: chưa đưa lịch thứ Bảy vào mô hình).
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLIDAY_FILE = PROJECT_ROOT / "data" / "relax_day.xlsx"
DEFAULT_COMPANY_DAYS_FILE = PROJECT_ROOT / "data" / "company_days_off.csv"

# Đường dẫn có thể ghi đè bằng biến môi trường (hữu ích khi chạy trên máy khác).
ENV_HOLIDAY_FILE = "MEAL_HOLIDAY_FILE"
ENV_COMPANY_DAYS_FILE = "MEAL_COMPANY_DAYS_FILE"

# Danh sách cũ (hard-code) - CHỈ dùng làm phương án dự phòng khi không có file lịch.
LEGACY_PUBLIC_HOLIDAYS = [
    "2025-01-01",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03",
    "2025-04-30", "2025-05-01",
    "2025-09-02",
    "2026-01-01",
    "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18",
    "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22",
    "2026-04-26", "2026-04-27",
    "2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03",
    "2026-08-31", "2026-09-01", "2026-09-02",
]

_SEPARATOR_RE = re.compile(r"\s*[-\u2013\u2014]\s*")  # '-', en dash, em dash


def _warn(message: str) -> None:
    print(f"[CẢNH BÁO][lịch nghỉ] {message}", file=sys.stderr)


def _parse_one_date(value, dayfirst: bool = True) -> pd.Timestamp:
    """Chuyển 1 giá trị (str hoặc datetime) thành ngày. Chuỗi dd/mm/yyyy hiểu là ngày/tháng/năm."""
    if isinstance(value, pd.Timestamp) or hasattr(value, "year"):
        return pd.Timestamp(value).normalize()
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", text):
        return pd.Timestamp(text).normalize()
    return pd.to_datetime(text, format="%d/%m/%Y", dayfirst=dayfirst).normalize()


def parse_period(value) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Trả về (ngày_bắt_đầu, ngày_kết_thúc) của ô 'Thời gian nghỉ'."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        raise ValueError("ô trống")
    if not isinstance(value, str):
        day = _parse_one_date(value)
        return day, day
    parts = [p for p in _SEPARATOR_RE.split(value.strip()) if p]
    if len(parts) == 1:
        day = _parse_one_date(parts[0])
        return day, day
    if len(parts) == 2:
        start, end = _parse_one_date(parts[0]), _parse_one_date(parts[1])
        if end < start:
            raise ValueError(f"ngày kết thúc {end.date()} trước ngày bắt đầu {start.date()}")
        return start, end
    raise ValueError(f"không hiểu định dạng: {value!r}")


def load_holiday_table(path=None) -> pd.DataFrame:
    """Đọc relax_day.xlsx thành bảng từng ngày: date, name, year_label, source.

    Có kiểm tra: số ngày của khoảng phải khớp cột 'Số ngày nghỉ' và năm của ngày phải khớp
    cột 'Năm'. Sai lệch chỉ cảnh báo (không dừng) để người dùng còn sửa file.
    """
    path = Path(path) if path else Path(os.environ.get(ENV_HOLIDAY_FILE, DEFAULT_HOLIDAY_FILE))
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file lịch nghỉ: {path}")

    raw = pd.read_excel(path, header=0)
    if raw.shape[1] < 3:
        raise ValueError(f"{path.name} cần ít nhất 3 cột (Năm, Dịp, Thời gian nghỉ).")
    raw = raw.iloc[:, :4].copy()
    raw.columns = ["year_label", "name", "period", "n_days"][: raw.shape[1]]
    if "n_days" not in raw.columns:
        raw["n_days"] = pd.NA
    raw = raw.dropna(subset=["period"]).reset_index(drop=True)

    rows = []
    for idx, r in raw.iterrows():
        excel_row = idx + 2  # +1 header, +1 vì Excel đếm từ 1
        try:
            start, end = parse_period(r["period"])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path.name}, dòng Excel {excel_row} ({r['name']}): {exc}") from exc

        days = pd.date_range(start, end, freq="D")
        stated = re.findall(r"\d+", str(r["n_days"])) if pd.notna(r["n_days"]) else []
        if stated and int(stated[0]) != len(days):
            _warn(
                f"{r['name']}: khoảng {start.date()}→{end.date()} có {len(days)} ngày "
                f"nhưng cột 'Số ngày nghỉ' ghi {stated[0]}."
            )
        year_label = pd.to_numeric(r["year_label"], errors="coerce")
        if pd.notna(year_label) and start.year not in (int(year_label), int(year_label) - 1, int(year_label) + 1):
            _warn(f"{r['name']}: năm của ngày bắt đầu ({start.year}) lệch cột 'Năm' ({int(year_label)}).")

        for d in days:
            rows.append({"date": d, "name": str(r["name"]), "year_label": year_label, "source": path.name})

    table = pd.DataFrame(rows)
    # Nếu 2 dịp trùng ngày thì giữ dòng đầu tiên.
    return table.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_company_days_off(path=None) -> pd.DataFrame:
    """Đọc ngày nghỉ/giảm hoạt động riêng của công ty (tuỳ chọn). Trả về bảng rỗng nếu không có."""
    path = Path(path) if path else Path(os.environ.get(ENV_COMPANY_DAYS_FILE, DEFAULT_COMPANY_DAYS_FILE))
    empty = pd.DataFrame(columns=["date", "name", "year_label", "source"])
    if not path.exists():
        return empty
    raw = pd.read_csv(path, comment="#", dtype=str, skip_blank_lines=True)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if raw.empty:
        return empty
    if "from_date" not in raw.columns:
        raise ValueError(f"{path.name} cần cột from_date (và tuỳ chọn to_date, name).")

    rows = []
    for idx, r in raw.iterrows():
        start = _parse_one_date(r["from_date"].strip())
        end_raw = r.get("to_date")
        end = _parse_one_date(end_raw.strip()) if isinstance(end_raw, str) and end_raw.strip() else start
        if end < start:
            raise ValueError(f"{path.name}, dòng {idx + 2}: to_date trước from_date.")
        name = r.get("name") if isinstance(r.get("name"), str) else "Ngày nghỉ riêng của công ty"
        for d in pd.date_range(start, end, freq="D"):
            rows.append({"date": d, "name": name, "year_label": d.year, "source": path.name})
    return pd.DataFrame(rows)


def load_calendar_table(holiday_path=None, company_days_path=None) -> pd.DataFrame:
    """Gộp lịch lễ + ngày nghỉ riêng. Nếu thiếu file lịch lễ thì dùng danh sách cũ và cảnh báo rõ."""
    try:
        holidays = load_holiday_table(holiday_path)
    except FileNotFoundError as exc:
        _warn(f"{exc}. ĐANG DÙNG DANH SÁCH LỄ CŨ (hard-code, chỉ đến 2026-09-02). "
              f"Hãy đặt relax_day.xlsx vào thư mục data/.")
        legacy_dates = pd.to_datetime(LEGACY_PUBLIC_HOLIDAYS)
        holidays = pd.DataFrame({
            "date": legacy_dates,
            "name": "Lịch lễ cũ (hard-code)",
            "year_label": legacy_dates.year,
            "source": "legacy",
        })
    company = load_company_days_off(company_days_path)
    # Không nối bảng rỗng: pandas 2.x cảnh báo FutureWarning khi concat với DataFrame rỗng/toàn NA.
    frames = [holidays] + ([company] if len(company) > 0 else [])
    table = frames[0].copy() if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return table.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_public_holidays(holiday_path=None, company_days_path=None) -> pd.DatetimeIndex:
    """DatetimeIndex các ngày nghỉ (lễ + ngày nghỉ riêng) - thay cho PUBLIC_HOLIDAYS hard-code."""
    table = load_calendar_table(holiday_path, company_days_path)
    return pd.DatetimeIndex(table["date"]).normalize()


def max_covered_year(holidays: pd.DatetimeIndex) -> int:
    """Năm lớn nhất có ngày nghỉ trong lịch (dùng để phát hiện lịch đã hết hạn)."""
    return int(pd.DatetimeIndex(holidays).year.max())