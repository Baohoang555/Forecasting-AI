"""
Chạy toàn bộ pipeline dự báo suất ăn (10 bước) bằng MỘT lệnh.

Chạy từ thư mục gốc dự án (nơi có thư mục src/ và data/):

    python run_pipeline.py                       # chạy đủ 10 bước
    python run_pipeline.py --skip-dl             # bỏ DL (bước 6) và các bước phụ thuộc DL (7, 8)
    python run_pipeline.py --from-step 9         # chỉ dự báo tháng tới và đối chiếu
    python run_pipeline.py --only 1 2 3          # chỉ chạy các bước chỉ định
    python run_pipeline.py --check               # chỉ kiểm tra môi trường/dữ liệu, không chạy
    python run_pipeline.py --dry-run             # in ra các lệnh sẽ chạy
    python run_pipeline.py --archive-stale       # chuyển các file results/metrics/*_new.csv cũ vào thư mục lưu trữ

Các bước:
    1 data_cleaning          2 monthly_aggregation     3 feature_engineering
    4 baselines              5 ml_models               6 dl_models (cần torch)
    7 ensemble               8 evaluate                9 predict_next_month
    10 monitor_forecast_accuracy  (tự đối chiếu mọi forecast mà tháng thực tế đã có đủ dữ liệu)

Log đầy đủ được lưu ở results/logs/pipeline_<thời gian>.log
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

GRID = "data/processed/daily_grid.parquet"
MONTHLY = "data/processed/monthly_dataset.parquet"
FEATURES = "data/processed/features_monthly.parquet"
FORECAST_DIR = "results/forecast"
METRICS_DIR = "results/metrics"

STEP_NAMES = {
    1: "Làm sạch log eCMS",
    2: "Tổng hợp theo tháng",
    3: "Tạo đặc trưng",
    4: "Đánh giá baseline",
    5: "Đánh giá mô hình ML",
    6: "Đánh giá mô hình DL",
    7: "Đánh giá ensemble",
    8: "Tổng hợp metric, phân khúc, outlier",
    9: "Dự báo tháng tới",
    10: "Đối chiếu dự báo với thực tế",
}
DL_DEPENDENT = {6, 7, 8}


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------
def ensure_utf8_stdio() -> None:
    """Bật UTF-8 cho console trên Windows để in tiếng Việt không phát sinh UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ensure_utf8_stdio()


class Tee:
    """Ghi đồng thời ra màn hình và file log."""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(log_path, "w", encoding="utf-8")
        self.path = log_path

    def write(self, text: str = "", end: str = "\n"):
        sys.stdout.write(text + end)
        sys.stdout.flush()
        self.fh.write(text + end)
        self.fh.flush()

    def close(self):
        self.fh.close()


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"  # tránh lỗi tiếng Việt/emoji khi chạy trên Windows
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"  # hiển thị output của từng bước ngay khi có
    return env


def run_command(cmd: list[str], out: Tee) -> int:
    out.write("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=child_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        out.write(line.rstrip("\n"))
    return proc.wait()


# ----------------------------------------------------------------------------
# Kiểm tra môi trường / dữ liệu (preflight)
# ----------------------------------------------------------------------------
def preflight(args, steps: list[int], out: Tee) -> bool:
    ok = True
    out.write("=== KIỂM TRA TRƯỚC KHI CHẠY ===")
    if not (ROOT / "src").is_dir() or not (ROOT / "data").is_dir():
        out.write(f"[LỖI] {ROOT} không giống thư mục gốc dự án (thiếu src/ hoặc data/).")
        return False
    out.write(f"Thư mục dự án : {ROOT}")
    out.write(f"Python        : {sys.version.split()[0]} ({PY})")

    needed = ["pandas", "numpy", "pyarrow", "scipy", "sklearn", "openpyxl"]
    if any(s in steps for s in (5, 7, 8, 9)):
        needed += ["lightgbm", "catboost"]
    if any(s in steps for s in DL_DEPENDENT):
        needed += ["torch"]
    missing = [m for m in dict.fromkeys(needed) if not has_module(m)]
    if missing:
        pip_names = {"sklearn": "scikit-learn"}
        out.write("[LỖI] Thiếu thư viện: " + ", ".join(missing))
        out.write("       Cài bằng: pip install " + " ".join(pip_names.get(m, m) for m in missing))
        ok = False
    else:
        out.write("Thư viện      : đủ")

    if 1 in steps:
        raw = ROOT / args.input
        if raw.exists():
            out.write(f"Log eCMS      : {args.input} ({raw.stat().st_size / 1e6:.1f} MB)")
        else:
            out.write(f"[LỖI] Không thấy file log đầu vào: {args.input}")
            candidates = sorted((ROOT / "data" / "raw").glob("*.csv"))
            if candidates:
                out.write("       Các file CSV có trong data/raw: " + ", ".join(c.name for c in candidates))
            ok = False

    holiday_file = ROOT / "data" /"raw"/ "relax_day.xlsx"
    if holiday_file.exists():
        sys.path.insert(0, str(ROOT / "src"))
        try:
            import calendar_utils  # noqa: WPS433

            table = calendar_utils.load_calendar_table()
            out.write(
                f"Lịch nghỉ lễ  : {len(table)} ngày, {table['date'].min().date()} -> {table['date'].max().date()} "
                f"(data/raw/relax_day.xlsx)"
            )
        except Exception as exc:  # noqa: BLE001
            out.write(f"[LỖI] Không đọc được lịch nghỉ lễ: {exc}")
            ok = False
    else:
        out.write("[CẢNH BÁO] Không thấy data/raw/relax_day.xlsx -> sẽ dùng danh sách lễ cũ (chỉ đến 2026-09-02).")

    hr_file = ROOT / args.hr
    if 9 in steps and not args.no_hr_filter:
        if hr_file.exists():
            mtime = datetime.fromtimestamp(hr_file.stat().st_mtime).strftime("%Y-%m-%d")
            out.write(f"Danh sách HR  : {args.hr} (sửa lần cuối {mtime}) - dùng để loại người đã nghỉ khỏi dự báo")
        else:
            out.write(f"[CẢNH BÁO] Không thấy {args.hr}: bước 9 sẽ bỏ qua việc lọc nhân viên đã nghỉ.")

    stale = sorted((ROOT / METRICS_DIR).glob("*_new.csv"))
    if stale:
        out.write(f"[LƯU Ý] Có {len(stale)} file '*_new.csv' trong {METRICS_DIR}. Dashboard (app.py) ưu tiên đọc "
                  f"các file này nên sẽ HIỂN THỊ SỐ CŨ dù pipeline đã chạy lại.")
        out.write("        Dùng --archive-stale để chuyển chúng vào thư mục lưu trữ (không xoá).")
    out.write("")
    return ok


def archive_stale(out: Tee) -> None:
    stale = sorted((ROOT / METRICS_DIR).glob("*_new.csv"))
    if not stale:
        out.write("Không có file '*_new.csv' cũ cần lưu trữ.")
        return
    dest = ROOT / METRICS_DIR / f"_archive_{datetime.now():%Y%m%d_%H%M%S}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in stale:
        shutil.move(str(f), dest / f.name)
    out.write(f"Đã chuyển {len(stale)} file '*_new.csv' cũ vào {dest.relative_to(ROOT)}")


# ----------------------------------------------------------------------------
# Định nghĩa lệnh từng bước
# ----------------------------------------------------------------------------
def build_commands(step: int, args) -> list[list[str]]:
    if step == 1:
        cmd = [PY, "src/data_cleaning.py", "--input", args.input, "--output", GRID,
               "--removed-log", "data/processed/removed_rows.csv",
               "--reinstated-log", "data/processed/reinstated_rows.csv"]
        if args.start_date:
            cmd += ["--start-date", args.start_date]
        return [cmd]
    if step == 2:
        cmd = [PY, "src/monthly_aggregation.py", "--grid", GRID, "--output", MONTHLY]
        if args.first_complete_month:
            cmd += ["--first-complete-month", args.first_complete_month]
        return [cmd]
    if step == 3:
        return [[PY, "src/feature_engineering.py"]]
    if step == 4:
        return [[PY, "src/baselines.py"]]
    if step == 5:
        return [[PY, "-m", "src.models.ml_models"]]
    if step == 6:
        return [[PY, "-m", "src.models.dl_models"]]
    if step == 7:
        return [[PY, "src/ensemble.py"]]
    if step == 8:
        return [[PY, "src/evaluate.py"]]
    if step == 9:
        cmd = [PY, "src/predict_next_month.py", "--monthly", MONTHLY, "--features", FEATURES,
               "--output-dir", FORECAST_DIR]
        cmd += ["--no-hr-filter"] if args.no_hr_filter else ["--hr", args.hr]
        return [cmd]
    raise ValueError(step)


def monitor_commands(out: Tee) -> list[list[str]]:
    """Bước 10: đối chiếu mọi forecast mà tháng thực tế đã có đủ dữ liệu trong monthly dataset."""
    import pandas as pd

    monthly_path = ROOT / MONTHLY
    if not monthly_path.exists():
        out.write(f"[BỎ QUA] Chưa có {MONTHLY}.")
        return []
    monthly = pd.read_parquet(monthly_path, columns=["month"])
    available = {str(m) for m in pd.PeriodIndex(monthly["month"], freq="M").unique()}
    last = max(available) if available else "?"

    commands = []
    forecasts = sorted((ROOT / FORECAST_DIR).glob("forecast_*_employee.csv"))
    if not forecasts:
        out.write(f"[BỎ QUA] Chưa có file dự báo nào trong {FORECAST_DIR}.")
    for f in forecasts:
        match = re.fullmatch(r"forecast_(\d{4}-\d{2})_employee\.csv", f.name)
        if not match:
            continue
        month = match.group(1)
        if month in available:
            commands.append([PY, "src/monitor_forecast_accuracy.py", "--forecast",
                             str(f.relative_to(ROOT)), "--monthly", MONTHLY, "--output-dir", METRICS_DIR])
        else:
            out.write(f"[CHƯA ĐỐI CHIẾU ĐƯỢC] {f.name}: tháng {month} chưa có đủ số liệu thực tế "
                      f"(dữ liệu hoàn chỉnh đến {last}). Sau khi export log đủ tháng {month}, chạy lại "
                      f"`python run_pipeline.py --only 1 2 10`.")
    return commands


# ----------------------------------------------------------------------------
# Điều phối
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Chạy toàn bộ pipeline dự báo suất ăn.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--input", default="data/raw/MealHistory_UTF8.csv", help="File log eCMS (CSV, UTF-8)")
    ap.add_argument("--hr", default="data/raw/hr.xlsx", help="Danh sách nhân viên đang làm việc")
    ap.add_argument("--no-hr-filter", action="store_true", help="Không lọc dự báo theo danh sách HR")
    ap.add_argument("--start-date", default=None,
                    help="Ngày bắt đầu phân tích (mặc định theo data_cleaning: 2025-08-01; 'none' = không giới hạn)")
    ap.add_argument("--first-complete-month", default=None, help="YYYY-MM: bỏ các tháng trước tháng này ở bước 2")
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--to-step", type=int, default=10)
    ap.add_argument("--only", type=int, nargs="+", help="Chỉ chạy các bước này")
    ap.add_argument("--skip-dl", action="store_true",
                    help="Bỏ bước 6 (DL) và các bước phụ thuộc DL (7 ensemble, 8 evaluate)")
    ap.add_argument("--keep-going", action="store_true", help="Không dừng khi một bước lỗi")
    ap.add_argument("--archive-stale", action="store_true", help="Chuyển results/metrics/*_new.csv cũ vào thư mục lưu trữ")
    ap.add_argument("--check", action="store_true", help="Chỉ kiểm tra môi trường/dữ liệu")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ in lệnh sẽ chạy")
    args = ap.parse_args()

    steps = sorted(set(args.only)) if args.only else [s for s in range(1, 11) if args.from_step <= s <= args.to_step]
    bad = [s for s in steps if s not in STEP_NAMES]
    if bad:
        ap.error(f"Bước không hợp lệ: {bad} (chỉ 1..10)")
    skipped_dl = []
    if args.skip_dl:
        skipped_dl = [s for s in steps if s in DL_DEPENDENT]
        steps = [s for s in steps if s not in DL_DEPENDENT]

    out = Tee(ROOT / "results" / "logs" / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
    try:
        out.write(f"Các bước sẽ chạy: {steps}" + (f" (bỏ qua do --skip-dl: {skipped_dl})" if skipped_dl else ""))
        if not preflight(args, steps, out):
            out.write("Dừng vì kiểm tra thất bại. Sửa các mục [LỖI] ở trên rồi chạy lại.")
            return 2
        if args.check:
            out.write("Kiểm tra xong, mọi thứ sẵn sàng.")
            return 0
        if args.archive_stale and not args.dry_run:
            archive_stale(out)

        results = []
        failed = False
        for step in steps:
            out.write("")
            out.write("=" * 78)
            out.write(f"BƯỚC {step}/10: {STEP_NAMES[step]}")
            out.write("=" * 78)
            commands = monitor_commands(out) if step == 10 else build_commands(step, args)
            if not commands:
                results.append((step, "BỎ QUA", 0.0))
                continue
            t0 = time.time()
            code = 0
            for cmd in commands:
                if args.dry_run:
                    out.write("$ " + " ".join(cmd))
                    continue
                code = run_command(cmd, out)
                if code != 0:
                    break
            elapsed = time.time() - t0
            status = "DRY-RUN" if args.dry_run else ("OK" if code == 0 else f"LỖI (mã {code})")
            results.append((step, status, elapsed))
            if code != 0:
                failed = True
                out.write(f"[LỖI] Bước {step} thất bại (mã thoát {code}).")
                if not args.keep_going:
                    break

        out.write("")
        out.write("=" * 78)
        out.write("TỔNG KẾT")
        out.write("=" * 78)
        for step, status, secs in results:
            out.write(f"  Bước {step:>2}  {STEP_NAMES[step]:<40} {status:<14} {secs:>7.1f}s")
        for step in skipped_dl:
            out.write(f"  Bước {step:>2}  {STEP_NAMES[step]:<40} {'BỎ QUA (--skip-dl)':<14}")
        if not failed and not args.dry_run:
            out.write("")
            out.write("Kết quả: results/metrics/ (metric), results/forecast/ (dự báo), data/processed/ (dữ liệu trung gian).")
            out.write("Xem dashboard:  streamlit run app.py")
        out.write(f"Log: {out.path.relative_to(ROOT)}")
        return 1 if failed else 0
    finally:
        out.close()


if __name__ == "__main__":
    sys.exit(main())
