"""
Bước 2: EDA (Exploratory Data Analysis)
========================================
Phân tích khám phá trên bảng lưới nhân viên x ngày (đầu ra của Bước 1) và
log đã làm sạch ở mức từng bữa ăn (để phân tích giờ ăn / ca ăn).

Mục tiêu của bước này KHÔNG phải để xây mô hình, mà để:
- Kiểm chứng các giả thuyết trước khi đưa vào feature engineering
  (vd: có thật sự khác biệt ngày thường/cuối tuần? phòng ban có ăn Âu
  nhiều hơn không? có nhóm nhân viên tenure ngắn cần xử lý riêng không?)
- Phát hiện bất thường còn sót lại sau Bước 1.

Cách dùng:
    python eda.py --raw-log data/raw/MealHistory_UTF8.csv \
                   --grid data/processed/daily_grid.parquet \
                   --output-dir results

Yêu cầu: file eda.py phải nằm CÙNG THƯ MỤC với data_cleaning.py (để import
lại đúng logic làm sạch, tránh viết lại 2 lần và tránh 2 nơi lệch nhau).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # không cần màn hình hiển thị, chỉ xuất file ảnh
import matplotlib.pyplot as plt
import pandas as pd

from data_cleaning import load_raw_log, clean_log, MEAL_TYPE_MAP, PRICE_MAP

WEEKDAY_NAMES_VI = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]

# Ngưỡng để coi một nhân viên là "tenure ngắn" — cần ít nhất 3 tháng lịch sử
# để tạo lag features (M-1, M-2, M-3) như đã thiết kế ở bước feature
# engineering. Chỉnh lại nếu quy ước của bạn khác.
SHORT_TENURE_DAYS = 90


def _savefig(fig, output_dir: Path, name: str):
    path = output_dir / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 1. Thống kê tổng quan
# ----------------------------------------------------------------------

def section_overview(grid: pd.DataFrame, clean: pd.DataFrame, report: list):
    report.append("## 1. Tổng quan")
    report.append(f"- Số nhân viên: {grid['employee_id'].nunique():,}")
    report.append(f"- Khoảng thời gian: {grid['date'].min().date()} -> {grid['date'].max().date()}")
    report.append(f"- Số dòng lưới (nhân viên x ngày): {len(grid):,}")
    report.append(f"- Tỷ lệ có ăn trên toàn bộ lưới: {grid['has_checkin'].mean():.1%}")
    report.append(f"- Tổng số bữa ăn hợp lệ: {len(clean):,}")
    report.append(f"- Tổng chi tiêu ghi nhận: {clean['amount'].sum():,.0f} đ")
    report.append("")


# ----------------------------------------------------------------------
# 2. Tỷ lệ có ăn theo thứ trong tuần (không chỉ weekday/weekend nhị phân)
# ----------------------------------------------------------------------

def section_day_of_week(grid: pd.DataFrame, output_dir: Path, report: list):
    tmp = grid.copy()
    tmp["dow"] = tmp["date"].dt.dayofweek
    rate = tmp.groupby("dow")["has_checkin"].mean().reindex(range(7))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(WEEKDAY_NAMES_VI, rate.to_numpy(dtype=float))
    ax.set_ylabel("Tỷ lệ có ăn")
    ax.set_title("Tỷ lệ có ăn theo thứ trong tuần")
    ax.set_ylim(0, 1)
    path = _savefig(fig, output_dir, "ty_le_an_theo_thu")

    report.append("## 2. Tỷ lệ có ăn theo thứ trong tuần")
    for name, val in zip(WEEKDAY_NAMES_VI, rate.to_numpy(dtype=float)):
        report.append(f"- {name}: {val:.1%}")
    report.append(f"\n![Tỷ lệ ăn theo thứ]({path.relative_to(output_dir)})\n")
    report.append(
        "**Diễn giải:** nếu cuối tuần thấp hẳn so với ngày thường, biến "
        "`is_weekend`/`is_workday` là đặc trưng quan trọng. Nếu có thứ cụ "
        "thể (vd thứ 2 hoặc thứ 6) khác biệt rõ so với các ngày thường "
        "khác, cân nhắc thêm đặc trưng `day_of_week` thay vì chỉ dùng cờ "
        "nhị phân weekday/weekend.\n"
    )


# ----------------------------------------------------------------------
# 3. Phân phối tỷ lệ chọn món — tổng thể và theo phòng ban
# ----------------------------------------------------------------------

def section_meal_type(clean: pd.DataFrame, output_dir: Path, report: list):
    overall = clean["meal_type"].value_counts(normalize=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        overall.index.astype(str).tolist(),
        overall.to_numpy(dtype=float),
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    ax.set_ylabel("Tỷ lệ")
    ax.set_title("Tỷ lệ chọn món (toàn bộ dữ liệu)")
    path1 = _savefig(fig, output_dir, "ty_le_mon_tong_the")

    by_dept = (
        clean.groupby(["department", "meal_type"]).size().unstack(fill_value=0)
    )
    by_dept_pct = by_dept.div(by_dept.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    by_dept_pct.plot(kind="bar", stacked=True, ax=ax,
                      color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Tỷ lệ")
    ax.set_title("Tỷ lệ chọn món theo phòng ban")
    ax.legend(title="Loại món", bbox_to_anchor=(1.02, 1), loc="upper left")
    path2 = _savefig(fig, output_dir, "ty_le_mon_theo_phong_ban")

    report.append("## 3. Tỷ lệ chọn món")
    for meal, pct in overall.items():
        report.append(f"- {meal}: {pct:.1%}")
    report.append(f"\n![Tỷ lệ món tổng thể]({path1.relative_to(output_dir)})")
    report.append(f"\n![Tỷ lệ món theo phòng ban]({path2.relative_to(output_dir)})\n")

    spread = by_dept_pct.get("Âu", pd.Series(dtype=float))
    if len(spread) > 0:
        report.append(
            f"**Diễn giải:** tỷ lệ chọn món Âu giữa các phòng ban dao động "
            f"từ {spread.min():.1%} đến {spread.max():.1%}. Nếu khoảng cách "
            f"này lớn, `department` là đặc trưng quan trọng cho việc dự "
            f"đoán N70. Nếu gần như đồng đều, thói quen cá nhân (Mã chấm "
            f"công) có thể quan trọng hơn phòng ban.\n"
        )


# ----------------------------------------------------------------------
# 4. Phân phối giờ ăn theo ca
# ----------------------------------------------------------------------

def section_hour_by_shift(clean: pd.DataFrame, output_dir: Path, report: list):
    tmp = clean.copy()
    tmp["hour"] = tmp["checkin_time"].dt.hour
    tmp["shift_label"] = tmp["shift"].fillna("Không xác định")

    fig, ax = plt.subplots(figsize=(8, 5))
    for shift_label, sub in tmp.groupby("shift_label"):
        ax.hist(
            sub["hour"].to_numpy(dtype=float),
            bins=range(0, 25),
            alpha=0.6,
            label=str(shift_label),
        )
    ax.set_xlabel("Giờ trong ngày")
    ax.set_ylabel("Số lượt ăn")
    ax.set_title("Phân phối giờ ăn theo ca")
    ax.legend(fontsize=8)
    path = _savefig(fig, output_dir, "gio_an_theo_ca")

    report.append("## 4. Giờ ăn theo ca")
    report.append(f"\n![Giờ ăn theo ca]({path.relative_to(output_dir)})\n")
    n_unknown_shift = (tmp["shift_label"] == "Không xác định").sum()
    if n_unknown_shift > 0:
        report.append(
            f"**Cảnh báo:** {n_unknown_shift:,} bữa ăn không nhận diện được "
            f"ca (`shift` = NaN). Kiểm tra `SHIFT_MAP` trong data_cleaning.py "
            f"nếu số này đáng kể.\n"
        )


# ----------------------------------------------------------------------
# 5. Phân phối chi tiêu theo tháng / theo nhân viên
# ----------------------------------------------------------------------

def section_monthly_spend(grid: pd.DataFrame, output_dir: Path, report: list):
    tmp = grid.copy()
    tmp["month"] = tmp["date"].dt.to_period("M")
    monthly = tmp.groupby(["employee_id", "month"])["total_amount"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(monthly["total_amount"], bins=30)
    ax.set_xlabel("Tổng tiền ăn / nhân viên / tháng (đ)")
    ax.set_ylabel("Số lượng (nhân viên x tháng)")
    ax.set_title("Phân phối tổng tiền ăn theo tháng")
    path1 = _savefig(fig, output_dir, "phan_phoi_tien_an_thang")

    monthly_company = tmp.groupby("month")["total_amount"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly_company.plot(kind="line", marker="o", ax=ax)
    ax.set_ylabel("Tổng tiền ăn toàn công ty (đ)")
    ax.set_title("Tổng tiền ăn toàn công ty theo tháng")
    ax.tick_params(axis="x", rotation=45)
    path2 = _savefig(fig, output_dir, "tong_tien_theo_thang")

    report.append("## 5. Chi tiêu theo tháng")
    report.append(monthly["total_amount"].describe().to_string())
    report.append(f"\n![Phân phối tiền ăn theo tháng]({path1.relative_to(output_dir)})")
    report.append(f"\n![Tổng tiền theo tháng]({path2.relative_to(output_dir)})\n")
    report.append(
        "**Diễn giải:** nếu phân phối lệch phải mạnh (nhiều nhân viên chi "
        "thấp, một số ít chi rất cao), cân nhắc dùng Huber loss hoặc biến "
        "đổi log cho biến mục tiêu ở bước mô hình hóa. Nhìn đường tổng theo "
        "tháng để phát hiện tháng bất thường (thiếu dữ liệu, nghỉ Tết...) "
        "trước khi chia train/val/test.\n"
    )


# ----------------------------------------------------------------------
# 6. Nhân viên tenure ngắn — cold-start / nghỉ việc giữa kỳ
# ----------------------------------------------------------------------

def section_tenure(grid: pd.DataFrame, report: list):
    span = grid.groupby("employee_id")["date"].agg(lambda s: (s.max() - s.min()).days)
    short = span[span < SHORT_TENURE_DAYS]

    report.append("## 6. Nhân viên có lịch sử ngắn (tenure)")
    report.append(f"- Ngưỡng coi là 'ngắn': dưới {SHORT_TENURE_DAYS} ngày hoạt động")
    report.append(f"- Số nhân viên tenure ngắn: {len(short):,} / {len(span):,} "
                   f"({len(short)/len(span):.1%})")

    if len(short) > 0:
        short_ids = short.index
        rate_short = grid[grid["employee_id"].isin(short_ids)]["has_checkin"].mean()
        report.append(f"- Tỷ lệ có ăn (nhóm tenure ngắn): {rate_short:.1%}")
        if len(short) < len(span):
            rate_normal = grid[~grid["employee_id"].isin(short_ids)]["has_checkin"].mean()
            report.append(f"- Tỷ lệ có ăn (nhóm còn lại): {rate_normal:.1%}")
        else:
            report.append(
                "- Không có nhân viên nào thuộc nhóm tenure dài để so sánh "
                "(mọi nhân viên trong tập này đều dưới ngưỡng)."
            )

    report.append(
        "\n**Diễn giải:** nhóm tenure ngắn không đủ 3 tháng lịch sử để tạo "
        "lag features (M-1, M-2, M-3) như thiết kế ban đầu. Cần quyết định: "
        "(a) loại nhóm này khỏi tập train/test chính, đánh giá riêng bằng "
        "Group-Mean baseline, hoặc (b) giữ lại nhưng dùng giá trị NaN/0 cho "
        "lag thiếu và để mô hình (đặc biệt cây quyết định) tự xử lý missing. "
        "Ghi rõ lựa chọn này vào phần giới hạn của báo cáo.\n"
    )


# ----------------------------------------------------------------------
# 7. Đối chiếu giá — QA
# ----------------------------------------------------------------------

def section_price_qa(clean: pd.DataFrame, report: list):
    expected = clean["meal_type"].map(PRICE_MAP)
    mismatch = clean[clean["amount"] != expected]
    report.append("## 7. Đối chiếu giá (QA)")
    report.append(f"- Số bữa có Thành tiền KHÁC bảng giá đã biết (44k/70k): {len(mismatch):,} "
                   f"({len(mismatch)/len(clean):.2%})")
    if len(mismatch) > 0:
        report.append("- Vài ví dụ:")
        report.append(mismatch[["employee_id", "date" if "date" in mismatch.columns
                                  else "checkin_time", "meal_type", "amount"]]
                       .head(5).to_string(index=False))
    report.append("")


# ----------------------------------------------------------------------
# HÀM CHẠY CHÍNH
# ----------------------------------------------------------------------

def run(raw_log_path: str, grid_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_log(raw_log_path)
    clean, _ = clean_log(raw)
    clean["date"] = clean["checkin_time"].dt.normalize()

    if grid_path.endswith(".parquet"):
        grid = pd.read_parquet(grid_path)
    else:
        grid = pd.read_csv(grid_path, parse_dates=["date"])

    report: list[str] = ["# Báo cáo EDA — Dự đoán tiền ăn nhân viên\n"]

    section_overview(grid, clean, report)
    section_day_of_week(grid, output_dir, report)
    section_meal_type(clean, output_dir, report)
    section_hour_by_shift(clean, output_dir, report)
    section_monthly_spend(grid, output_dir, report)
    section_tenure(grid, report)
    section_price_qa(clean, report)

    report_path = output_dir / "eda_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Đã ghi báo cáo: {report_path}")
    print(f"Ảnh biểu đồ nằm trong: {output_dir / 'figures'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run(args.raw_log, args.grid, Path(args.output_dir))