#!/usr/bin/env python3
"""
visualize_results.py — Vẽ biểu đồ cho TOÀN BỘ kết quả pipeline đã export.

Cấu trúc thư mục đầu vào (giống file figures.zip):
    results/
        metrics/    *.csv + final_predictions_evaluated.parquet
        forecast/   forecast_<YYYY-MM>_department.csv / _employee.csv / _excluded_not_in_hr.csv
        logs/       pipeline_<YYYYMMDD>_<HHMMSS>.log

Cách chạy:
    pip install pandas numpy matplotlib pyarrow
    python visualize_results.py                       # đọc ./results, ghi ./results/figures
    python visualize_results.py --results-dir D:/Meal/results
    python visualize_results.py --out-dir my_figs --dpi 200
    python visualize_results.py --show                # mở cửa sổ matplotlib thay vì chỉ lưu file

Kết quả: các file PNG đánh số 01_..., 02_... + index.html (thư viện ảnh để xem một lượt).
Mỗi biểu đồ chạy độc lập: thiếu file nào thì chỉ bỏ qua biểu đồ đó, không làm hỏng cả script.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import textwrap
import traceback
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------------
# Giao diện chung
# ----------------------------------------------------------------------------
GROUP_COLORS = {
    "Baseline": "#9aa5b1",
    "ML": "#2f6fdb",
    "DL": "#f0883e",
    "Ensemble": "#2da44e",
}
C_ACTUAL = "#334155"
C_PRED = "#2f6fdb"
C_BAD = "#d1453b"
C_GOOD = "#2da44e"
SEGMENT_ORDER = ["Không ăn (0 đ)", "Ăn ít (<300k)", "Ăn vừa (300k - 800k)", "Ăn nhiều (>=800k)"]
SEGMENT_COLORS = {
    "Không ăn (0 đ)": "#9aa5b1",
    "Ăn ít (<300k)": "#f0883e",
    "Ăn vừa (300k - 800k)": "#2f6fdb",
    "Ăn nhiều (>=800k)": "#2da44e",
}

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",  # hỗ trợ đầy đủ dấu tiếng Việt
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.frameon": False,
    }
)


def fmt_vnd(x, pos=None) -> str:
    """Định dạng tiền VND gọn: 1.23 tỷ / 456 tr / 78k."""
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:,.2f} tỷ"
    if a >= 1e7:
        return f"{x / 1e6:,.0f} tr"
    if a >= 1e6:
        return f"{x / 1e6:,.1f} tr"
    if a >= 1e3:
        return f"{x / 1e3:,.0f}k"
    return f"{x:,.0f}"


VND = FuncFormatter(fmt_vnd)


def wrap(s: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(str(s), width)) or str(s)


# ----------------------------------------------------------------------------
# Đọc dữ liệu
# ----------------------------------------------------------------------------
class Results:
    def __init__(self, root: Path):
        self.root = root
        self.metrics_dir = root / "metrics"
        self.forecast_dir = root / "forecast"
        self.logs_dir = root / "logs"

    def metric(self, name: str) -> pd.DataFrame:
        p = self.metrics_dir / name
        if not p.exists():
            raise FileNotFoundError(p)
        return pd.read_csv(p)

    def predictions(self) -> pd.DataFrame:
        p = self.metrics_dir / "final_predictions_evaluated.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_parquet(p)
        df["month"] = df["month"].astype(str)
        return df

    def forecast(self, kind: str) -> tuple[pd.DataFrame, str]:
        """kind: department | employee | excluded_not_in_hr. Lấy file của tháng mới nhất."""
        files = sorted(self.forecast_dir.glob(f"forecast_*_{kind}.csv"))
        if not files:
            raise FileNotFoundError(self.forecast_dir / f"forecast_*_{kind}.csv")
        p = files[-1]
        month = re.search(r"forecast_(\d{4}-\d{2})_", p.name).group(1)
        return pd.read_csv(p), month


def load_model_summaries(R: Results) -> pd.DataFrame:
    parts = []
    for fn, grp in [
        ("baselines_summary.csv", "Baseline"),
        ("ml_models_summary.csv", "ML"),
        ("dl_models_summary.csv", "DL"),
        ("ensemble_multi_summary.csv", "Ensemble"),
    ]:
        try:
            df = R.metric(fn).rename(columns={"Method": "Model"})
        except FileNotFoundError:
            continue
        df["Group"] = grp
        parts.append(df)
    if not parts:
        raise FileNotFoundError("Không có file summary nào của baseline/ML/DL/ensemble")
    return pd.concat(parts, ignore_index=True)


def parse_logs(logs_dir: Path) -> pd.DataFrame:
    """Đọc bảng tóm tắt 'Bước N <tên> <trạng thái> <giây>s' ở cuối mỗi file log."""
    step_re = re.compile(r"^\s*Bước\s+(\d+):?\s+(.+?)\s{2,}(\S.*?)(?:\s+([\d.]+)s)?\s*$")
    rows = []
    for p in sorted(logs_dir.glob("pipeline_*.log")):
        m = re.search(r"pipeline_(\d{8})_(\d{6})", p.name)
        ts = pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S") if m else pd.NaT
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            mm = step_re.match(line.rstrip("\r"))
            if mm:
                rows.append(
                    dict(
                        run=p.stem,
                        ts=ts,
                        step=int(mm.group(1)),
                        name=mm.group(2).strip(),
                        status=mm.group(3).strip(),
                        seconds=float(mm.group(4)) if mm.group(4) else 0.0,
                    )
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Đăng ký biểu đồ
# ----------------------------------------------------------------------------
FIGURES: list[tuple[str, str, callable]] = []


def figure(name: str, title: str):
    def deco(fn):
        FIGURES.append((name, title, fn))
        return fn

    return deco


def heatmap(ax, mat, xlabels, ylabels, cmap, vmin, vmax, fmt):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = im.cmap(im.norm(v))
            color = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9, color=color)
    return im


# ----------------------------------------------------------------------------
# 01-04: So sánh mô hình
# ----------------------------------------------------------------------------
@figure("01_model_leaderboard", "Bảng xếp hạng mô hình: WAPE / MAE / RMSE tiền")
def fig_leaderboard(R: Results):
    df = load_model_summaries(R).sort_values("WAPE_Cost_%").reset_index(drop=True)
    y = np.arange(len(df))
    colors = df["Group"].map(GROUP_COLORS)
    panels = [
        ("WAPE_Cost_%", "WAPE (%) — thấp hơn là tốt hơn", lambda v: f"{v:.1f}%"),
        ("MAE_Cost", "MAE tiền (đ)", lambda v: f"{v / 1e3:,.0f}k"),
        ("RMSE_Cost", "RMSE tiền (đ)", lambda v: f"{v / 1e3:,.0f}k"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.52 * len(df) + 2.4), sharey=True)
    for ax, (col, title, f) in zip(axes, panels):
        bars = ax.barh(y, df[col], color=colors)
        bars[0].set_edgecolor("black")
        bars[0].set_linewidth(1.8)
        for yi, v in zip(y, df[col]):
            ax.text(v, yi, " " + f(v), va="center", fontsize=8.5)
        if col != "WAPE_Cost_%":
            ax.xaxis.set_major_formatter(VND)
        ax.set_title(title)
        ax.set_xlim(0, df[col].max() * 1.18)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["Model"])
    axes[0].invert_yaxis()
    handles = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items() if g in set(df["Group"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("So sánh tất cả mô hình (xếp theo WAPE; viền đen = tốt nhất)", fontweight="bold", y=1.02)
    return fig


@figure("02_model_mae_meals", "MAE số suất ăn N44 / N70 theo mô hình")
def fig_mae_meals(R: Results):
    df = load_model_summaries(R).sort_values("WAPE_Cost_%").reset_index(drop=True)
    y = np.arange(len(df))
    colors = df["Group"].map(GROUP_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(df) + 2.2), sharey=True)
    for ax, col, title in zip(
        axes, ["MAE_N44", "MAE_N70"], ["MAE số suất N44 (suất 44k)", "MAE số suất N70 (suất 70k)"]
    ):
        ax.barh(y, df[col], color=colors)
        for yi, v in zip(y, df[col]):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8.5)
        ax.set_title(title)
        ax.set_xlim(0, df[col].max() * 1.18)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["Model"])
    axes[0].invert_yaxis()
    handles = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items() if g in set(df["Group"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Sai số dự đoán số suất ăn theo loại suất (thứ tự giống biểu đồ 01)", fontweight="bold", y=1.02)
    return fig


@figure("03_ensemble_by_month", "Ensemble theo từng tháng test")
def fig_ensemble_month(R: Results):
    df = R.metric("ensemble_multi_by_month.csv")
    methods = list(dict.fromkeys(df["Method"]))
    months = sorted(df["Month"].unique())
    x = np.arange(len(months))
    w = 0.8 / len(methods)
    palette = ["#2da44e", "#8957e5", "#f0883e", "#2f6fdb"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title, f in [
        (axes[0], "WAPE_Cost_%", "WAPE tiền (%)", lambda v: f"{v:.1f}"),
        (axes[1], "MAE_Cost", "MAE tiền (đ)", lambda v: f"{v / 1e3:,.0f}k"),
    ]:
        for i, m in enumerate(methods):
            sub = df[df["Method"] == m].set_index("Month").reindex(months)
            bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, sub[col], w, label=m, color=palette[i % 4])
            for b, v in zip(bars, sub[col]):
                ax.text(b.get_x() + b.get_width() / 2, v, f(v), ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(months)
        ax.set_title(title)
        ax.set_ylim(0, df[col].max() * 1.3)
        if col == "MAE_Cost":
            ax.yaxis.set_major_formatter(VND)
    axes[0].legend(loc="upper center", ncol=2, fontsize=8.5)
    fig.suptitle("Ensemble S0 (trung bình đơn giản) vs S2 (trọng số tối ưu lồi) theo tháng", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("04_ensemble_pairs", "Các cặp ensemble (E0 trung bình đơn / E1 trọng số tối ưu)")
def fig_ensemble_pairs(R: Results):
    df = R.metric("ensemble_pairs_summary.csv").sort_values("WAPE_Cost_%").reset_index(drop=True)
    y = np.arange(len(df))
    colors = df["Technique"].map(lambda t: "#2da44e" if t.startswith("E0") else "#8957e5")
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(df) + 2))
    ax.hlines(y, df["WAPE_Cost_%"].min() - 0.05, df["WAPE_Cost_%"], color=colors, alpha=0.5, lw=2)
    ax.scatter(df["WAPE_Cost_%"], y, color=colors, s=55, zorder=3)
    for yi, (v, w) in enumerate(zip(df["WAPE_Cost_%"], df["Weight_A"])):
        txt = f" {v:.2f}%" + (f"   (trọng số mô hình đầu = {w:.2f})" if pd.notna(w) else "")
        ax.text(v, yi, txt, va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Pair"])
    ax.invert_yaxis()
    ax.set_xlim(df["WAPE_Cost_%"].min() - 0.05, df["WAPE_Cost_%"].max() + 0.9)
    ax.set_xlabel("WAPE tiền (%) — trục x bắt đầu từ giá trị nhỏ nhất để thấy khác biệt nhỏ")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#2da44e", label="E0 — trung bình đơn giản"),
            Line2D([], [], marker="o", ls="", color="#8957e5", label="E1 — trọng số tối ưu"),
        ],
        loc="upper right",
    )
    ax.set_title("Cặp ensemble xếp theo WAPE (trên cùng = tốt nhất)")
    return fig


# ----------------------------------------------------------------------------
# 05-08: Ngân sách & chuỗi thời gian
# ----------------------------------------------------------------------------
@figure("05_company_budget", "Ngân sách công ty: thực tế vs dự đoán theo tháng")
def fig_company_budget(R: Results):
    df = R.metric("company_budget_eval.csv")
    x = np.arange(len(df))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.4))
    b1 = ax.bar(x - w / 2, df["Tien_Thuc_Te"], w, color=C_ACTUAL, label="Thực tế")
    b2 = ax.bar(x + w / 2, df["Tien_Du_Doan"], w, color=C_PRED, label="Dự đoán")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt_vnd(bar.get_height()), ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}\nlệch {p:+.1f}% ({'+' if d >= 0 else ''}{fmt_vnd(d)})" for m, p, d in zip(df["month"], df["Ti_Le_Lech_%"], df["Chenh_Lech"])]
    )
    ax.yaxis.set_major_formatter(VND)
    ax.set_ylim(0, max(df["Tien_Thuc_Te"].max(), df["Tien_Du_Doan"].max()) * 1.2)
    ax.set_ylabel("Tổng tiền ăn")
    ax.legend(loc="upper center", ncol=2)
    ax.set_title("Tổng ngân sách tiền ăn toàn công ty — thực tế vs dự đoán\n(dưới mỗi tháng: dự đoán lệch bao nhiêu so với thực tế)")
    fig.tight_layout()
    return fig


@figure("06_budget_model_comparison", "So sánh mô hình theo sai số ngân sách (cùng split)")
def fig_budget_models(R: Results):
    df = R.metric("same_split_budget_model_comparison.csv").sort_values("WAPE_Budget_%").reset_index(drop=True)
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.5 * len(df) + 2.6), sharey=True)

    ax = axes[0]
    colors = [C_GOOD if i == 0 else "#2f6fdb" for i in range(len(df))]
    ax.set_xscale("log")
    # dùng chấm (không dùng cột) vì trên thang log độ dài cột không tỉ lệ với giá trị
    ax.hlines(y, df["WAPE_Budget_%"].min() * 0.6, df["WAPE_Budget_%"], color=colors, alpha=0.3, lw=2)
    ax.scatter(df["WAPE_Budget_%"], y, color=colors, s=70, zorder=3)
    for yi, v in zip(y, df["WAPE_Budget_%"]):
        ax.text(v * 1.15, yi, f"{v:,.1f}%", va="center", fontsize=8.5)
    ax.set_xlim(df["WAPE_Budget_%"].min() * 0.6, df["WAPE_Budget_%"].max() * 8)
    ax.set_title("WAPE ngân sách (%) — thang log vì ARIMA lệch quá lớn")
    ax.grid(axis="y", visible=False)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Model"])
    ax.invert_yaxis()

    ax = axes[1]
    ok = df["WAPE_Budget_%"] < 1000  # loại ARIMA khỏi biểu đồ bias để không làm bẹp các mô hình khác
    vals = np.where(ok, df["Mean_Bias_Budget"], np.nan)
    ax.barh(y, vals, color=[C_BAD if (v is not np.nan and v > 0) else "#2f6fdb" for v in vals])
    for yi, v, o in zip(y, df["Mean_Bias_Budget"], ok):
        ax.text(v if o else 0, yi, (" " + fmt_vnd(v)) if o else "  (ngoài thang — bị loại)", va="center", fontsize=8.5, ha="left" if (not o or v >= 0) else "right")
    ax.axvline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(VND)
    ax.set_title("Bias ngân sách (dự đoán − thực tế): đỏ = dư, xanh = thiếu")
    ax.grid(axis="y", visible=False)
    lim = np.nanmax(np.abs(vals)) * 1.4
    ax.set_xlim(-lim, lim)
    fig.suptitle("Sai số tổng ngân sách tháng của từng mô hình (cùng tập split)", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("07_budget_error_heatmap", "Heatmap sai số ngân sách (%) theo mô hình × tháng")
def fig_budget_heatmap(R: Results):
    df = R.metric("same_split_budget_model_comparison_by_month.csv")
    df["err_pct"] = df["Budget_Error"] / df["Actual_Budget"] * 100
    pivot = df.pivot(index="Model", columns="month", values="err_pct")
    order = pivot.abs().mean(axis=1).sort_values().index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(pivot) + 2))
    im = heatmap(ax, pivot.values, list(pivot.columns), list(pivot.index), "RdBu_r", -30, 30, lambda v: f"{v:+,.1f}%")
    fig.colorbar(im, ax=ax, label="Sai số (%) — màu cắt ở ±30%")
    ax.set_title("Sai số ngân sách = (dự đoán − thực tế) / thực tế\nđỏ = dự đoán dư, xanh = dự đoán thiếu")
    return fig


@figure("08_time_series_models", "Mô hình chuỗi thời gian (Prophet / MLP daily / ARIMA)")
def fig_time_series(R: Results):
    s = R.metric("time_series_models_summary.csv")
    m = R.metric("time_series_models_by_month.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    s = s.sort_values("WAPE_Cost_%").reset_index(drop=True)
    colors = ["#8957e5", "#a78bfa", "#c4b5fd"][: len(s)]
    ax = axes[0]
    bars = ax.bar(s["Model"], s["WAPE_Cost_%"], color=colors)
    for b_, v in zip(bars, s["WAPE_Cost_%"]):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("WAPE tiền — tổng công ty theo ngày")
    ax.set_ylim(0, s["WAPE_Cost_%"].max() * 1.15)

    ax = axes[1]
    x = np.arange(len(s))
    # chấm thay cho cột vì thang log làm độ dài cột không còn tỉ lệ với giá trị
    ax.scatter(x - 0.08, s["MAE_N44"], s=90, color="#2f6fdb", label="MAE N44", zorder=3)
    ax.scatter(x + 0.08, s["MAE_N70"], s=90, color="#f0883e", label="MAE N70", zorder=3)
    for xi, a_, b_ in zip(x, s["MAE_N44"], s["MAE_N70"]):
        ax.text(xi - 0.14, a_, f"{a_:,.0f}", ha="right", va="center", fontsize=8.5)
        ax.text(xi + 0.14, b_, f"{b_:,.1f}", ha="left", va="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(s["Model"])
    ax.set_xlim(-0.6, len(s) - 0.4)
    ax.set_yscale("log")
    ax.set_title("MAE số suất N44 / N70 (thang log)")
    ax.legend(loc="upper left")

    ax = axes[2]
    models = list(s["Model"])
    months = sorted(m["Test_Month"].unique())
    xm = np.arange(len(months))
    w = 0.8 / len(models)
    for i, mod in enumerate(models):
        sub = m[m["Model"] == mod].set_index("Test_Month").reindex(months)
        ax.bar(xm + (i - (len(models) - 1) / 2) * w, sub["WAPE_Cost_%"], w, label=mod, color=colors[i])
    ax.set_xticks(xm)
    ax.set_xticklabels(months)
    ax.set_ylim(0, m["WAPE_Cost_%"].max() * 1.3)
    ax.set_title("WAPE tiền theo tháng test (%)")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Mô hình chuỗi thời gian — đánh giá ở mức tổng công ty theo ngày", fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 09-11: Phân khúc, cohort, outlier
# ----------------------------------------------------------------------------
@figure("09_segment_eval", "Đánh giá theo phân khúc mức ăn")
def fig_segment(R: Results):
    df = R.metric("segment_eval.csv")
    order = [s for s in SEGMENT_ORDER if s in set(df["segment"])] or list(df["segment"])
    df = df.set_index("segment").loc[order].reset_index()
    labels = [wrap(s, 14) for s in df["segment"]]
    colors = [SEGMENT_COLORS.get(s, "#2f6fdb") for s in df["segment"]]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(x, df["So_Luong"], color=colors)
    for xi, v in zip(x, df["So_Luong"]):
        ax.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Số lượng (nhân viên-tháng)")

    ax = axes[1]
    wape = df["WAPE_%"].fillna(0)
    ax.bar(x, wape, color=colors)
    for xi, v, raw in zip(x, wape, df["WAPE_%"]):
        ax.text(xi, v, "n/a\n(thực tế = 0)" if pd.isna(raw) else f"{raw:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("WAPE (%)  — nhóm ăn ít bị phóng đại vì mẫu số nhỏ")
    ax.set_ylim(0, wape.max() * 1.2)

    ax = axes[2]
    ax.bar(x - 0.2, df["Tong_Thuc"], 0.4, color=C_ACTUAL, label="Tổng thực tế")
    ax.bar(x + 0.2, df["Tong_Doan"], 0.4, color=C_PRED, label="Tổng dự đoán")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(VND)
    ax.legend()
    ax.set_title("Tổng tiền thực tế vs dự đoán")
    fig.suptitle("Chất lượng dự đoán theo phân khúc mức ăn", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("10_cohort_heatmaps", "So sánh mô hình theo cohort (ALL/HIST3/ACTIVE/DORMANT/NEW)")
def fig_cohort(R: Results):
    df = R.metric("compare_same_cohort.csv")
    cohorts = [c for c in ["ALL", "HIST3", "ACTIVE", "DORMANT", "NEW"] if c in set(df["cohort"])]
    navg = df.groupby("cohort")["n_avg"].first()
    xl = [f"{c}\n(~{navg[c]:,.0f} NV)" for c in cohorts]
    w = df.pivot(index="model", columns="cohort", values="WAPE_%")[cohorts]
    b = df.pivot(index="model", columns="cohort", values="Bias_%")[cohorts]
    order = w["ALL"].sort_values().index if "ALL" in w else w.index
    w, b = w.loc[order], b.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.55 * len(w) + 2.6))
    im1 = heatmap(axes[0], w.values, xl, list(w.index), "YlOrRd", 15, min(60, np.nanmax(w.values)), lambda v: f"{v:.1f}")
    fig.colorbar(im1, ax=axes[0], label="WAPE (%) — màu cắt ở 60%")
    axes[0].set_title("WAPE (%) — càng nhạt càng tốt")
    lim = min(40, np.nanmax(np.abs(b.values)))
    im2 = heatmap(axes[1], b.values, xl, list(b.index), "RdBu_r", -lim, lim, lambda v: f"{v:+.1f}")
    fig.colorbar(im2, ax=axes[1], label=f"Bias (%) — màu cắt ở ±{lim:.0f}%")
    axes[1].set_title("Bias (%) — đỏ = dự đoán dư, xanh = thiếu")
    fig.suptitle("Hiệu năng từng mô hình trên các nhóm nhân viên (cùng cohort)", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("11_top_outliers", "Top các dự đoán sai lệch nhiều nhất")
def fig_outliers(R: Results):
    df = R.metric("top_outliers.csv").sort_values("abs_error_cost", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    labels = [f"NV {e} · {m} · {d}" for e, m, d in zip(df["employee_id"], df["month"], df["department"])]
    colors = [SEGMENT_COLORS.get(s, "#2f6fdb") for s in df["segment"]]
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(df) + 2))
    ax.barh(y, df["error_cost"], color=colors)
    for yi, a, p, e in zip(y, df["actual_cost"], df["pred_cost"], df["error_cost"]):
        ax.text(e, yi, f" thực {a / 1e3:,.0f}k → dự đoán {p / 1e3:,.0f}k" if e >= 0 else f"thực {a / 1e3:,.0f}k → dự đoán {p / 1e3:,.0f}k ", va="center", ha="left" if e >= 0 else "right", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(VND)
    lim = df["error_cost"].abs().max() * 1.55
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Sai số (dự đoán − thực tế)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=SEGMENT_COLORS.get(s, "#2f6fdb"), label=s) for s in SEGMENT_ORDER if s in set(df["segment"])], loc="lower left", title="Phân khúc thực tế")
    ax.set_title(f"Top {len(df)} nhân viên-tháng có sai số tiền lớn nhất")
    return fig


# ----------------------------------------------------------------------------
# 12-14: Dự đoán cấp nhân viên (parquet)
# ----------------------------------------------------------------------------
@figure("12_pred_vs_actual", "Thực tế vs dự đoán cấp nhân viên-tháng")
def fig_pred_vs_actual(R: Results):
    df = R.predictions()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    ax = axes[0]
    handles = []
    for seg in SEGMENT_ORDER:
        sub = df[df["segment"] == seg]
        if len(sub):
            ax.scatter(sub["actual_cost"], sub["pred_cost"], s=7, alpha=0.3, color=SEGMENT_COLORS[seg])
            handles.append(Line2D([], [], marker="o", ls="", color=SEGMENT_COLORS[seg], label=f"{seg} (n={len(sub):,})"))
    lim = max(df["actual_cost"].max(), df["pred_cost"].max()) * 1.03
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.xaxis.set_major_formatter(VND)
    ax.yaxis.set_major_formatter(VND)
    ax.set_xlabel("Thực tế")
    ax.set_ylabel("Dự đoán")
    ax.set_title("Tiền ăn / nhân viên / tháng")
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    for ax, a, p, title in [
        (axes[1], "N44", "pred_n44_s2", "Số suất N44 (ensemble S2)"),
        (axes[2], "N70", "pred_n70_s2", "Số suất N70 (ensemble S2)"),
    ]:
        jit = rng.uniform(-0.25, 0.25, len(df))
        ax.scatter(df[a] + jit, df[p], s=7, alpha=0.25, color=C_PRED)
        lim = max(df[a].max(), df[p].max()) * 1.03
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("Thực tế (có nhiễu nhẹ trục x để thấy mật độ)")
        ax.set_ylabel("Dự đoán")
        ax.set_title(title)
    fig.suptitle("Độ khớp giữa dự đoán và thực tế (đường nét đứt = dự đoán hoàn hảo)", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("13_error_distribution", "Phân phối sai số dự đoán")
def fig_error_dist(R: Results):
    df = R.predictions()
    months = sorted(df["month"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    bins = np.linspace(-600_000, 600_000, 61)
    for m, c in zip(months, ["#2f6fdb", "#f0883e", "#2da44e", "#8957e5"]):
        ax.hist(df.loc[df["month"] == m, "error_cost"].clip(-600_000, 600_000), bins=bins, histtype="step", lw=1.8, color=c, label=m)
    ax.axvline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(VND)
    ax.set_title("Histogram sai số (dự đoán − thực tế), cắt ở ±600k")
    ax.set_xlabel("Sai số")
    ax.set_ylabel("Số nhân viên-tháng")
    ax.legend()

    ax = axes[1]
    segs = [s for s in SEGMENT_ORDER if s in set(df["segment"])]
    data = [df.loc[df["segment"] == s, "error_cost"] for s in segs]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, medianprops=dict(color="black"))
    ax.set_xticks(range(1, len(segs) + 1))
    ax.set_xticklabels([wrap(s, 14) for s in segs])
    for patch, s in zip(bp["boxes"], segs):
        patch.set_facecolor(SEGMENT_COLORS[s])
        patch.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(VND)
    ax.set_title("Sai số theo phân khúc (không vẽ điểm ngoại lai)")

    ax = axes[2]
    g = df.groupby("month").agg(MAE=("abs_error_cost", "mean"), Bias=("error_cost", "mean"))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["MAE"], 0.4, color=C_PRED, label="MAE")
    ax.bar(x + 0.2, g["Bias"], 0.4, color=C_BAD, label="Bias (trung bình có dấu)")
    for xi, a, b in zip(x, g["MAE"], g["Bias"]):
        ax.text(xi - 0.2, a, f"{a / 1e3:,.0f}k", ha="center", va="bottom", fontsize=9)
        ax.text(xi + 0.2, b, f"{b / 1e3:+,.0f}k", ha="center", va="bottom" if b >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index)
    ax.yaxis.set_major_formatter(VND)
    ax.legend()
    ax.set_title("MAE & bias trung bình theo tháng (mỗi nhân viên)")
    fig.suptitle("Phân phối sai số tiền ăn ở cấp nhân viên-tháng", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("14_department_error", "Sai số theo phòng ban")
def fig_department(R: Results):
    df = R.predictions()
    g = (
        df.groupby("department")
        .agg(actual=("actual_cost", "sum"), pred=("pred_cost", "sum"), n=("employee_id", "size"), mae=("abs_error_cost", "mean"))
        .assign(err=lambda d: d["pred"] - d["actual"])
    )
    g["err_pct"] = g["err"] / g["actual"].replace(0, np.nan) * 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    sc = ax.scatter(g["actual"], g["pred"], s=20 + g["n"] / g["n"].max() * 180, c=g["err_pct"].clip(-40, 40), cmap="RdBu_r", vmin=-40, vmax=40, alpha=0.85, edgecolor="k", linewidth=0.4)
    lim = max(g["actual"].max(), g["pred"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    for k, (name, r) in enumerate(g.reindex(g["err"].abs().sort_values(ascending=False).index).head(4).iterrows()):
        ax.annotate(name, (r["actual"], r["pred"]), textcoords="offset points", xytext=(8, 8 if k % 2 == 0 else -14), fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.xaxis.set_major_formatter(VND)
    ax.yaxis.set_major_formatter(VND)
    ax.set_xlabel("Tổng thực tế (3 tháng test)")
    ax.set_ylabel("Tổng dự đoán (3 tháng test)")
    ax.set_title("Mỗi chấm = 1 phòng ban (kích thước ∝ số NV-tháng, màu = % lệch)")
    fig.colorbar(sc, ax=ax, label="% lệch (cắt ±40%)")

    ax = axes[1]
    top = pd.concat([g.nlargest(8, "err"), g.nsmallest(8, "err")]).drop_duplicates().sort_values("err", ascending=False)
    y = np.arange(len(top))
    ax.barh(y, top["err"], color=[C_BAD if v > 0 else "#2f6fdb" for v in top["err"]])
    for yi, v, pct in zip(y, top["err"], top["err_pct"]):
        ax.text(v, yi, f" {fmt_vnd(v)} ({pct:+.0f}%)" if v >= 0 else f"{fmt_vnd(v)} ({pct:+.0f}%) ", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(VND)
    lim = top["err"].abs().max() * 1.45
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set_title("8 phòng ban dự đoán dư nhiều nhất & 8 phòng ban thiếu nhiều nhất")
    fig.suptitle("Sai số tổng tiền ăn theo phòng ban (tổng 3 tháng test)", fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 15-17: Dự báo tháng tới
# ----------------------------------------------------------------------------
@figure("15_forecast_departments", "Dự báo tháng tới theo phòng ban")
def fig_forecast_dept(R: Results):
    d, month = R.forecast("department")
    d = d.sort_values("tong_du_bao", ascending=False)
    top = d.head(20)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    ax = axes[0]
    y = np.arange(len(top))
    ax.barh(y, top["tong_du_bao"], color=C_PRED)
    for yi, v, n in zip(y, top["tong_du_bao"], top["so_nhan_vien"]):
        ax.text(v, yi, f" {fmt_vnd(v)} · {n} NV", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(top["department"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(VND)
    ax.set_xlim(0, top["tong_du_bao"].max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Top 20 phòng ban theo tổng dự báo (tổng {len(d)} phòng ban)")

    ax = axes[1]
    per_head = d["tong_du_bao"].sum() / d["so_nhan_vien"].sum()
    dd = d.assign(per_head=d["tong_du_bao"] / d["so_nhan_vien"])
    dd = dd[dd["so_nhan_vien"] >= 5]  # bỏ phòng ban quá nhỏ để tránh nhiễu
    sel = pd.concat([dd.nlargest(8, "per_head"), dd.nsmallest(8, "per_head")]).drop_duplicates("department").sort_values("per_head", ascending=False)
    y = np.arange(len(sel))
    ax.barh(y, sel["per_head"], color=[C_BAD if v > per_head else "#2f6fdb" for v in sel["per_head"]])
    for yi, v, n in zip(y, sel["per_head"], sel["so_nhan_vien"]):
        ax.text(v, yi, f" {v / 1e3:,.0f}k · {n} NV", va="center", fontsize=8.5)
    ax.axvline(per_head, color="black", ls="--", lw=1, label=f"Trung bình công ty ≈ {per_head / 1e3:,.0f}k đ/người")
    ax.set_yticks(y)
    ax.set_yticklabels(sel["department"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(VND)
    ax.set_xlim(0, sel["per_head"].max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Dự báo bình quân đầu người: 8 cao nhất & 8 thấp nhất (phòng ban ≥ 5 NV)")
    fig.suptitle(f"Dự báo ngân sách tháng {month} — tổng {fmt_vnd(d['tong_du_bao'].sum())} đ", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("16_forecast_employees", "Dự báo tháng tới theo nhân viên")
def fig_forecast_emp(R: Results):
    e, month = R.forecast("employee")
    try:
        x_, _ = R.forecast("excluded_not_in_hr")
    except FileNotFoundError:
        x_ = pd.DataFrame(columns=e.columns)
    sources = list(dict.fromkeys(e["source"]))
    palette = {"CatBoost": "#2f6fdb", "Group-Mean (cold-start)": "#f0883e"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    bins = np.linspace(0, e["pred_cost"].max() * 1.02, 40)
    ax.hist([e.loc[e["source"] == s, "pred_cost"] for s in sources], bins=bins, stacked=True, color=[palette.get(s, "#9aa5b1") for s in sources], label=sources)
    ax.xaxis.set_major_formatter(VND)
    ax.set_xlabel("Dự báo tiền ăn / nhân viên")
    ax.set_ylabel("Số nhân viên")
    ax.legend()
    ax.set_title(f"Phân phối dự báo tháng {month} ({len(e):,} nhân viên)")

    ax = axes[1]
    rng = np.random.default_rng(1)
    for s in sources:
        sub = e[e["source"] == s]
        ax.scatter(sub["pred_N44"] + rng.uniform(-0.2, 0.2, len(sub)), sub["pred_N70"] + rng.uniform(-0.2, 0.2, len(sub)), s=10, alpha=0.4, color=palette.get(s, "#9aa5b1"), label=s)
    ax.set_xlabel("Dự báo số suất N44")
    ax.set_ylabel("Dự báo số suất N70")
    ax.set_title("Cơ cấu suất ăn dự báo (có nhiễu nhẹ để thấy mật độ)")
    ax.legend(markerscale=2.5)

    ax = axes[2]
    rows = []
    for s in sources:
        sub = e[e["source"] == s]
        rows.append((f"Trong HR\n{s}", sub["pred_cost"].sum(), len(sub), palette.get(s, "#9aa5b1")))
    if len(x_):
        rows.append(("Ngoài HR\n(đã loại khỏi dự báo)", x_["pred_cost"].sum(), len(x_), "#9aa5b1"))
    xs = np.arange(len(rows))
    bars = ax.bar(xs, [r[1] for r in rows], color=[r[3] for r in rows])
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, r[1], f"{fmt_vnd(r[1])}\n{r[2]:,} NV", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.yaxis.set_major_formatter(VND)
    ax.set_ylim(0, max(r[1] for r in rows) * 1.2)
    ax.set_title("Tổng dự báo theo nguồn")
    fig.suptitle(f"Dự báo cấp nhân viên — tháng {month}", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("17_budget_trend", "Xu hướng ngân sách: thực tế, dự đoán và dự báo tháng tới")
def fig_budget_trend(R: Results):
    b = R.metric("company_budget_eval.csv")
    e, fmonth = R.forecast("employee")
    months = list(b["month"]) + [fmonth]
    x = np.arange(len(months))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.4))
    b1 = ax.bar(x[:-1] - w / 2, b["Tien_Thuc_Te"], w, color=C_ACTUAL, label="Thực tế")
    b2 = ax.bar(x[:-1] + w / 2, b["Tien_Du_Doan"], w, color=C_PRED, label="Dự đoán (backtest)")
    bottom = 0
    palette = {"CatBoost": "#2da44e", "Group-Mean (cold-start)": "#f0883e"}
    for s in dict.fromkeys(e["source"]):
        v = e.loc[e["source"] == s, "pred_cost"].sum()
        ax.bar(x[-1], v, w * 1.4, bottom=bottom, color=palette.get(s, "#9aa5b1"), hatch="//", edgecolor="white", label=f"Dự báo {fmonth} — {s}")
        bottom += v
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt_vnd(bar.get_height()), ha="center", va="bottom", fontsize=9)
    ax.text(x[-1], bottom, fmt_vnd(bottom), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.axvline(x[-1] - 0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.yaxis.set_major_formatter(VND)
    ax.set_ylim(0, max(b["Tien_Thuc_Te"].max(), b["Tien_Du_Doan"].max(), bottom) * 1.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    ax.set_title("Tổng tiền ăn công ty: 3 tháng backtest và dự báo tháng kế tiếp")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 18: Log chạy pipeline
# ----------------------------------------------------------------------------
@figure("18_pipeline_runs", "Lịch sử chạy pipeline (từ file log)")
def fig_pipeline(R: Results):
    df = parse_logs(R.logs_dir)
    if df.empty:
        raise ValueError("Không parse được bảng tóm tắt 'Bước N ...' từ file log nào")
    df = df.sort_values(["ts", "step"])
    runs = list(dict.fromkeys(df["run"]))

    def label(run):
        sub = df[df["run"] == run]
        ts = sub["ts"].iloc[0]
        extra = " [skip-dl]" if sub["status"].str.contains("skip-dl").any() else ""
        extra = " [dry-run]" if (sub["status"] == "DRY-RUN").any() else extra
        extra = " [LỖI]" if sub["status"].str.startswith("LỖI").any() else extra
        return ts.strftime("%d/%m %H:%M:%S") + extra

    step_names = df.drop_duplicates("step").set_index("step")["name"].to_dict()
    steps = sorted(step_names)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 1, figsize=(13, 0.5 * len(runs) * 2 + 4.5), gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    ok_runs = [r for r in runs if (df[(df["run"] == r)]["seconds"] > 0).any()]
    y = np.arange(len(ok_runs))
    left = np.zeros(len(ok_runs))
    for s in steps:
        vals = np.array([df[(df["run"] == r) & (df["step"] == s)]["seconds"].sum() for r in ok_runs])
        ax.barh(y, vals, left=left, color=cmap((s - 1) % 10), label=f"{s}. {step_names[s]}")
        left += vals
    for yi, t in zip(y, left):
        ax.text(t, yi, f" {t:,.0f}s ({t / 60:.1f} phút)", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([label(r) for r in ok_runs])
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.2)
    ax.set_xlabel("Thời gian chạy (giây)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    ax.set_title("Thời gian từng bước của mỗi lần chạy (bước lỗi vẫn tính thời gian đã tốn)")

    ax = axes[1]
    status_color = {"OK": C_GOOD, "DRY-RUN": "#f5d36b"}
    for yi, r in enumerate(runs):
        for s in steps:
            row = df[(df["run"] == r) & (df["step"] == s)]
            if row.empty:
                color, txt = "#ffffff", ""
            else:
                st = row["status"].iloc[0]
                color = status_color.get(st, "#cbd5e1" if st.startswith("BỎ QUA") else C_BAD)
                txt = "OK" if st == "OK" else ("dry" if st == "DRY-RUN" else "bỏ qua" if st.startswith("BỎ QUA") else "lỗi")
            ax.add_patch(plt.Rectangle((s - 1, yi), 0.94, 0.9, color=color))
            ax.text(s - 1 + 0.47, yi + 0.45, txt, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, len(runs))
    ax.set_xticks([s - 1 + 0.47 for s in steps])
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("Bước")
    ax.set_yticks([i + 0.45 for i in range(len(runs))])
    ax.set_yticklabels([label(r) for r in runs])
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.legend(
        handles=[Patch(color=C_GOOD, label="OK"), Patch(color="#cbd5e1", label="Bỏ qua"), Patch(color="#f5d36b", label="Dry-run"), Patch(color=C_BAD, label="Lỗi")],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    ax.set_title("Trạng thái từng bước (ô trắng = log không có bảng tóm tắt cho bước đó)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Chạy tất cả + thư viện ảnh
# ----------------------------------------------------------------------------
def write_gallery(out_dir: Path, done: list[tuple[str, str]]):
    items = "\n".join(
        f'<section><h2>{html.escape(t)}</h2><a href="{n}.png"><img loading="lazy" src="{n}.png" alt="{html.escape(t)}"></a></section>'
        for n, t in done
    )
    (out_dir / "index.html").write_text(
        f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<title>Kết quả pipeline dự báo tiền ăn</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;background:#f8fafc;color:#1e293b}}
section{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.25rem;margin:1.25rem 0}}
h2{{font-size:1.05rem;margin:.2rem 0 .8rem}}img{{max-width:100%;height:auto}}</style></head>
<body><h1>Kết quả pipeline dự báo tiền ăn</h1>{items}</body></html>""",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results", help="thư mục chứa metrics/ forecast/ logs/ (mặc định: ./results)")
    ap.add_argument("--out-dir", default=None, help="nơi lưu PNG (mặc định: <results-dir>/figures)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--only", nargs="*", help="chỉ vẽ các biểu đồ có tên bắt đầu bằng, vd: --only 01 05 12")
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ matplotlib")
    ap.add_argument("--debug", action="store_true", help="in traceback đầy đủ khi một biểu đồ bị lỗi")
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not (root / "metrics").exists() and (Path.cwd() / "metrics").exists():
        root = Path.cwd()
    if not root.exists():
        sys.exit(f"Không tìm thấy thư mục kết quả: {root}")
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    R = Results(root)

    done, failed = [], []
    for name, title, fn in FIGURES:
        if args.only and not any(name.startswith(p) for p in args.only):
            continue
        try:
            fig = fn(R)
            fig.savefig(out_dir / f"{name}.png", dpi=args.dpi, facecolor="white")
            if not args.show:
                plt.close(fig)
            done.append((name, title))
            print(f"[OK]   {name}.png  — {title}")
        except Exception as exc:  # noqa: BLE001 — muốn các biểu đồ còn lại vẫn chạy tiếp
            failed.append((name, exc))
            print(f"[LỖI]  {name}: {type(exc).__name__}: {exc}")
            if args.debug:
                traceback.print_exc()
            plt.close("all")

    if done:
        write_gallery(out_dir, done)
    print(f"\nHoàn tất: {len(done)} biểu đồ, {len(failed)} lỗi. Thư mục: {out_dir.resolve()}")
    if done:
        print(f"Mở {out_dir / 'index.html'} để xem tất cả cùng lúc.")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()