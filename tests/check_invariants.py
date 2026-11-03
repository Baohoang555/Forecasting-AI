"""
Kiểm tra các BẤT BIẾN của pipeline bằng thực nghiệm (không dựa vào việc đọc code).

Chạy từ thư mục gốc dự án, trên dữ liệu thật:
    python tests/check_invariants.py --monthly data/processed/monthly_dataset.parquet

In PASS / FAIL / SKIP kèm con số cụ thể. Thoát mã 1 nếu có FAIL.

T1  Không rò rỉ target: MỌI cột của bảng đặc trưng (trừ các cột thô của chính tháng đó)
    phải không đổi khi ta phá hỏng giá trị thật của tháng cuối. Test này tự bắt được cả
    những cột rò rỉ mà không ai biết tên (như `expected_amount` trước đây).
T2  Mỗi nhân viên có các tháng liên tiếp (không nhảy tháng).
T3  Parity train/inference: dòng đặc trưng dựng cho tháng cần dự báo == dòng dùng khi train.
T4  (employee_id, month) là khoá duy nhất.
T5  Chỉ chứa tháng hoàn chỉnh (nếu có cột is_complete_month).
T6  Dùng schema đặc trưng tường minh và không chứa cột cấm (nếu có src/feature_schema.py).
T7  Lag theo THÁNG LỊCH: nếu một nhân viên thiếu 1 tháng giữa chừng thì tháng kế sau
    không được mượn giá trị của tháng xa hơn làm lag_1.
"""

import argparse
import contextlib
import importlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = []

# Cột thô của CHÍNH tháng đó (hiển nhiên phụ thuộc target) — không dùng để kiểm tra rò rỉ.
RAW_SAME_MONTH = {"N44", "N70", "total_meals", "total_amount", "days_with_meal", "calendar_days"}
ID_COLS = {"employee_id", "month", "department", "is_complete_month"}


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def skip(name, why):
    print(f"[SKIP] {name} — {why}")


def build(monthly_df, tmp):
    fe = importlib.import_module("feature_engineering")
    mp, fp = Path(tmp) / "m.parquet", Path(tmp) / "f.parquet"
    monthly_df.to_parquet(mp, index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        feats = fe.build_features(str(mp), str(fp))
    feats = feats.copy()
    feats["month"] = pd.PeriodIndex(feats["month"], freq="M")
    return feats


def candidate_cols(feats):
    """Mọi cột số của bảng đặc trưng, trừ định danh và cột thô của chính tháng đó."""
    return [c for c in feats.select_dtypes(include=[np.number]).columns
            if c not in RAW_SAME_MONTH and c not in ID_COLS]


def test_t1(monthly, tmp):
    base = build(monthly, tmp)
    last = monthly["month"].max()
    pert = monthly.copy()
    rng = np.random.default_rng(0)
    m = pert["month"] == last
    for col in ["N44", "N70", "total_meals", "days_with_meal", "total_amount"]:
        pert[col] = pert[col].astype(float)
        pert.loc[m, col] = rng.integers(0, 40, int(m.sum())).astype(float)
    # các cột dẫn xuất từ target cũng phải bị phá để phát hiện rò rỉ gián tiếp
    for col in ["expected_amount", "amount_diff", "au_share"]:
        if col in pert.columns:
            pert[col] = pert[col].astype(float)
            pert.loc[m, col] = rng.random(int(m.sum())) * 1e6
    after = build(pert, tmp)
    cols = candidate_cols(base)
    a = base[base["month"] == last].set_index("employee_id")[cols].sort_index()
    b = after[after["month"] == last].set_index("employee_id")[cols].sort_index()
    common = a.index.intersection(b.index)
    d = (a.loc[common].astype(float) - b.loc[common].astype(float)).abs().max()
    bad = d[d > 1e-9]
    report("T1 mọi cột đặc trưng của tháng M độc lập với target tháng M", bad.empty,
           f"đã kiểm tra {len(cols)} cột, {len(common)} nhân viên"
           + ("" if bad.empty else f"; CỘT RÒ RỈ: {sorted(bad.index)}"))
    return base


def test_t2(monthly):
    gaps = 0
    for _, g in monthly.groupby("employee_id"):
        o = np.sort(g["month"].map(lambda p: p.ordinal).values)
        if len(o) > 1 and (np.diff(o) != 1).any():
            gaps += 1
    report("T2 mỗi nhân viên có tháng liên tiếp", gaps == 0, f"{gaps} nhân viên có khoảng trống")


def test_t3(monthly, feats, k=3):
    try:
        pn = importlib.import_module("predict_next_month")
        fn = getattr(pn, "build_forecast_feature_row")
    except Exception as e:  # noqa: BLE001
        skip("T3 parity train/inference", f"không import được build_forecast_feature_row ({type(e).__name__})")
        return
    strict = False
    try:
        fs = importlib.import_module("feature_schema")
        cols = [c for c in fs.FEATURE_COLS if c in feats.columns]
        strict = True
    except ImportError:
        cols = candidate_cols(feats)
    worst, worst_col, covered, total = 0.0, None, 0, 0
    for M in sorted(feats["month"].unique())[-k:]:
        fc = fn(monthly[monthly["month"] < M], M).set_index("employee_id")
        tr = feats[feats["month"] == M].set_index("employee_id")
        missing = [c for c in cols if c not in fc.columns]
        if missing and strict:
            report("T3 dòng forecast có đủ đặc trưng như dòng train", False, f"thiếu: {missing}")
            return
        if missing:
            print(f"[WARN] T3 các cột có trong bảng train nhưng không có trong dòng forecast "
                  f"(chỉ ổn nếu mô hình đã loại chúng): {missing}")
            cols = [c for c in cols if c in fc.columns]
        common = fc.index.intersection(tr.index)
        total += len(tr)
        covered += len(common)
        d = (fc.loc[common, cols].astype(float) - tr.loc[common, cols].astype(float)).abs().max()
        if d.max() > worst:
            worst, worst_col = float(d.max()), d.idxmax()
    ok = worst < 1e-6 and covered > 0
    report("T3 parity train/inference", ok,
           f"lệch lớn nhất {worst:.3g}" + ("" if ok else f" ở cột {worst_col}")
           + f"; phủ {covered}/{total} dòng, {len(cols)} cột")


def test_t4(monthly):
    dup = int(monthly.duplicated(["employee_id", "month"]).sum())
    report("T4 (employee_id, month) là khoá duy nhất", dup == 0, f"{dup} dòng trùng")


def test_t5(monthly):
    if "is_complete_month" not in monthly.columns:
        return skip("T5 chỉ chứa tháng hoàn chỉnh", "không có cột is_complete_month")
    n = int((~monthly["is_complete_month"].astype(bool)).sum())
    report("T5 chỉ chứa tháng hoàn chỉnh", n == 0, f"{n} dòng thuộc tháng chưa hoàn chỉnh")


def test_t6(feats):
    try:
        fs = importlib.import_module("feature_schema")
    except ImportError:
        return skip("T6 schema tường minh", "chưa có src/feature_schema.py")
    missing = [c for c in fs.FEATURE_COLS if c not in feats.columns]
    forbidden = sorted(set(fs.FEATURE_COLS) & set(fs.FORBIDDEN_COLS))
    report("T6 schema đặc trưng tường minh, không chứa cột cấm", not missing and not forbidden,
           f"{len(fs.FEATURE_COLS)} cột; thiếu={missing}; cấm={forbidden}")


def test_t7(monthly, tmp):
    """Cố ý xoá 1 tháng giữa chừng của vài nhân viên rồi kiểm tra lag_1."""
    months = sorted(monthly["month"].unique())
    if len(months) < 8:
        return skip("T7 lag theo tháng lịch", "dữ liệu quá ngắn")
    hole = months[len(months) // 2]
    ids = monthly["employee_id"].drop_duplicates().iloc[:30]
    gapped = monthly[~(monthly["employee_id"].isin(ids) & (monthly["month"] == hole))].copy()
    f = build(gapped, tmp)
    nxt = hole + 1
    bad = f[f["employee_id"].isin(ids) & (f["month"] == nxt)]
    leaked = int(bad["N44_lag_1"].notna().sum()) if "N44_lag_1" in bad.columns else 0
    report("T7 lag theo tháng lịch (tháng thiếu không bị 'lấp' bằng tháng xa hơn)", leaked == 0,
           f"{leaked} dòng tháng {nxt} vẫn mang lag_1 lấy từ tháng không liền kề")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly", default="data/processed/monthly_dataset.parquet")
    args = ap.parse_args()
    monthly = pd.read_parquet(args.monthly)
    monthly["month"] = pd.PeriodIndex(monthly["month"], freq="M")
    with tempfile.TemporaryDirectory() as tmp:
        test_t4(monthly)
        test_t5(monthly)
        test_t2(monthly)
        feats = test_t1(monthly, tmp)
        test_t3(monthly, feats)
        test_t6(feats)
        test_t7(monthly, tmp)
    print(f"\nTổng: {sum(RESULTS)}/{len(RESULTS)} test PASS")
    sys.exit(0 if all(RESULTS) else 1)