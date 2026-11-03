"""
So sánh baseline và mô hình ML/DL trên CÙNG MỘT TẬP nhân viên, tách theo nhóm.

Vì sao cần: baselines.py chỉ đo các dòng đủ 3 tháng lịch sử, còn ml_models.py/dl_models.py
đo toàn bộ nhân viên của tháng test (người mới dùng fallback). Hai bên đo trên tập khác nhau
nên bảng so sánh gộp chưa công bằng. Ngoài ra, lưới tháng điền 0 đến cuối dữ liệu cho cả
người đã nghỉ việc => nhiều dòng "0 -> 0" dễ đoán, làm baseline đơn giản trông tốt hơn.

Chạy (sau khi đã chạy ml_models.py và dl_models.py):
    python src/compare_same_cohort.py

Nhóm nhân viên (mỗi tháng test):
  ALL      mọi nhân viên có dòng trong tháng đó (cohort end-to-end)
  HIST3    có đủ 3 tháng lịch sử liền trước (mô hình chính được dùng)
  ACTIVE   HIST3 và có ít nhất 1 bữa trong 3 tháng đó (loại người đã nghỉ/không dùng căn-tin)
  DORMANT  HIST3 nhưng 3 tháng trước đều 0 bữa (nghỉ việc hoặc ngưng ăn; thường dự đoán 0)
  NEW      chưa đủ 3 tháng lịch sử (cold-start, dùng trung bình phòng ban)

Mọi mô hình/baseline đều có dự đoán cho mọi dòng: thiếu lịch sử -> trung bình phòng ban.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from prediction_utils import department_mean_fallback, latest_complete_months  # noqa: E402

P44, P70 = 44_000, 70_000


def metrics(n44, n70, p44, p70):
    p44, p70 = np.clip(p44, 0, None), np.clip(p70, 0, None)
    t = n44 * P44 + n70 * P70
    p = p44 * P44 + p70 * P70
    err = np.abs(t - p)
    return {
        "n": len(t),
        "MAE": err.mean(),
        "WAPE_%": err.sum() / t.sum() * 100 if t.sum() > 0 else np.nan,
        "Bias_%": (p.sum() / t.sum() - 1) * 100 if t.sum() > 0 else np.nan,
    }


def calendar_lags(monthly):
    m = monthly.sort_values(["employee_id", "month"]).reset_index(drop=True)
    for lag in (1, 2, 3):
        prev_m = m.groupby("employee_id")["month"].shift(lag)
        ok = (m["month"].map(lambda p: p.ordinal) - prev_m.map(lambda p: p.ordinal if pd.notna(p) else np.nan)) == lag
        for c in ("N44", "N70"):
            m[f"{c}_lag{lag}"] = m.groupby("employee_id")[c].shift(lag).where(ok)
    return m


def main(monthly_path, ml_path, dl_path, out_path):
    monthly = pd.read_parquet(monthly_path)
    monthly["month"] = pd.PeriodIndex(monthly["month"], freq="M")
    ml = pd.read_parquet(ml_path)
    ml["month"] = pd.PeriodIndex(ml["month"], freq="M")
    dl = pd.read_parquet(dl_path) if os.path.exists(dl_path) else None
    if dl is not None:
        dl["month"] = pd.PeriodIndex(dl["month"], freq="M")
        ml = ml.merge(dl, on=["employee_id", "month"], how="left")

    lagged = calendar_lags(monthly)
    rows = []
    rowerr = []  # sai số tuyệt đối từng (nhân viên, tháng, mô hình) để kiểm định cặp
    for tm in sorted(ml["month"].unique()):
        t = ml[ml["month"] == tm].merge(
            lagged[lagged["month"] == tm][["employee_id", "department"]
                                          + [f"{c}_lag{l}" for c in ("N44", "N70") for l in (1, 2, 3)]],
            on="employee_id", how="left")
        hist = monthly[monthly["month"] < tm]
        fb = department_mean_fallback(hist, t[["employee_id", "department"]]).set_index("employee_id")

        has3 = t[[f"{c}_lag{l}" for c in ("N44", "N70") for l in (1, 2, 3)]].notna().all(axis=1)
        prior = t[[f"{c}_lag{l}" for c in ("N44", "N70") for l in (1, 2, 3)]].fillna(0).sum(axis=1)
        t["cohort_HIST3"], t["cohort_NEW"] = has3, ~has3
        t["cohort_ACTIVE"], t["cohort_DORMANT"] = has3 & (prior > 0), has3 & (prior == 0)
        t["cohort_ALL"] = True

        preds = {
            "Naive Lag-1": (t["N44_lag1"], t["N70_lag1"]),
            "MA 3M": (t[[f"N44_lag{l}" for l in (1, 2, 3)]].mean(axis=1), t[[f"N70_lag{l}" for l in (1, 2, 3)]].mean(axis=1)),
            "Dept mean": (fb.loc[t["employee_id"], "N44"].values, fb.loc[t["employee_id"], "N70"].values),
        }
        for name, (a, b) in {"M1 RF": ("m1_p44", "m1_p70"), "M2 LGB": ("m2_p44", "m2_p70"),
                              "M3 CatBoost": ("m3_p44", "m3_p70"), "M4 GRU": ("m4_p44", "m4_p70"),
                              "M5 MLP": ("m5_p44", "m5_p70")}.items():
            if a in t.columns:
                preds[name] = (t[a], t[b])
        month_err = t[["employee_id", "cohort_ALL", "cohort_HIST3", "cohort_ACTIVE"]].copy()
        month_err["month"] = str(tm)
        t_cost = t["N44"].values * P44 + t["N70"].values * P70
        for name, (p44, p70) in preds.items():
            p44 = pd.Series(np.asarray(p44, dtype=float), index=t.index)
            p70 = pd.Series(np.asarray(p70, dtype=float), index=t.index)
            miss = p44.isna() | p70.isna()  # thiếu lịch sử -> trung bình phòng ban
            p44[miss] = fb.loc[t.loc[miss, "employee_id"], "N44"].values
            p70[miss] = fb.loc[t.loc[miss, "employee_id"], "N70"].values
            month_err[name] = np.abs(
                t_cost - (np.clip(p44.values, 0, None) * P44 + np.clip(p70.values, 0, None) * P70))
            for coh in ("ALL", "HIST3", "ACTIVE", "DORMANT", "NEW"):
                m = t[f"cohort_{coh}"]
                if m.sum() == 0:
                    continue
                r = metrics(t.loc[m, "N44"].values, t.loc[m, "N70"].values, p44[m].values, p70[m].values)
                rows.append({"month": str(tm), "cohort": coh, "model": name, **r})

        rowerr.append(month_err)

    df = pd.DataFrame(rows)
    # trung bình 3 tháng, trọng số bằng số dòng
    agg = (df.assign(w=df["n"]).groupby(["cohort", "model"])
             .apply(lambda g: pd.Series({
                 "n_avg": g["n"].mean(),
                 "MAE": np.average(g["MAE"], weights=g["w"]),
                 "WAPE_%": g["WAPE_%"].mean(),
                 "Bias_%": g["Bias_%"].mean()}), include_groups=False).reset_index())
    for coh in ("ALL", "HIST3", "ACTIVE", "DORMANT", "NEW"):
        sub = agg[agg["cohort"] == coh].sort_values("MAE")
        if sub.empty:
            continue
        print(f"\n=== Nhóm {coh}  (~{sub['n_avg'].iloc[0]:.0f} nhân viên/tháng) ===")
        print(sub.drop(columns=["cohort", "n_avg"]).to_string(
            index=False, formatters={"MAE": "{:,.0f}".format, "WAPE_%": "{:.2f}".format, "Bias_%": "{:+.2f}".format}))
    # ---- sai lệch ngân sách và MAE THEO TỪNG THÁNG (nhóm ALL) ----
    by_month = df[df["cohort"] == "ALL"]
    keep = [m for m in ("MA 3M", "M2 LGB", "M3 CatBoost") if m in by_month["model"].unique()]
    print("\n=== Theo từng tháng, nhóm ALL: Bias_% (dự báo tổng so với thực tế) và MAE ===")
    pv = by_month[by_month["model"].isin(keep)].pivot(index="month", columns="model", values=["Bias_%", "MAE"])
    print(pv.round(1).to_string())

    # ---- kiểm định cặp: mô hình có thật sự tốt hơn MA 3M? (bootstrap theo nhân viên) ----
    allerr = pd.concat(rowerr, ignore_index=True)
    rng = np.random.default_rng(0)
    print("\n=== Mô hình so với MA 3M (gain > 0 nghĩa là mô hình tốt hơn), CI 95% bootstrap theo nhân viên ===")
    for model in [m for m in ("M3 CatBoost", "M2 LGB") if m in allerr.columns]:
        for coh in ("ALL", "HIST3", "ACTIVE"):
            sub = allerr[allerr[f"cohort_{coh}"]]
            d = (sub["MA 3M"] - sub[model]).groupby(sub["employee_id"]).agg(["sum", "count"])
            sums, cnts = d["sum"].values, d["count"].values
            n = len(sums)
            counts = rng.multinomial(n, np.full(n, 1.0 / n), size=2000)
            est = (counts @ sums) / (counts @ cnts)
            lo, hi = np.percentile(est, [2.5, 97.5])
            gain = sums.sum() / cnts.sum()
            base = sub["MA 3M"].mean()
            verdict = ("TỐT HƠN có ý nghĩa" if lo > 0 else "KÉM HƠN có ý nghĩa" if hi < 0
                       else "KHÔNG phân biệt được với nhiễu")
            print(f"{model:12s} {coh:7s} gain = {gain:9,.0f} đ ({gain / base * 100:+.2f}% MAE MA 3M)  CI95 [{lo:,.0f}; {hi:,.0f}]  -> {verdict}")
    print("Lưu ý: bootstrap chỉ phản ánh biến thiên giữa các nhân viên, KHÔNG phản ánh biến thiên giữa các tháng (chỉ có 3 tháng test).")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    by_month.to_csv(out_path.replace(".csv", "_by_month.csv"), index=False)
    agg.to_csv(out_path, index=False)
    print(f"\nĐã lưu: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly", default="data/processed/monthly_dataset.parquet")
    ap.add_argument("--ml", default="data/processed/ml_test_predictions.parquet")
    ap.add_argument("--dl", default="data/processed/dl_test_predictions.parquet")
    ap.add_argument("--out", default="results/metrics/compare_same_cohort.csv")
    a = ap.parse_args()
    main(a.monthly, a.ml, a.dl, a.out)