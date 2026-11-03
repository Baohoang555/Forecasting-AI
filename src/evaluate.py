import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PRICE_44 = 44000
PRICE_70 = 70000


def run_evaluation():
    print("Đang đọc dữ liệu dự đoán out-of-fold và thông tin gốc...")
    ml_preds = pd.read_parquet("data/processed/ml_test_predictions.parquet")
    dl_preds = pd.read_parquet("data/processed/dl_test_predictions.parquet")
    features = pd.read_parquet("data/processed/features_monthly.parquet")
    monthly = pd.read_parquet("data/processed/monthly_dataset.parquet")

    # 1. GHÉP BẢNG DỰ ĐOÁN VÀ KIỂM TRA TÍNH TOÀN VẸN COHORT
    df = ml_preds.merge(
        dl_preds,
        on=["employee_id", "month"],
        how="inner",
        validate="one_to_one",
    )
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    features["month"] = pd.PeriodIndex(features["month"], freq="M")
    monthly["month"] = pd.PeriodIndex(monthly["month"], freq="M")

    if len(df) != len(ml_preds) or len(df) != len(dl_preds):
        raise ValueError(
            f"Inner merge làm mất dữ liệu! Sau merge: {len(df):,} dòng "
            f"(ML có {len(ml_preds):,}, DL có {len(dl_preds):,}). Kiểm tra lại output ML/DL."
        )

    if "is_complete_month" not in features.columns:
        raise ValueError("features thiếu is_complete_month; chạy lại pipeline.")
    if "is_complete_month" not in monthly.columns:
        raise ValueError("monthly dataset thiếu is_complete_month; chạy lại pipeline.")
    
    complete_months = set(
        features.loc[features["is_complete_month"].astype(bool), "month"].unique()
    )
    prediction_months = set(df["month"].unique())
    if not prediction_months.issubset(complete_months):
        raise ValueError(
            "Evaluation chứa tháng chưa hoàn chỉnh hoặc stale prediction: "
            f"{sorted(prediction_months - complete_months)}. Chạy lại ML/DL."
        )
    
    # SỬA 1: Kiểm tra Cohort bằng so sánh tập hợp (set), không chỉ so sánh số lượng (count)
    for m in prediction_months:
        expected_ids = set(monthly.loc[monthly["month"] == m, "employee_id"])
        pred_ids = set(df.loc[df["month"] == m, "employee_id"])
        if expected_ids != pred_ids:
            diff_ids = expected_ids.symmetric_difference(pred_ids)
            raise ValueError(
                f"Prediction không khớp tập nhân viên tại tháng {m}: lệch {len(diff_ids)} nhân viên. "
                "Kiểm tra lại fallback cold-start hoặc tái tạo predictions."
            )
    
    # Bổ sung thông tin department
    if "department" in features.columns:
        dept_map = features[["employee_id", "month", "department"]].drop_duplicates()
        df = df.merge(dept_map, on=["employee_id", "month"], how="left")
        df["department"] = df["department"].fillna("Unknown")
    elif "department" in monthly.columns:
        dept_map = monthly[["employee_id", "month", "department"]].drop_duplicates()
        df = df.merge(dept_map, on=["employee_id", "month"], how="left")
        df["department"] = df["department"].fillna("Unknown")
    else:
        df["department"] = "Toàn công ty"

    models = ["m1", "m2", "m3", "m4", "m5"]
    months = sorted(df["month"].unique())

    # 2. TÍNH TOÁN S2 THEO WALK-FORWARD TRÊN TEST PERIOD
    df["pred_n44_s2"] = 0.0
    df["pred_n70_s2"] = 0.0

    for m in months:
        idx = df[df["month"] == m].index
        sub = df.loc[idx]

        prior = df[df["month"] < m]
        if prior.empty:
            w = np.full(len(models), 1.0 / len(models))
        else:
            prior_matrix = np.column_stack([
                prior[f"{mod}_p44"].values * PRICE_44 + prior[f"{mod}_p70"].values * PRICE_70
                for mod in models
            ])
            prior_cost = prior["N44"].values * PRICE_44 + prior["N70"].values * PRICE_70

            def loss_multi(weights):
                return np.mean(np.abs(prior_cost - prior_matrix @ weights))

            opt_s2 = minimize(
                loss_multi,
                np.full(len(models), 1.0 / len(models)),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(models),
                constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
            )
            w = opt_s2.x if opt_s2.success else np.full(len(models), 1.0 / len(models))

        df.loc[idx, "pred_n44_s2"] = np.clip(np.sum([w[i] * sub[f"{models[i]}_p44"].values for i in range(5)], axis=0), 0, None)
        df.loc[idx, "pred_n70_s2"] = np.clip(np.sum([w[i] * sub[f"{models[i]}_p70"].values for i in range(5)], axis=0), 0, None)

    # Tính tiền thực tế và dự đoán
    df["actual_cost"] = df["N44"] * PRICE_44 + df["N70"] * PRICE_70
    df["pred_cost"] = df["pred_n44_s2"] * PRICE_44 + df["pred_n70_s2"] * PRICE_70
    df["error_cost"] = df["pred_cost"] - df["actual_cost"]
    df["abs_error_cost"] = np.abs(df["error_cost"])

    # 3. ĐÁNH GIÁ CẤP ĐỘ TOÀN CÔNG TY (BUDGET LEVEL)
    print("\n" + "=" * 75)
    print("1. ĐÁNH GIÁ CẤP ĐỘ TOÀN CÔNG TY (BUDGET LEVEL)")
    print("=" * 75)
    company_eval = (
        df.groupby("month")
        .agg(
            Tong_Nhan_Vien=("employee_id", "count"),
            Tien_Thuc_Te=("actual_cost", "sum"),
            Tien_Du_Doan=("pred_cost", "sum"),
            Chenh_Lech=("error_cost", "sum"),
            MAE_Nhan_Vien=("abs_error_cost", "mean"),
        )
        .reset_index()
    )
    # SỬA 2: Bổ sung Budget WAPE và tỷ lệ lệch ngân sách chuẩn
    company_eval["Ti_Le_Lech_%"] = np.divide(
        company_eval["Chenh_Lech"] * 100,
        company_eval["Tien_Thuc_Te"],
        out=np.zeros(len(company_eval), dtype=float),
        where=company_eval["Tien_Thuc_Te"] != 0,
    )
    company_eval["Budget_WAPE_%"] = np.abs(company_eval["Ti_Le_Lech_%"])

    print(company_eval.to_string(index=False, formatters={
        "Tien_Thuc_Te": "{:,.0f} đ".format,
        "Tien_Du_Doan": "{:,.0f} đ".format,
        "Chenh_Lech": "{:+,.0f} đ".format,
        "MAE_Nhan_Vien": "{:,.0f} đ".format,
        "Ti_Le_Lech_%": "{:+.2f}%".format,
        "Budget_WAPE_%": "{:.2f}%".format,
    }))

    # 4. ĐÁNH GIÁ THEO PHÂN KHÚC HÀNH VI (BEHAVIOR SEGMENTS)
    print("\n" + "=" * 75)
    print("2. ĐÁNH GIÁ THEO PHÂN KHÚC HÀNH VI (BEHAVIOR SEGMENTS)")
    print("=" * 75)
    conditions = [
        (df["actual_cost"] == 0),
        (df["actual_cost"] > 0) & (df["actual_cost"] < 300000),
        (df["actual_cost"] >= 300000) & (df["actual_cost"] < 800000),
        (df["actual_cost"] >= 800000),
    ]
    labels = ["Không ăn (0 đ)", "Ăn ít (<300k)", "Ăn vừa (300k - 800k)", "Ăn nhiều (>=800k)"]
    df["segment"] = np.select(conditions, labels, default="Khác")

    segment_eval = (
        df.groupby("segment")
        .agg(
            So_Luong=("employee_id", "count"),
            MAE_Cost=("abs_error_cost", "mean"),
            RMSE_Cost=("error_cost", lambda x: np.sqrt(np.mean(x**2))),
            Tong_Thuc=("actual_cost", "sum"),
            Tong_Doan=("pred_cost", "sum"),
        )
        .reset_index()
    )
    segment_wape = (
        df.groupby("segment", sort=False)
        .agg(abs_error=("abs_error_cost", "sum"), actual=("actual_cost", "sum"))
    )
    
    # SỬA 3: Xử lý ngoại lệ actual == 0 trong WAPE.
    # Với nhóm "Không ăn (0 đ)", mẫu số bằng 0 -> trả về NaN thay vì 0% để tránh ngộ nhận sai số bằng 0.
    segment_wape["WAPE_%"] = np.where(
        segment_wape["actual"] == 0,
        np.nan,
        segment_wape["abs_error"] * 100 / segment_wape["actual"]
    )
    segment_eval = segment_eval.merge(
        segment_wape[["WAPE_%"]].reset_index(),
        on="segment",
        how="left",
    )

    print(segment_eval.to_string(index=False, formatters={
        "So_Luong": "{:,}".format,
        "MAE_Cost": "{:,.0f} đ".format,
        "RMSE_Cost": "{:,.0f} đ".format,
        "WAPE_%": lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A (Mẫu số = 0)",
        "Tong_Thuc": "{:,.0f} đ".format,
        "Tong_Doan": "{:,.0f} đ".format,
    }))

    # 5. PHÂN TÍCH NHÓM KHẨU VỊ: MÓN ÂU (N70 > 0)
    print("\n" + "=" * 75)
    print("3. PHÂN TÍCH NHÓM KHẨU VỊ: MÓN ÂU (N70 > 0)")
    print("=" * 75)
    df["has_au"] = df["N70"] > 0
    au_eval = (
        df.groupby("has_au")
        .agg(
            So_Luong=("employee_id", "count"),
            MAE_N70=("N70", lambda x: np.mean(np.abs(x - df.loc[x.index, "pred_n70_s2"]))),
            MAE_Cost=("abs_error_cost", "mean"),
        )
        .reset_index()
    )
    au_eval["Nhom"] = au_eval["has_au"].map({True: "Có ăn món Âu", False: "Không ăn món Âu"})
    print(au_eval[["Nhom", "So_Luong", "MAE_N70", "MAE_Cost"]].to_string(index=False, formatters={
        "So_Luong": "{:,}".format,
        "MAE_N70": "{:.2f}".format,
        "MAE_Cost": "{:,.0f} đ".format,
    }))

    # 6. TOP 10 TRƯỜNG HỢP NGOẠI LỆ SAI SỐ LỚN NHẤT (OUTLIERS)
    print("\n" + "=" * 75)
    print("4. TOP 10 TRƯỜNG HỢP NGOẠI LỆ SAI SỐ LỚN NHẤT (OUTLIERS)")
    print("=" * 75)
    top_outliers = df.sort_values("abs_error_cost", ascending=False).head(10)
    print(
        top_outliers[[
            "employee_id", "month", "N44", "pred_n44_s2", "N70", "pred_n70_s2", "actual_cost", "pred_cost", "error_cost"
        ]].to_string(
            index=False,
            formatters={
                "pred_n44_s2": "{:.1f}".format,
                "pred_n70_s2": "{:.1f}".format,
                "actual_cost": "{:,.0f} đ".format,
                "pred_cost": "{:,.0f} đ".format,
                "error_cost": "{:+,.0f} đ".format,
            },
        )
    )

    # 7. LƯU BÁO CÁO CHI TIẾT
    os.makedirs("results/metrics", exist_ok=True)
    df.to_parquet("results/metrics/final_predictions_evaluated.parquet", index=False)
    company_eval.to_csv("results/metrics/company_budget_eval.csv", index=False)
    segment_eval.to_csv("results/metrics/segment_eval.csv", index=False)
    top_outliers.to_csv("results/metrics/top_outliers.csv", index=False)
    print("\nĐã lưu toàn bộ báo cáo chi tiết trong thư mục results/metrics/")


if __name__ == "__main__":
    run_evaluation()