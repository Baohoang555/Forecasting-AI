import os
from typing import Optional

import numpy as np
import pandas as pd

PRICE_44 = 44000
PRICE_70 = 70000


# ============================================================
# CÁC MÔ HÌNH THUẦN NUMPY 
# ============================================================
class PureNumpyRidge:
    """Ridge Regression giải tích thuần NumPy: w = (X^T X + alpha*I)^(-1) X^T y"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w: Optional[np.ndarray] = None
        self.intercept = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Trừ trung bình để tính intercept
        x_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - x_mean
        y_centered = y - y_mean
        
        n_features = X.shape[1]
        A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
        b = X_centered.T @ y_centered
        
        self.w = np.linalg.solve(A, b)
        self.intercept = y_mean - x_mean @ self.w
        return self

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("Mô hình Ridge chưa được fit.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w + self.intercept


class PureNumpyDecisionTreeRegressor:
    """Cây quyết định hồi quy thuần NumPy (giảm phương sai, MSE)"""
    def __init__(self, max_depth=4, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _best_split(self, X, y):
        best_mse = float("inf")
        best_feat, best_thresh = None, None
        n_samples, n_feats = X.shape

        for feat in range(n_feats):
            vals = X[:, feat]
            thresholds = np.unique(vals)
            if len(thresholds) > 20:
                # Lấy 20 quantiles để tăng tốc tính toán
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 20))

            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)

                if n_l < 2 or n_r < 2:
                    continue

                y_l, y_r = y[left_mask], y[right_mask]
                mse = np.sum((y_l - np.mean(y_l)) ** 2) + np.sum((y_r - np.mean(y_r)) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build_tree(self, X, y, depth=0):
        mean_val = float(np.mean(y)) if len(y) > 0 else 0.0
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {"type": "leaf", "value": mean_val}

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {"type": "leaf", "value": mean_val}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        return {
            "type": "split",
            "feat": feat,
            "thresh": thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.tree = self._build_tree(X, y)
        return self

    def _predict_row(self, row, node):
        if node["type"] == "leaf":
            return node["value"]
        if row[node["feat"]] <= node["thresh"]:
            return self._predict_row(row, node["left"])
        return self._predict_row(row, node["right"])

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_row(row, self.tree) for row in X])


# ============================================================
# TÍNH METRICS & BACKTEST
# ============================================================
def calculate_metrics(y_true_n44, y_true_n70, y_pred_n44, y_pred_n70):
    y_pred_n44 = np.clip(y_pred_n44, 0, None)
    y_pred_n70 = np.clip(y_pred_n70, 0, None)

    cost_true = y_true_n44 * PRICE_44 + y_true_n70 * PRICE_70
    cost_pred = y_pred_n44 * PRICE_44 + y_pred_n70 * PRICE_70

    mae_cost = np.mean(np.abs(cost_true - cost_pred))
    rmse_cost = np.sqrt(np.mean((cost_true - cost_pred) ** 2))
    total_cost = np.sum(cost_true)
    wape_cost = (
        np.sum(np.abs(cost_true - cost_pred)) / total_cost * 100
        if total_cost > 0
        else 0.0
    )

    mae_n44 = np.mean(np.abs(y_true_n44 - y_pred_n44))
    mae_n70 = np.mean(np.abs(y_true_n70 - y_pred_n70))

    return {
        "MAE_Cost": mae_cost,
        "RMSE_Cost": rmse_cost,
        "WAPE_Cost_%": wape_cost,
        "MAE_N44": mae_n44,
        "MAE_N70": mae_n70,
    }


def run_baselines(data_path="data/processed/monthly_dataset.parquet"):
    df = pd.read_parquet(data_path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df = df.sort_values(["employee_id", "month"]).reset_index(drop=True)

    # 1. SỬA LỖI CALENDAR-GAP CHO CÁC LAG TRONG BASELINES
    # Kiểm tra khoảng cách tháng lịch thực tế (diff == lag) trước khi gán giá trị
    m_series = df["month"]
    for lag in [1, 2, 3]:
        m_shift = df.groupby("employee_id")["month"].shift(lag)
        diff_months = (m_series - m_shift).map(lambda d: getattr(d, "n", np.nan))
        valid_lag = (diff_months == lag)

        shifted_44 = df.groupby("employee_id")["N44"].shift(lag)
        shifted_70 = df.groupby("employee_id")["N70"].shift(lag)

        df[f"n44_lag_{lag}"] = shifted_44.where(valid_lag, np.nan)
        df[f"n70_lag_{lag}"] = shifted_70.where(valid_lag, np.nan)

    # 2. SỬA LỖI MA3: TÍNH TRỰC TIẾP TỪ 3 THÁNG LỊCH LIÊN TIẾP (T-1, T-2, T-3)
    # Tránh rolling theo dòng làm lấy nhầm các tháng bị gián đoạn
    df["n44_ma3"] = df[["n44_lag_1", "n44_lag_2", "n44_lag_3"]].mean(axis=1)
    df["n70_ma3"] = df[["n70_lag_1", "n70_lag_2", "n70_lag_3"]].mean(axis=1)

    # 3. SỬA LỖI HARDCODE "2025-10" & LOẠI BỎ BIẾN ĐỔI GIAN LẬN BẰNG FILLNA(0)
    # Lọc động: Chỉ đánh giá trên các dòng nhân viên có đủ 3 tháng lịch sử liên tiếp.
    # Không hardcode "2025-10", giúp code linh hoạt khi dữ liệu mở rộng thời gian.
    feat_cols = ["n44_lag_1", "n70_lag_1", "n44_lag_2", "n70_lag_2", "n44_lag_3", "n70_lag_3"]
    df_eval = df[df[feat_cols].notna().all(axis=1)].copy()

    # Chọn 3 tháng test TỰ ĐỘNG = 3 tháng gần nhất hiện có trong dữ liệu
    available_months = sorted(df_eval["month"].unique())
    test_months = available_months[-3:]
    print(f"Các tháng test (3 tháng gần nhất hiện có): {[str(m) for m in test_months]}")

    print("\n" + "=" * 65)
    print("BACKTESTING CÁC BASELINE (Expanding Window qua 3 tháng Test)")
    print("=" * 65)

    all_results = []

    for test_m in test_months:
        train_df = df_eval[df_eval["month"] < test_m].copy()
        test_df = df_eval[df_eval["month"] == test_m].copy()

        y_true_n44 = test_df["N44"].values
        y_true_n70 = test_df["N70"].values

        # 1. Naive Lag-1 (Do df_eval đã lọc đủ lag nên lag_1 luôn tồn tại giá trị thực tế)
        p_l1_44 = test_df["n44_lag_1"].values
        p_l1_70 = test_df["n70_lag_1"].values
        res_l1 = calculate_metrics(y_true_n44, y_true_n70, p_l1_44, p_l1_70)
        res_l1.update({"Model": "Naive Lag-1", "Test_Month": str(test_m)})
        all_results.append(res_l1)

        # 2. Moving Average 3M
        p_ma3_44 = test_df["n44_ma3"].values
        p_ma3_70 = test_df["n70_ma3"].values
        res_ma3 = calculate_metrics(y_true_n44, y_true_n70, p_ma3_44, p_ma3_70)
        res_ma3.update({"Model": "MA 3M", "Test_Month": str(test_m)})
        all_results.append(res_ma3)

        # 3. Group/Global Mean (Hoàn toàn chỉ fit từ train_df)
        dept_col = "department" if "department" in train_df.columns else None
        if dept_col:
            dept_stats = train_df.groupby(dept_col)[["N44", "N70"]].mean()
            pred_gm = test_df[[dept_col]].merge(dept_stats, on=dept_col, how="left")
            p_gm_44 = pred_gm["N44"].fillna(train_df["N44"].mean()).values
            p_gm_70 = pred_gm["N70"].fillna(train_df["N70"].mean()).values
        else:
            p_gm_44 = np.full(len(test_df), train_df["N44"].mean())
            p_gm_70 = np.full(len(test_df), train_df["N70"].mean())

        res_gm = calculate_metrics(y_true_n44, y_true_n70, p_gm_44, p_gm_70)
        res_gm.update({"Model": "Group/Global Mean", "Test_Month": str(test_m)})
        all_results.append(res_gm)

        # Chuẩn bị feature lag cho Ridge & Decision Tree
        X_train = train_df[feat_cols].values
        X_test = test_df[feat_cols].values

        # 4. Ridge Regression (Pure NumPy)
        ridge_44 = PureNumpyRidge(alpha=10.0).fit(X_train, train_df["N44"].values)
        ridge_70 = PureNumpyRidge(alpha=10.0).fit(X_train, train_df["N70"].values)
        p_rd_44 = ridge_44.predict(X_test)
        p_rd_70 = ridge_70.predict(X_test)
        res_rd = calculate_metrics(y_true_n44, y_true_n70, p_rd_44, p_rd_70)
        res_rd.update({"Model": "Ridge", "Test_Month": str(test_m)})
        all_results.append(res_rd)

        # 5. Shallow Decision Tree (Pure NumPy, depth=4)
        dt_44 = PureNumpyDecisionTreeRegressor(max_depth=4).fit(X_train, train_df["N44"].values)
        dt_70 = PureNumpyDecisionTreeRegressor(max_depth=4).fit(X_train, train_df["N70"].values)
        p_dt_44 = dt_44.predict(X_test)
        p_dt_70 = dt_70.predict(X_test)
        res_dt = calculate_metrics(y_true_n44, y_true_n70, p_dt_44, p_dt_70)
        res_dt.update({"Model": "DecisionTree (d=4)", "Test_Month": str(test_m)})
        all_results.append(res_dt)

    summary_df = pd.DataFrame(all_results)
    avg_df = (
        summary_df.groupby("Model")[["MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"]]
        .mean()
        .reset_index()
        .sort_values("MAE_Cost")
    )

    print("\n=== KẾT QUẢ BASELINES TRUNG BÌNH QUA 3 THÁNG TEST ===")
    print(
        avg_df.to_string(
            index=False,
            formatters={
                "MAE_Cost": "{:,.0f} đ".format,
                "RMSE_Cost": "{:,.0f} đ".format,
                "WAPE_Cost_%": "{:.2f}%".format,
                "MAE_N44": "{:.2f}".format,
                "MAE_N70": "{:.2f}".format,
            },
        )
    )

    os.makedirs("results/metrics", exist_ok=True)
    avg_df.to_csv("results/metrics/baselines_summary.csv", index=False)
    print("\nĐã lưu kết quả tại: results/metrics/baselines_summary.csv")


if __name__ == "__main__":
    run_baselines()