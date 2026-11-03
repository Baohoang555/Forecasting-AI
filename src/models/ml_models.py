
# src/models/ml_models.py

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.models.prediction_utils import (
    complete_prediction_cohort,
    latest_complete_months,
)

warnings.filterwarnings("ignore")

PRICE_44 = 44_000
PRICE_70 = 70_000

TARGET_COLUMNS = {"N44", "N70"}
NON_FEATURE_COLUMNS = {
    "employee_id", "month", "department",
    "N44", "N70", "total_meals", "total_amount",
    "days_with_meal", "calendar_days", "expected_amount",
    "amount_diff", "au_share", "is_complete_month",
    "pred_N44", "pred_N70", "pred_cost",
    "N44_lag1", "N44_lag2", "N44_lag3",
    "N70_lag1", "N70_lag2", "N70_lag3",
    "au_share_3m",
}
FORBIDDEN_TOKENS = (
    "future", "next_month", "target", "actual",
    "label", "leakage", "pred_cost", "pred_n44", "pred_n70",
)


def calculate_metrics(y_true_n44, y_true_n70, y_pred_n44, y_pred_n70):
    y_true_n44 = np.asarray(y_true_n44, dtype=float)
    y_true_n70 = np.asarray(y_true_n70, dtype=float)
    y_pred_n44 = np.clip(np.asarray(y_pred_n44, dtype=float), 0, None)
    y_pred_n70 = np.clip(np.asarray(y_pred_n70, dtype=float), 0, None)

    cost_true = y_true_n44 * PRICE_44 + y_true_n70 * PRICE_70
    cost_pred = y_pred_n44 * PRICE_44 + y_pred_n70 * PRICE_70
    errors = cost_true - cost_pred

    return {
        "MAE_Cost": float(np.mean(np.abs(errors))),
        "RMSE_Cost": float(np.sqrt(np.mean(errors ** 2))),
        "WAPE_Cost_%": float(
            np.sum(np.abs(errors)) / np.sum(cost_true) * 100
            if np.sum(cost_true) > 0 else 0.0
        ),
        "MAE_N44": float(np.mean(np.abs(y_true_n44 - y_pred_n44))),
        "MAE_N70": float(np.mean(np.abs(y_true_n70 - y_pred_n70))),
    }


def select_numeric_features(df):
    """Chỉ chọn numeric features; loại nhãn và kết quả tháng hiện tại."""
    selected = []
    for col in df.select_dtypes(include=[np.number]).columns:
        name = str(col).lower()
        if col in NON_FEATURE_COLUMNS:
            continue
        if any(token in name for token in FORBIDDEN_TOKENS):
            continue
        selected.append(col)

    if not selected:
        raise ValueError("Không tìm thấy numeric feature hợp lệ.")
    return selected


def _prepare_numeric(frame, columns):
    result = frame.reindex(columns=columns).copy()
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result.astype(np.float32)


def train_predict_m1_rf(X_tr, y_tr_44, y_tr_70, X_te):
    params = {
        "boosting_type": "rf",
        "objective": "regression",
        "learning_rate": 0.1,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "verbosity": -1,
        "num_threads": -1,
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
        "force_col_wise": True,
    }

    def fit_predict(y):
        train_set = lgb.Dataset(X_tr, label=np.asarray(y, dtype=float))
        model = lgb.train(params, train_set, num_boost_round=100)
        return np.clip(np.asarray(model.predict(X_te), dtype=float), 0, None)

    return fit_predict(y_tr_44), fit_predict(y_tr_70)


def train_predict_m2_lgb(X_tr, y_tr_44, y_tr_70, X_te):
    params_44 = {
        "objective": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "num_threads": -1,
        "seed": 42,
        "force_col_wise": True,
    }
    params_70 = {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "verbosity": -1,
        "num_threads": -1,
        "seed": 42,
        "force_col_wise": True,
    }

    d44 = lgb.Dataset(X_tr, label=np.asarray(y_tr_44, dtype=float))
    d70 = lgb.Dataset(X_tr, label=np.asarray(y_tr_70, dtype=float))
    m44 = lgb.train(params_44, d44, num_boost_round=250)
    m70 = lgb.train(params_70, d70, num_boost_round=250)

    return (
        np.clip(np.asarray(m44.predict(X_te), dtype=float), 0, None),
        np.clip(np.asarray(m70.predict(X_te), dtype=float), 0, None),
    )


def train_predict_m3_catboost(
    X_tr, y_tr_44, y_tr_70, X_te, cat_features=None
):
    def fit_predict(y, iterations, learning_rate, depth):
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="MAE",
            random_seed=42,
            verbose=0,
            thread_count=-1,
            cat_features=cat_features or [],
            allow_writing_files=False,
        )
        model.fit(X_tr, np.asarray(y, dtype=float))
        return np.clip(np.asarray(model.predict(X_te), dtype=float), 0, None)

    return (
        fit_predict(y_tr_44, 250, 0.05, 6),
        fit_predict(y_tr_70, 250, 0.03, 5),
    )


def run_ml_pipeline(
    features_path="data/processed/features_monthly.parquet",
    monthly_path="data/processed/monthly_dataset.parquet",
):
    print("Đang đọc dữ liệu...")
    df = pd.read_parquet(features_path)
    monthly = pd.read_parquet(monthly_path)

    for name, frame in (("features", df), ("monthly", monthly)):
        required = {"employee_id", "month", "N44", "N70"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} thiếu cột: {sorted(missing)}")
        frame["month"] = pd.PeriodIndex(frame["month"], freq="M")
        if frame.duplicated(["employee_id", "month"]).any():
            raise ValueError(f"{name} có employee_id-month bị trùng.")

    if "department" not in df.columns:
        df["department"] = "unknown"
    df["department"] = df["department"].fillna("unknown").astype(str)

    feature_cols = select_numeric_features(df)
    catboost_cols = feature_cols + ["department"]
    cat_idx = [catboost_cols.index("department")]

    test_months = latest_complete_months(monthly)
    if not test_months:
        raise ValueError("Không tìm thấy tháng hoàn chỉnh để backtest.")

    print(f"Features số: {len(feature_cols)}")
    print(f"Tháng test: {[str(m) for m in test_months]}")

    results = []
    prediction_dumps = []
    prediction_pairs = [
        ("m1_p44", "m1_p70"),
        ("m2_p44", "m2_p70"),
        ("m3_p44", "m3_p70"),
    ]
    model_names = ["M1_RandomForest", "M2_LightGBM", "M3_CatBoost"]

    for test_month in test_months:
        train_df = df[df["month"] < test_month].copy()
        test_df = df[df["month"] == test_month].copy()
        target_df = monthly[monthly["month"] == test_month].copy()

        if "is_complete_month" in train_df.columns:
            train_df = train_df[train_df["is_complete_month"].fillna(False).astype(bool)]

        if target_df.empty:
            print(f"Bỏ qua {test_month}: không có nhãn tháng test.")
            continue

        target_ids = set(target_df["employee_id"])
        unexpected = set(test_df["employee_id"]) - target_ids
        if unexpected:
            raise ValueError(
                f"Feature rows tháng {test_month} có nhân viên không có trong monthly."
            )

        if train_df.empty or test_df.empty:
            fold_preds = pd.DataFrame(columns=["employee_id", *(c for p in prediction_pairs for c in p)])
        else:
            X_train = _prepare_numeric(train_df, feature_cols)
            X_test = _prepare_numeric(test_df, feature_cols)
            y44 = train_df["N44"].to_numpy(dtype=float)
            y70 = train_df["N70"].to_numpy(dtype=float)

            X_train_cb = train_df.reindex(columns=catboost_cols).copy()
            X_test_cb = test_df.reindex(columns=catboost_cols).copy()
            for frame in (X_train_cb, X_test_cb):
                frame["department"] = frame["department"].fillna("unknown").astype(str)
                frame[feature_cols] = _prepare_numeric(frame, feature_cols)

            print(
                f"\n{test_month}: train={len(train_df):,}, "
                f"test={len(test_df):,}"
            )
            p1_44, p1_70 = train_predict_m1_rf(X_train, y44, y70, X_test)
            p2_44, p2_70 = train_predict_m2_lgb(X_train, y44, y70, X_test)
            p3_44, p3_70 = train_predict_m3_catboost(
                X_train_cb, y44, y70, X_test_cb, cat_features=cat_idx
            )

            fold_preds = pd.DataFrame({
                "employee_id": test_df["employee_id"].to_numpy(),
                "m1_p44": p1_44, "m1_p70": p1_70,
                "m2_p44": p2_44, "m2_p70": p2_70,
                "m3_p44": p3_44, "m3_p70": p3_70,
            })

        history = monthly[monthly["month"] < test_month].copy()
        complete_preds = complete_prediction_cohort(
            history, target_df, fold_preds, prediction_pairs
        )

        for model_name, (col44, col70) in zip(model_names, prediction_pairs):
            metrics = calculate_metrics(
                complete_preds["N44"].to_numpy(),
                complete_preds["N70"].to_numpy(),
                complete_preds[col44].to_numpy(),
                complete_preds[col70].to_numpy(),
            )
            metrics.update({"Model": model_name, "Test_Month": str(test_month)})
            results.append(metrics)

        prediction_dumps.append(
            complete_preds[
                ["employee_id", "month", "N44", "N70",
                 *(c for pair in prediction_pairs for c in pair)]
            ].copy()
        )

    if not results or not prediction_dumps:
        raise ValueError("Backtest ML không tạo được kết quả.")

    summary = pd.DataFrame(results)
    metrics_cols = ["MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"]
    avg = (
        summary.groupby("Model")[metrics_cols]
        .mean()
        .reset_index()
        .sort_values("MAE_Cost")
    )

    print("\n===== KẾT QUẢ ML END-TO-END =====")
    print(avg.to_string(index=False, formatters={
        "MAE_Cost": "{:,.0f} đ".format,
        "RMSE_Cost": "{:,.0f} đ".format,
        "WAPE_Cost_%": "{:.2f}%".format,
        "MAE_N44": "{:.2f}".format,
        "MAE_N70": "{:.2f}".format,
    }))

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    avg.to_csv("results/metrics/ml_models_summary.csv", index=False)
    pd.concat(prediction_dumps, ignore_index=True).to_parquet(
        "data/processed/ml_test_predictions.parquet", index=False
    )
    print("Đã lưu results/metrics/ml_models_summary.csv")
    print("Đã lưu data/processed/ml_test_predictions.parquet")


if __name__ == "__main__":
    run_ml_pipeline()