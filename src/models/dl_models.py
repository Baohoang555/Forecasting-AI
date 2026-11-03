
# src/models/dl_models.py

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.prediction_utils import (
    complete_prediction_cohort,
    latest_complete_months,
)

warnings.filterwarnings("ignore")

PRICE_44 = 44_000
PRICE_70 = 70_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

SEQUENCE_PREFIXES = (
    "N44_lag_",
    "N70_lag_",
    "total_meals_lag_",
    "days_with_meal_lag_",
    "total_amount_lag_",
)
SEQUENCE_STEPS = (1, 2, 3)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_numeric_features(df):
    selected = []
    for col in df.select_dtypes(include=[np.number]).columns:
        name = str(col).lower()
        if col in NON_FEATURE_COLUMNS:
            continue
        if any(token in name for token in FORBIDDEN_TOKENS):
            continue
        selected.append(col)

    if not selected:
        raise ValueError("Không tìm thấy numeric feature an toàn cho DL.")
    return selected


def _finite_matrix(values, name):
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        raise ValueError(f"{name} rỗng.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} còn NaN/Inf.")
    return arr


def _standardize(train_values, test_values):
    train_values = _finite_matrix(train_values, "X_train")
    test_values = _finite_matrix(test_values, "X_test")

    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0

    train_scaled = (train_values - mean) / std
    test_scaled = (test_values - mean) / std

    return (
        _finite_matrix(train_scaled, "X_train_scaled"),
        _finite_matrix(test_scaled, "X_test_scaled"),
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


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.training and x.ndim == 2 and x.shape[0] == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps,
            )
        return super().forward(x)


class GRUMultiTask(nn.Module):
    def __init__(self, seq_features, static_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc_static = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
        )
        self.head_44 = nn.Sequential(
            nn.Linear(hidden_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.head_70 = nn.Sequential(
            nn.Linear(hidden_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_static):
        out, _ = self.gru(x_seq)
        last_hidden = out[:, -1, :]
        static_out = self.fc_static(x_static)
        combined = torch.cat([last_hidden, static_out], dim=1)
        return (
            self.head_44(combined).squeeze(-1),
            self.head_70(combined).squeeze(-1),
        )


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            SafeBatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            SafeBatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            SafeBatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.head_44 = nn.Linear(hidden_dim, 1)
        self.head_70 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head_44(h).squeeze(-1), self.head_70(h).squeeze(-1)


def _train_model(
    model, dataset, test_inputs, y44, y70,
    epochs=30, batch_size=256, learning_rate=0.003,
):
    if len(dataset) == 0 or len(test_inputs[0]) == 0:
        raise ValueError("Train/test rỗng.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    loss_fn = nn.SmoothL1Loss()
    model = model.to(DEVICE)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            *inputs, target44, target70 = batch
            inputs = [x.to(DEVICE) for x in inputs]
            target44 = target44.to(DEVICE)
            target70 = target70.to(DEVICE)

            optimizer.zero_grad()
            pred44, pred70 = model(*inputs)
            loss = loss_fn(pred44, target44) + 1.5 * loss_fn(pred70, target70)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        inputs = [x.to(DEVICE) for x in test_inputs]
        pred44, pred70 = model(*inputs)

    return (
        np.clip(pred44.cpu().numpy().astype(float), 0, None),
        np.clip(pred70.cpu().numpy().astype(float), 0, None),
    )


def train_m4_gru(
    train_seq, train_stat, y44_tr, y70_tr,
    test_seq, test_stat, epochs=25, batch_size=256,
):
    train_seq = _finite_matrix(train_seq, "train_seq")
    train_stat = _finite_matrix(train_stat, "train_stat")
    test_seq = _finite_matrix(test_seq, "test_seq")
    test_stat = _finite_matrix(test_stat, "test_stat")
    y44_tr = _finite_matrix(y44_tr, "y44_train").reshape(-1)
    y70_tr = _finite_matrix(y70_tr, "y70_train").reshape(-1)

    if not (
        len(train_seq) == len(train_stat) == len(y44_tr) == len(y70_tr)
    ):
        raise ValueError("Kích thước dữ liệu train GRU không khớp.")

    dataset = TensorDataset(
        torch.tensor(train_seq, dtype=torch.float32),
        torch.tensor(train_stat, dtype=torch.float32),
        torch.tensor(y44_tr, dtype=torch.float32),
        torch.tensor(y70_tr, dtype=torch.float32),
    )
    model = GRUMultiTask(
        seq_features=train_seq.shape[2],
        static_dim=train_stat.shape[1],
    )
    return _train_model(
        model,
        dataset,
        (
            torch.tensor(test_seq, dtype=torch.float32),
            torch.tensor(test_stat, dtype=torch.float32),
        ),
        y44_tr,
        y70_tr,
        epochs=epochs,
        batch_size=batch_size,
    )


def train_m5_mlp(
    train_x, y44_tr, y70_tr, test_x,
    epochs=30, batch_size=256,
):
    train_x = _finite_matrix(train_x, "train_x")
    test_x = _finite_matrix(test_x, "test_x")
    y44_tr = _finite_matrix(y44_tr, "y44_train").reshape(-1)
    y70_tr = _finite_matrix(y70_tr, "y70_train").reshape(-1)

    if not (len(train_x) == len(y44_tr) == len(y70_tr)):
        raise ValueError("Kích thước dữ liệu train MLP không khớp.")

    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(y44_tr, dtype=torch.float32),
        torch.tensor(y70_tr, dtype=torch.float32),
    )
    model = TabularMLP(input_dim=train_x.shape[1])
    return _train_model(
        model,
        dataset,
        (torch.tensor(test_x, dtype=torch.float32),),
        y44_tr,
        y70_tr,
        epochs=epochs,
        batch_size=batch_size,
    )


def _build_sequence(df):
    required = [
        f"{prefix}{step}"
        for prefix in SEQUENCE_PREFIXES
        for step in SEQUENCE_STEPS
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột sequence GRU: {missing}")

    # Mỗi bước có 5 biến; trục thời gian được sắp xếp cũ -> mới.
    # Thứ tự: tháng t-3, t-2, t-1.
    sequence = np.stack(
        [
            df[[f"{prefix}{lag}" for prefix in SEQUENCE_PREFIXES]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_numpy(dtype=np.float32)
            for lag in (3, 2, 1)
        ],
        axis=1,
    )
    return _finite_matrix(sequence, "sequence")


def _scale_sequence(train_seq, test_seq):
    train_seq = _finite_matrix(train_seq, "train_seq")
    test_seq = _finite_matrix(test_seq, "test_seq")
    mean = train_seq.mean(axis=(0, 1), keepdims=True)
    std = train_seq.std(axis=(0, 1), keepdims=True)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return (
        _finite_matrix((train_seq - mean) / std, "train_seq_scaled"),
        _finite_matrix((test_seq - mean) / std, "test_seq_scaled"),
    )


def run_dl_pipeline(
    features_path="data/processed/features_monthly.parquet",
    monthly_path="data/processed/monthly_dataset.parquet",
):
    print(f"Khởi chạy DL trên thiết bị: {DEVICE}")
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

    numeric_features = select_numeric_features(df)
    sequence_cols = {
        f"{prefix}{lag}"
        for prefix in SEQUENCE_PREFIXES
        for lag in SEQUENCE_STEPS
    }
    missing_seq = sequence_cols - set(df.columns)
    if missing_seq:
        raise ValueError(f"Thiếu feature sequence: {sorted(missing_seq)}")

    static_features = [c for c in numeric_features if c not in sequence_cols]
    if not static_features:
        raise ValueError("Không có static features cho GRU.")

    test_months = latest_complete_months(monthly)
    if not test_months:
        raise ValueError("Không tìm thấy tháng hoàn chỉnh để backtest DL.")

    print(f"Features số: {len(numeric_features)}")
    print(f"Static features: {len(static_features)}")
    print(f"Tháng test: {[str(m) for m in test_months]}")

    prediction_pairs = [("m4_p44", "m4_p70"), ("m5_p44", "m5_p70")]
    model_names = ["M4_GRU", "M5_MLP"]
    results = []
    prediction_dumps = []

    for test_month in test_months:
        train_df = df[df["month"] < test_month].copy()
        test_df = df[df["month"] == test_month].copy()
        target_df = monthly[monthly["month"] == test_month].copy()

        if "is_complete_month" in train_df.columns:
            complete_flag = train_df["is_complete_month"]
            if complete_flag.dtype == object:
                complete_flag = (
                    complete_flag.astype(str).str.strip().str.lower()
                    .isin(["true", "1", "yes", "y"])
                )
            else:
                complete_flag = complete_flag.fillna(False).astype(bool)
            train_df = train_df[complete_flag]

        if target_df.empty:
            print(f"Bỏ qua {test_month}: không có nhãn tháng test.")
            continue

        unexpected = set(test_df["employee_id"]) - set(target_df["employee_id"])
        if unexpected:
            raise ValueError(
                f"Feature rows tháng {test_month} có nhân viên không có trong monthly."
            )

        if train_df.empty or test_df.empty:
            fold_preds = pd.DataFrame(
                columns=["employee_id", "m4_p44", "m4_p70", "m5_p44", "m5_p70"]
            )
        else:
            y44 = train_df["N44"].to_numpy(dtype=np.float32)
            y70 = train_df["N70"].to_numpy(dtype=np.float32)

            X_train_raw = train_df[numeric_features].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0).to_numpy(dtype=np.float32)
            X_test_raw = test_df[numeric_features].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0).to_numpy(dtype=np.float32)
            X_train_scaled, X_test_scaled = _standardize(
                X_train_raw, X_test_raw
            )

            static_idx = [numeric_features.index(c) for c in static_features]
            static_train = X_train_scaled[:, static_idx]
            static_test = X_test_scaled[:, static_idx]

            seq_train_raw = _build_sequence(train_df)
            seq_test_raw = _build_sequence(test_df)
            seq_train, seq_test = _scale_sequence(
                seq_train_raw, seq_test_raw
            )

            print(
                f"\n{test_month}: train={len(train_df):,}, "
                f"test={len(test_df):,}"
            )

            preds_m4_44, preds_m4_70 = [], []
            for seed in (42, 100, 2024):
                seed_everything(seed)
                p44, p70 = train_m4_gru(
                    seq_train, static_train, y44, y70,
                    seq_test, static_test, epochs=25,
                )
                preds_m4_44.append(p44)
                preds_m4_70.append(p70)

            preds_m5_44, preds_m5_70 = [], []
            for seed in (42, 100, 2024):
                seed_everything(seed)
                p44, p70 = train_m5_mlp(
                    X_train_scaled, y44, y70, X_test_scaled, epochs=30
                )
                preds_m5_44.append(p44)
                preds_m5_70.append(p70)

            fold_preds = pd.DataFrame({
                "employee_id": test_df["employee_id"].to_numpy(),
                "m4_p44": np.mean(preds_m4_44, axis=0),
                "m4_p70": np.mean(preds_m4_70, axis=0),
                "m5_p44": np.mean(preds_m5_44, axis=0),
                "m5_p70": np.mean(preds_m5_70, axis=0),
            })

            for model_name, col44, col70 in (
                ("M4_GRU", "m4_p44", "m4_p70"),
                ("M5_MLP", "m5_p44", "m5_p70"),
            ):
                pure_metrics = calculate_metrics(
                    test_df["N44"].to_numpy(),
                    test_df["N70"].to_numpy(),
                    fold_preds[col44].to_numpy(),
                    fold_preds[col70].to_numpy(),
                )
                print(
                    f"  [Pure DL] {model_name}: "
                    f"MAE={pure_metrics['MAE_Cost']:,.0f} đ | "
                    f"WAPE={pure_metrics['WAPE_Cost_%']:.2f}%"
                )

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
                ["employee_id", "month",
                 *(col for pair in prediction_pairs for col in pair)]
            ].copy()
        )

    if not results or not prediction_dumps:
        raise ValueError("Backtest DL không tạo được kết quả.")

    summary = pd.DataFrame(results)
    metric_cols = ["MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"]
    avg = (
        summary.groupby("Model")[metric_cols]
        .mean()
        .reset_index()
        .sort_values("MAE_Cost")
    )

    print("\n===== KẾT QUẢ DL END-TO-END =====")
    print(avg.to_string(index=False, formatters={
        "MAE_Cost": "{:,.0f} đ".format,
        "RMSE_Cost": "{:,.0f} đ".format,
        "WAPE_Cost_%": "{:.2f}%".format,
        "MAE_N44": "{:.2f}".format,
        "MAE_N70": "{:.2f}".format,
    }))

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    avg.to_csv("results/metrics/dl_models_summary.csv", index=False)
    pd.concat(prediction_dumps, ignore_index=True).to_parquet(
        "data/processed/dl_test_predictions.parquet", index=False
    )
    print("Đã lưu results/metrics/dl_models_summary.csv")
    print("Đã lưu data/processed/dl_test_predictions.parquet")


if __name__ == "__main__":
    run_dl_pipeline()