"""Evaluate pairwise and multi-model ensembles using time-based predictions."""

import itertools
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.prediction_utils import latest_complete_months


PRICE_44 = 44_000
PRICE_70 = 70_000

MODELS = ["m1", "m2", "m3", "m4", "m5"]

MODEL_NAMES = {
    "m1": "M1(LGB-RF)",
    "m2": "M2(LGB)",
    "m3": "M3(CatB)",
    "m4": "M4(GRU)",
    "m5": "M5(MLP)",
}

KEYS = ["employee_id", "month"]
TARGETS = ["N44", "N70"]
METRICS = ["MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"]


def normalize_month(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = set(KEYS)
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} thiếu cột: {sorted(missing)}.")

    result = df.copy()

    if result["employee_id"].isna().any():
        raise ValueError(f"{name} có employee_id bị thiếu.")

    try:
        result["month"] = pd.PeriodIndex(result["month"], freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} có cột month không hợp lệ.") from exc

    if result["month"].isna().any():
        raise ValueError(f"{name} có month bị thiếu.")

    if result.duplicated(KEYS).any():
        raise ValueError(f"{name} có employee-month trùng lặp.")

    return result


def parse_complete_flags(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("is_complete_month có giá trị thiếu.")

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}

    invalid = ~normalized.isin(mapping)
    if invalid.any():
        raise ValueError(
            "is_complete_month chỉ được chứa bool hoặc giá trị 0/1."
        )

    return normalized.map(mapping).astype(bool)


def validate_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"{name} thiếu cột {column}.")

        values = pd.to_numeric(df[column], errors="coerce").to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name}.{column} có giá trị thiếu hoặc không hữu hạn."
            )


def calculate_metrics(
    y_true_n44,
    y_true_n70,
    y_pred_n44,
    y_pred_n70,
) -> dict:
    y_true_n44 = np.asarray(y_true_n44, dtype=float)
    y_true_n70 = np.asarray(y_true_n70, dtype=float)
    y_pred_n44 = np.asarray(y_pred_n44, dtype=float)
    y_pred_n70 = np.asarray(y_pred_n70, dtype=float)

    arrays = [y_true_n44, y_true_n70, y_pred_n44, y_pred_n70]

    if not len(y_true_n44):
        raise ValueError("Không thể tính metric trên tập rỗng.")

    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Metric đầu vào phải là mảng một chiều.")

    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Metric đầu vào không cùng số dòng.")

    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Metric đầu vào có NaN hoặc vô cực.")

    y_pred_n44 = np.clip(y_pred_n44, 0, None)
    y_pred_n70 = np.clip(y_pred_n70, 0, None)

    cost_true = y_true_n44 * PRICE_44 + y_true_n70 * PRICE_70
    cost_pred = y_pred_n44 * PRICE_44 + y_pred_n70 * PRICE_70

    error = cost_true - cost_pred
    absolute_error = np.abs(error)
    total_cost = np.sum(np.abs(cost_true))

    wape_cost = (
        np.sum(absolute_error) / total_cost * 100
        if total_cost > 0
        else np.nan
    )

    return {
        "MAE_Cost": float(np.mean(absolute_error)),
        "RMSE_Cost": float(np.sqrt(np.mean(error**2))),
        "WAPE_Cost_%": float(wape_cost),
        "MAE_N44": float(np.mean(np.abs(y_true_n44 - y_pred_n44))),
        "MAE_N70": float(np.mean(np.abs(y_true_n70 - y_pred_n70))),
    }


def optimize_e1_weight(
    p_a_44,
    p_a_70,
    p_b_44,
    p_b_70,
    y44,
    y70,
) -> float:
    arrays = [
        np.asarray(x, dtype=float)
        for x in [p_a_44, p_a_70, p_b_44, p_b_70, y44, y70]
    ]

    if len(arrays[0]) == 0:
        return 0.5

    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Dữ liệu tối ưu trọng số không cùng số dòng.")

    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Dữ liệu tối ưu trọng số có NaN hoặc vô cực.")

    p_a_44, p_a_70, p_b_44, p_b_70, y44, y70 = arrays

    cost_true = y44 * PRICE_44 + y70 * PRICE_70
    cost_a = p_a_44 * PRICE_44 + p_a_70 * PRICE_70
    cost_b = p_b_44 * PRICE_44 + p_b_70 * PRICE_70

    def loss(weight):
        predicted = weight[0] * cost_a + (1 - weight[0]) * cost_b
        return np.mean(np.abs(cost_true - predicted))

    result = minimize(
        loss,
        x0=[0.5],
        bounds=[(0.0, 1.0)],
        method="L-BFGS-B",
    )

    if not result.success or not np.isfinite(result.x[0]):
        return 0.5

    return float(np.clip(result.x[0], 0.0, 1.0))


def fit_convex_weights(cost_matrix, y_cost) -> np.ndarray:
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    y_cost = np.asarray(y_cost, dtype=float)

    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix phải là ma trận 2 chiều.")

    n_models = cost_matrix.shape[1]

    if n_models == 0:
        raise ValueError("Không có mô hình để ensemble.")

    if cost_matrix.shape[0] != len(y_cost):
        raise ValueError("cost_matrix và y_cost không cùng số dòng.")

    if not len(y_cost):
        return np.full(n_models, 1.0 / n_models)

    if not np.isfinite(cost_matrix).all() or not np.isfinite(y_cost).all():
        raise ValueError("Dữ liệu tối ưu trọng số có NaN hoặc vô cực.")

    def loss(weights):
        return np.mean(np.abs(y_cost - cost_matrix @ weights))

    result = minimize(
        loss,
        x0=np.full(n_models, 1.0 / n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
        },
        options={"maxiter": 2000, "ftol": 1e-9},
    )

    if not result.success or not np.isfinite(result.x).all():
        return np.full(n_models, 1.0 / n_models)

    weights = np.clip(result.x, 0.0, 1.0)
    weight_sum = weights.sum()

    if weight_sum <= 0:
        return np.full(n_models, 1.0 / n_models)

    return weights / weight_sum


def load_and_validate_data():
    print("Đang đọc dữ liệu dự đoán ML, DL và monthly dataset...")

    ml = normalize_month(
        pd.read_parquet("data/processed/ml_test_predictions.parquet"),
        "ML predictions",
    )
    dl = normalize_month(
        pd.read_parquet("data/processed/dl_test_predictions.parquet"),
        "DL predictions",
    )
    monthly = normalize_month(
        pd.read_parquet("data/processed/monthly_dataset.parquet"),
        "Monthly dataset",
    )

    if "is_complete_month" not in monthly.columns:
        raise ValueError(
            "monthly_dataset.parquet thiếu is_complete_month."
        )

    monthly["is_complete_month"] = parse_complete_flags(
        monthly["is_complete_month"]
    )

    ml_models = ["m1", "m2", "m3"]
    dl_models = ["m4", "m5"]

    ml_prediction_columns = [
        f"{model}_p{meal}"
        for model in ml_models
        for meal in [44, 70]
    ]
    dl_prediction_columns = [
        f"{model}_p{meal}"
        for model in dl_models
        for meal in [44, 70]
    ]
    prediction_columns = ml_prediction_columns + dl_prediction_columns

    missing_ml = set(ml_prediction_columns) - set(ml.columns)
    missing_dl = set(dl_prediction_columns) - set(dl.columns)

    if missing_ml:
        raise ValueError(
            f"ML predictions thiếu cột: {sorted(missing_ml)}."
        )

    if missing_dl:
        raise ValueError(
            f"DL predictions thiếu cột: {sorted(missing_dl)}."
        )

    validate_numeric_columns(ml, ml_prediction_columns, "ML predictions")
    validate_numeric_columns(dl, dl_prediction_columns, "DL predictions")

    ml_keys = set(map(tuple, ml[KEYS].itertuples(index=False, name=None)))
    dl_keys = set(map(tuple, dl[KEYS].itertuples(index=False, name=None)))

    if ml_keys != dl_keys:
        raise ValueError(
            "Cohort ML và DL không giống nhau. "
            f"Chỉ có ở ML: {len(ml_keys - dl_keys):,}; "
            f"chỉ có ở DL: {len(dl_keys - ml_keys):,}."
        )

    preds = ml[KEYS + ml_prediction_columns].merge(
        dl[KEYS + dl_prediction_columns],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(preds) != len(ml) or len(preds) != len(dl):
        raise ValueError("Ghép ML/DL làm mất bản ghi.")

    actual = monthly[KEYS + TARGETS + ["is_complete_month"]].copy()
    validate_numeric_columns(actual, TARGETS, "Monthly dataset")

    df = preds.merge(
        actual,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_actual = df["_merge"] != "both"
    if missing_actual.any():
        raise ValueError(
            f"Có {int(missing_actual.sum()):,} employee-month "
            "không tìm thấy trong monthly dataset."
        )

    df = df.drop(columns="_merge")

    labels = ml[KEYS + TARGETS].merge(
        df[KEYS + TARGETS],
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("_ml", "_monthly"),
        indicator=True,
    )

    if not (labels["_merge"] == "both").all():
        raise ValueError(
            "Nhãn thực tế trong ML predictions không khớp monthly dataset."
        )

    for target in TARGETS:
        if not np.allclose(
            labels[f"{target}_ml"].to_numpy(dtype=float),
            labels[f"{target}_monthly"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError(
                f"Nhãn thực tế {target} trong ML predictions "
                "không khớp monthly dataset."
            )

    validate_numeric_columns(df, prediction_columns, "Ensemble predictions")

    incomplete_predictions = ~df["is_complete_month"]
    if incomplete_predictions.any():
        raise ValueError(
            "Dữ liệu dự đoán chứa tháng chưa hoàn chỉnh: "
            f"{sorted(df.loc[incomplete_predictions, 'month'].astype(str).unique())}."
        )

    prediction_months = sorted(df["month"].unique())

    # Helper latest_complete_months sử dụng API mặc định của project.
    expected_months = pd.PeriodIndex(
        latest_complete_months(monthly),
        freq="M",
    ).sort_values().tolist()

    if prediction_months != expected_months:
        raise ValueError(
            "Các tháng dự đoán không khớp 3 tháng hoàn chỉnh gần nhất. "
            f"Đang có: {[str(m) for m in prediction_months]}; "
            f"mong đợi: {[str(m) for m in expected_months]}. "
            "Hãy chạy lại ML/DL sau khi kiểm tra dữ liệu."
        )

    complete_monthly = monthly[monthly["is_complete_month"]]

    for month in prediction_months:
        expected_ids = set(
            complete_monthly.loc[
                complete_monthly["month"] == month, "employee_id"
            ]
        )
        predicted_ids = set(
            df.loc[df["month"] == month, "employee_id"]
        )

        if expected_ids != predicted_ids:
            raise ValueError(
                f"Cohort tháng {month} không khớp monthly dataset. "
                f"Chênh lệch: "
                f"{len(expected_ids.symmetric_difference(predicted_ids)):,} nhân viên."
            )

    return df, prediction_months


def format_results(df: pd.DataFrame) -> str:
    return df.to_string(
        index=False,
        formatters={
            "MAE_Cost": "{:,.0f} đ".format,
            "RMSE_Cost": "{:,.0f} đ".format,
            "WAPE_Cost_%": lambda x: (
                f"{x:.2f}%" if pd.notna(x) else "N/A"
            ),
            "MAE_N44": "{:.2f}".format,
            "MAE_N70": "{:.2f}".format,
        },
    )


def run_ensemble():
    df, months = load_and_validate_data()

    pair_results = []

    print("\n" + "=" * 75)
    print("ĐÁNH GIÁ ENSEMBLE CẶP ĐÔI")
    print("=" * 75)

    for model_a, model_b in itertools.combinations(MODELS, 2):
        label = f"{MODEL_NAMES[model_a]} + {MODEL_NAMES[model_b]}"
        e0_results = []
        e1_results = []

        for month in months:
            sub = df[df["month"] == month]

            y44 = sub["N44"].to_numpy(dtype=float)
            y70 = sub["N70"].to_numpy(dtype=float)

            a44 = sub[f"{model_a}_p44"].to_numpy(dtype=float)
            a70 = sub[f"{model_a}_p70"].to_numpy(dtype=float)
            b44 = sub[f"{model_b}_p44"].to_numpy(dtype=float)
            b70 = sub[f"{model_b}_p70"].to_numpy(dtype=float)

            e0 = calculate_metrics(
                y44,
                y70,
                (a44 + b44) / 2,
                (a70 + b70) / 2,
            )
            e0_results.append(e0)

            # Chỉ dùng các tháng trước tháng đang đánh giá.
            prior = df[df["month"] < month]

            weight = optimize_e1_weight(
                prior[f"{model_a}_p44"].to_numpy(dtype=float),
                prior[f"{model_a}_p70"].to_numpy(dtype=float),
                prior[f"{model_b}_p44"].to_numpy(dtype=float),
                prior[f"{model_b}_p70"].to_numpy(dtype=float),
                prior["N44"].to_numpy(dtype=float),
                prior["N70"].to_numpy(dtype=float),
            )

            e1 = calculate_metrics(
                y44,
                y70,
                weight * a44 + (1 - weight) * b44,
                weight * a70 + (1 - weight) * b70,
            )
            e1["Weight_A"] = weight
            e1_results.append(e1)

        for results, technique in [
            (e0_results, "E0 (Simple Avg)"),
            (e1_results, "E1 (Optimal Weight)"),
        ]:
            summary = (
                pd.DataFrame(results)
                .mean(numeric_only=True)
                .to_dict()
            )
            summary.update({"Pair": label, "Technique": technique})
            pair_results.append(summary)

    pairs_df = pd.DataFrame(pair_results).sort_values(
        "MAE_Cost",
        na_position="last",
    ).reset_index(drop=True)

    print("\nTop cấu hình ensemble cặp đôi — macro-average theo tháng")
    print("MAE_Cost và RMSE_Cost có đơn vị đồng/nhân viên/tháng.")
    print(format_results(
        pairs_df.head(10)[
            [
                "Pair",
                "Technique",
                "MAE_Cost",
                "RMSE_Cost",
                "WAPE_Cost_%",
                "MAE_N44",
                "MAE_N70",
            ]
        ]
    ))

    multi_results = []

    for month in months:
        sub = df[df["month"] == month]
        y44 = sub["N44"].to_numpy(dtype=float)
        y70 = sub["N70"].to_numpy(dtype=float)

        p_s0_44 = np.mean(
            [
                sub[f"{model}_p44"].to_numpy(dtype=float)
                for model in MODELS
            ],
            axis=0,
        )
        p_s0_70 = np.mean(
            [
                sub[f"{model}_p70"].to_numpy(dtype=float)
                for model in MODELS
            ],
            axis=0,
        )

        result_s0 = calculate_metrics(y44, y70, p_s0_44, p_s0_70)
        result_s0.update({
            "Method": "S0 (All 5 Simple Avg)",
            "Month": str(month),
        })
        multi_results.append(result_s0)

        prior = df[df["month"] < month]

        if prior.empty:
            cost_matrix = np.empty((0, len(MODELS)), dtype=float)
            prior_cost = np.array([], dtype=float)
        else:
            cost_matrix = np.column_stack([
                prior[f"{model}_p44"].to_numpy(dtype=float) * PRICE_44
                + prior[f"{model}_p70"].to_numpy(dtype=float) * PRICE_70
                for model in MODELS
            ])

            prior_cost = (
                prior["N44"].to_numpy(dtype=float) * PRICE_44
                + prior["N70"].to_numpy(dtype=float) * PRICE_70
            )

        weights = fit_convex_weights(cost_matrix, prior_cost)

        p_s2_44 = sum(
            weights[i] * sub[f"{model}_p44"].to_numpy(dtype=float)
            for i, model in enumerate(MODELS)
        )
        p_s2_70 = sum(
            weights[i] * sub[f"{model}_p70"].to_numpy(dtype=float)
            for i, model in enumerate(MODELS)
        )

        result_s2 = calculate_metrics(y44, y70, p_s2_44, p_s2_70)
        result_s2.update({
            "Method": "S2 (5-Model Convex Opt)",
            "Month": str(month),
        })
        multi_results.append(result_s2)

    multi_raw = pd.DataFrame(multi_results)

    multi_df = (
        multi_raw.groupby("Method")[METRICS]
        .mean()
        .reset_index()
        .sort_values("MAE_Cost", na_position="last")
        .reset_index(drop=True)
    )

    print("\nKết quả ensemble 5 mô hình")
    print(format_results(multi_df))

    os.makedirs("results/metrics", exist_ok=True)

    pairs_df.to_csv(
        "results/metrics/ensemble_pairs_summary.csv",
        index=False,
    )
    multi_df.to_csv(
        "results/metrics/ensemble_multi_summary.csv",
        index=False,
    )
    multi_raw.to_csv(
        "results/metrics/ensemble_multi_by_month.csv",
        index=False,
    )

    print("\nĐã lưu:")
    print("- results/metrics/ensemble_pairs_summary.csv")
    print("- results/metrics/ensemble_multi_summary.csv")
    print("- results/metrics/ensemble_multi_by_month.csv")


if __name__ == "__main__":
    run_ensemble()