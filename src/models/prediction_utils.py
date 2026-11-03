"""Shared helpers for time-based model evaluation and cold-start fallback."""

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["N44", "N70"]


def _normalize_month_column(df: pd.DataFrame) -> pd.Series:
    if "month" not in df.columns:
        raise ValueError("Dataset is missing the 'month' column.")

    try:
        return pd.Series(
            pd.PeriodIndex(df["month"], freq="M"),
            index=df.index,
            name="month",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "The 'month' column cannot be converted to Period[M] format."
        ) from exc


def latest_complete_months(
    df: pd.DataFrame,
    n_months: int = 3,
) -> list[pd.Period]:
    if (
        isinstance(n_months, bool)
        or not isinstance(n_months, (int, np.integer))
        or n_months < 1
    ):
        raise ValueError("n_months must be an integer greater than or equal to 1.")

    required = {"month", "is_complete_month"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns {sorted(missing)}. "
            "Please check monthly_aggregation.py."
        )

    if df.empty:
        raise ValueError("Dataset contains no data.")

    months = _normalize_month_column(df)
    raw_flags = df["is_complete_month"]

    if raw_flags.isna().any():
        raise ValueError("'is_complete_month' contains missing values.")

    if pd.api.types.is_bool_dtype(raw_flags):
        flags = raw_flags.astype(bool)
    else:
        normalized = raw_flags.astype(str).str.strip().str.lower()
        mapping = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }

        invalid = ~normalized.isin(mapping)
        if invalid.any():
            invalid_values = normalized.loc[invalid].unique().tolist()
            raise ValueError(
                "'is_complete_month' must only contain boolean or 0/1 values. "
                f"Invalid values: {invalid_values[:10]}."
            )

        flags = normalized.map(mapping).astype(bool)

    completeness_df = pd.DataFrame({
        "month": months.to_numpy(),
        "is_complete_month": flags.to_numpy(),
    })

    completeness = completeness_df.groupby(
        "month", sort=True
    )["is_complete_month"].all()

    complete_months = sorted(completeness[completeness].index.tolist())

    if not complete_months:
        raise ValueError("No months are marked as complete.")

    latest_run = [complete_months[-1]]

    for month in reversed(complete_months[:-1]):
        if month + 1 == latest_run[0]:
            latest_run.insert(0, month)
        else:
            break

    if len(latest_run) < n_months:
        raise ValueError(
            f"There are only {len(latest_run)} latest consecutive complete months, "
            f"but {n_months} months are required. "
            "Check data coverage and 'is_complete_month'."
        )

    return latest_run[-n_months:]


def department_mean_fallback(
    history: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    required_history = {"N44", "N70"}
    required_target = {"employee_id", "department"}

    if history.empty:
        raise ValueError("History is empty, cannot generate fallback.")

    missing_history = required_history - set(history.columns)
    missing_target = required_target - set(target.columns)

    if missing_history:
        raise ValueError(f"History is missing columns {sorted(missing_history)}.")

    if missing_target:
        raise ValueError(f"Target is missing columns {sorted(missing_target)}.")

    if target.empty:
        return pd.DataFrame(columns=["employee_id", "N44", "N70"])

    if target["employee_id"].isna().any():
        raise ValueError("Target contains missing employee_ids.")

    if target["employee_id"].duplicated().any():
        raise ValueError("Target contains duplicate employee_ids.")

    history_values = history[TARGET_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    history_values = history_values.replace([np.inf, -np.inf], np.nan)

    valid_mask = history_values.notna().all(axis=1)
    valid_history = history.loc[valid_mask].copy()
    valid_history[TARGET_COLUMNS] = history_values.loc[valid_mask].to_numpy()

    if valid_history.empty:
        raise ValueError("History has no rows with valid N44/N70 values.")

    global_mean = valid_history[TARGET_COLUMNS].mean()

    if not np.isfinite(global_mean.to_numpy(dtype=float)).all():
        raise ValueError("Unable to calculate a valid global mean.")

    department_means = valid_history.groupby(
        "department", dropna=True
    )[TARGET_COLUMNS].mean()

    predictions = target[["employee_id", "department"]].copy()

    for column in TARGET_COLUMNS:
        predictions[column] = predictions["department"].map(
            department_means[column]
        ).fillna(global_mean[column])

    predictions[TARGET_COLUMNS] = predictions[TARGET_COLUMNS].clip(lower=0)

    values = predictions[TARGET_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Fallback produced NaN or infinite predictions.")

    return predictions[["employee_id", "N44", "N70"]]


def complete_prediction_cohort(
    history: pd.DataFrame,
    target: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_pairs: list[tuple[str, str]],
) -> pd.DataFrame:

    if target.empty:
        raise ValueError("No employees in the test month.")

    if not prediction_pairs:
        raise ValueError("prediction_pairs cannot be empty.")

    if "employee_id" not in target.columns:
        raise ValueError("Target is missing 'employee_id' column.")

    if "employee_id" not in predictions.columns:
        raise ValueError("Predictions is missing 'employee_id' column.")

    if target["employee_id"].isna().any():
        raise ValueError("Target contains missing employee_ids.")

    if predictions["employee_id"].isna().any():
        raise ValueError("Predictions contains missing employee_ids.")

    if target["employee_id"].duplicated().any():
        raise ValueError(
            "Monthly dataset contains duplicate employee_ids in the test month."
        )

    if predictions["employee_id"].duplicated().any():
        raise ValueError(
            "Model predictions contain duplicate employee_ids in the test month."
        )

    normalized_pairs = []
    for pair in prediction_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                "Each element in prediction_pairs must be a pair "
                "(N44 prediction column, N70 prediction column)."
            )

        n44_column, n70_column = pair
        if not all(
            isinstance(column, str) and column.strip()
            for column in (n44_column, n70_column)
        ):
            raise ValueError("Column names in prediction_pairs are invalid.")

        if n44_column == n70_column:
            raise ValueError("The two columns in each prediction pair must be different.")

        normalized_pairs.append((n44_column, n70_column))

    prediction_columns = [
        column
        for pair in normalized_pairs
        for column in pair
    ]

    if len(set(prediction_columns)) != len(prediction_columns):
        raise ValueError("prediction_pairs contains duplicate column names.")

    missing_prediction_columns = set(prediction_columns) - set(predictions.columns)
    if missing_prediction_columns:
        raise ValueError(
            f"Predictions are missing columns: {sorted(missing_prediction_columns)}."
        )

    target_ids = set(target["employee_id"])
    prediction_ids = set(predictions["employee_id"])

    unexpected_ids = prediction_ids - target_ids
    if unexpected_ids:
        raise ValueError(
            "Model predictions contain employees outside the test month cohort: "
            f"{sorted(unexpected_ids, key=str)[:20]}."
        )

    predictions = predictions[["employee_id", *prediction_columns]].copy()

    for column in prediction_columns:
        predictions[column] = pd.to_numeric(
            predictions[column], errors="coerce"
        )

    predictions[prediction_columns] = predictions[
        prediction_columns
    ].replace([np.inf, -np.inf], np.nan)


    incomplete_rows = predictions[prediction_columns].isna().any(axis=1)
    predictions = predictions.loc[~incomplete_rows].copy()

    prediction_ids = set(predictions["employee_id"])
    missing = target.loc[
        ~target["employee_id"].isin(prediction_ids)
    ].copy()

    if not missing.empty:
        fallback = department_mean_fallback(history, missing)

        fallback_predictions = pd.DataFrame({
            "employee_id": fallback["employee_id"].to_numpy()
        })

        for n44_column, n70_column in normalized_pairs:
            fallback_predictions[n44_column] = fallback["N44"].to_numpy()
            fallback_predictions[n70_column] = fallback["N70"].to_numpy()

        predictions = pd.concat(
            [predictions, fallback_predictions],
            ignore_index=True,
        )

    completed = target.merge(
        predictions,
        on="employee_id",
        how="left",
        validate="one_to_one",
    )

    if len(completed) != len(target):
        raise ValueError(
            "Row count after completing the cohort does not match target."
        )

    if completed[prediction_columns].isna().any().any():
        raise ValueError(
            "Predictions still contain missing values after applying fallback."
        )

    values = completed[prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "Predictions contain NaN or infinite values after fallback."
        )

    return completed