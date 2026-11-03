import pandas as pd


INPUT_PATH = "data/processed/monthly_dataset.parquet"


df = pd.read_parquet(INPUT_PATH)

df["month"] = pd.PeriodIndex(df["month"], freq="M")


# ============================================================
# 1. SỐ THÁNG DỮ LIỆU CỦA MỖI EMPLOYEE
# ============================================================

history = (
    df.groupby("employee_id")["month"]
    .nunique()
    .reset_index(name="n_months")
)


print("\n=== SỐ THÁNG LỊCH SỬ / EMPLOYEE ===")

print(
    history["n_months"]
    .describe()
)


print("\n=== PHÂN BỐ SỐ THÁNG ===")

distribution = (
    history["n_months"]
    .value_counts()
    .sort_index()
)

print(distribution)


# ============================================================
# 2. NHÓM COLD-START
# ============================================================

print("\n=== COLD START ===")

for threshold in [1, 2, 3, 4, 6, 12]:

    n_employee = (
        history["n_months"] < threshold
    ).sum()

    percentage = (
        n_employee
        / len(history)
        * 100
    )

    print(
        f"< {threshold:2d} tháng: "
        f"{n_employee:4d} employees "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 3. KIỂM TRA THÁNG BỊ THIẾU
# ============================================================

def count_missing_months(group):

    months = sorted(group["month"])

    if len(months) <= 1:
        return 0

    expected = pd.period_range(
        start=months[0],
        end=months[-1],
        freq="M"
    )

    return len(expected) - len(months)


missing = (
    df.groupby("employee_id")
    .apply(count_missing_months)
    .reset_index(name="missing_months")
)


print("\n=== THÁNG BỊ THIẾU GIỮA FIRST/LAST MONTH ===")

print(
    missing["missing_months"]
    .describe()
)


print(
    "\nEmployees có ít nhất 1 tháng bị thiếu:",
    (missing["missing_months"] > 0).sum()
)


# ============================================================
# 4. KIỂM TRA 3 THÁNG LỊCH SỬ LIÊN TIẾP
# ============================================================

history["has_3_months"] = (
    history["n_months"] >= 3
)

history["has_6_months"] = (
    history["n_months"] >= 6
)

history["has_12_months"] = (
    history["n_months"] >= 12
)


print("\n=== ĐỦ LỊCH SỬ ===")

print(
    ">= 3 tháng:",
    history["has_3_months"].sum(),
    f"({history['has_3_months'].mean() * 100:.2f}%)"
)

print(
    ">= 6 tháng:",
    history["has_6_months"].sum(),
    f"({history['has_6_months'].mean() * 100:.2f}%)"
)

print(
    ">= 12 tháng:",
    history["has_12_months"].sum(),
    f"({history['has_12_months'].mean() * 100:.2f}%)"
)


# ============================================================
# 5. MIN / MAX MONTH
# ============================================================

print("\n=== THỜI GIAN CỦA EMPLOYEE ===")

employee_period = (
    df.groupby("employee_id")["month"]
    .agg(
        first_month="min",
        last_month="max"
    )
    .reset_index()
)

employee_period["span_months"] = (
    employee_period["last_month"].astype("int64")
    - employee_period["first_month"].astype("int64")
    + 1
)

print(
    employee_period["span_months"]
    .describe()
)