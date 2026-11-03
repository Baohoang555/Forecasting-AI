import pandas as pd
from data_cleaning import load_raw_log, clean_log

raw = load_raw_log("data/raw/MealHistory_UTF8.csv")
clean, _ = clean_log(raw)

# Xem quy mô từng phòng ban và tỷ lệ chọn Âu - để biết 0%/100% có phải do
# phòng ban quá nhỏ hay không
dept_stats = (
    clean.groupby("department")
    .agg(so_nhan_vien=("employee_id", "nunique"),
         so_bua=("meal_type", "size"),
         ty_le_au=("meal_type", lambda s: (s == "Âu").mean()))
    .sort_values("ty_le_au")
)
print(dept_stats.to_string())

# Xác nhận chắc chắn Chủ nhật = 0 bữa, không phải làm tròn
clean["dow"] = clean["checkin_time"].dt.dayofweek
print("Số bữa ăn hợp lệ vào Chủ nhật:", (clean["dow"] == 6).sum())