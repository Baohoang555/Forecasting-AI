"""
Test cho các thay đổi pipeline (lịch nghỉ lễ từ file, khôi phục suất thiếu log điểm danh,
bỏ giai đoạn khởi động, lọc theo HR, bỏ trần working_days, kiểm tra tháng đầu).

Chạy từ thư mục gốc dự án:
    python tests/test_calendar_and_cleaning.py
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import calendar_utils as cal  # noqa: E402
import data_cleaning as dc  # noqa: E402
import monthly_aggregation as ma  # noqa: E402
import predict_next_month as pnm  # noqa: E402


def _quiet(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


class CalendarTests(unittest.TestCase):
    def test_parse_period_formats(self):
        d = lambda s: pd.Timestamp(s)  # noqa: E731
        self.assertEqual(cal.parse_period("30/08/2025 - 02/09/2025"), (d("2025-08-30"), d("2025-09-02")))
        self.assertEqual(cal.parse_period("24/11/2026"), (d("2026-11-24"), d("2026-11-24")))
        self.assertEqual(cal.parse_period("04/02/2027 \u2013 10/02/2027"), (d("2027-02-04"), d("2027-02-10")))
        self.assertEqual(cal.parse_period(pd.Timestamp("2026-01-01")), (d("2026-01-01"), d("2026-01-01")))
        with self.assertRaises(ValueError):
            cal.parse_period("10/02/2027 - 04/02/2027")
        with self.assertRaises(ValueError):
            cal.parse_period("không phải ngày")

    def _write_holiday_xlsx(self, tmp, rows):
        path = Path(tmp) / "relax.xlsx"
        pd.DataFrame(rows, columns=["Năm", "Dịp lễ/Tết", "Thời gian nghỉ dự kiến", "Số ngày nghỉ"]).to_excel(path, index=False)
        return path

    def test_load_holiday_table_expands_ranges_and_warns_on_day_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_holiday_xlsx(tmp, [
                [2026, "Lễ A", "01/05/2026 - 03/05/2026", "3 ngày"],
                [2026, "Lễ B", "10/05/2026 - 11/05/2026", "5 ngày"],  # ghi sai số ngày
                [2026, "Lễ C", pd.Timestamp("2026-06-01"), "1 ngày"],
            ])
            table, log = _quiet(cal.load_holiday_table, path)
            self.assertEqual(len(table), 3 + 2 + 1)
            self.assertIn("Lễ B", log)  # cảnh báo lệch số ngày
            self.assertNotIn("Lễ A:", log)

    def test_fallback_to_legacy_list_when_file_missing(self):
        old = os.environ.get(cal.ENV_HOLIDAY_FILE)
        os.environ[cal.ENV_HOLIDAY_FILE] = "/khong/ton/tai.xlsx"
        try:
            holidays, log = _quiet(cal.load_public_holidays)
        finally:
            if old is None:
                os.environ.pop(cal.ENV_HOLIDAY_FILE)
            else:
                os.environ[cal.ENV_HOLIDAY_FILE] = old
        self.assertEqual(len(holidays), len(cal.LEGACY_PUBLIC_HOLIDAYS))
        self.assertIn("DANH SÁCH LỄ CŨ", log)

    def test_company_days_off_ignores_comments_and_expands_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "company.csv"
            path.write_text(
                "# ghi chú\nfrom_date,to_date,name\n# 01/01/2026,,bỏ qua\n09/02/2026,13/02/2026,Trước Tết\n20/03/2026,,Một ngày\n",
                encoding="utf-8",
            )
            table = cal.load_company_days_off(path)
            self.assertEqual(len(table), 5 + 1)
            self.assertEqual(table["date"].min(), pd.Timestamp("2026-02-09"))

    @unittest.skipUnless(
        (ROOT / "data" / "relax_day.xlsx").exists(),
        "requires the private data/relax_day.xlsx calendar",
    )
    def test_shipped_calendar_loads_and_covers_2027(self):
        holidays = cal.load_public_holidays()
        self.assertGreaterEqual(cal.max_covered_year(holidays), 2027)
        self.assertIn(pd.Timestamp("2025-09-01"), holidays)  # thiếu trong danh sách cũ
        self.assertIn(pd.Timestamp("2026-11-24"), holidays)


class CleaningTests(unittest.TestCase):
    @staticmethod
    def _raw(rows):
        base = dict(department="D1", bonus_meal_raw=np.nan, shift_raw="Ca 1 (Shift 1)", cumulative_raw=np.nan)
        recs = []
        for r in rows:
            rec = dict(base)
            rec.update(r)
            recs.append(rec)
        return pd.DataFrame(recs)

    def test_attendance_sync_rows_are_reinstated_but_other_invalid_rows_are_not(self):
        meal = "Món Việt (Vietnamese Meal)"
        raw = self._raw([
            dict(employee_id=1, checkin_time="9/10/2025 11:00", meal_type_raw=meal, amount=44000, status="Hợp lệ", reason=np.nan),
            dict(employee_id=2, checkin_time="9/10/2025 11:01", meal_type_raw=meal, amount=44000,
                 status="Không hợp lệ", reason="Ch?a có log ?i?m danh"),
            dict(employee_id=3, checkin_time="9/10/2025 11:02", meal_type_raw=meal, amount=44000,
                 status="Không hợp lệ", reason="Chưa có log điểm danh"),
            dict(employee_id=4, checkin_time="9/10/2025 11:03", meal_type_raw=meal, amount=44000,
                 status="Không hợp lệ", reason="Quá s? l?n ?n ??ng kư trong ngày"),
        ])
        (clean, removed), _ = _quiet(dc.clean_log, raw)
        self.assertEqual(sorted(clean["employee_id"].tolist()), [1, 2, 3])
        self.assertEqual(int(clean["reinstated_attendance_sync"].sum()), 2)
        self.assertEqual(removed["drop_reason"].tolist(), ["trang_thai_khong_hop_le"])
        self.assertEqual(int(removed.iloc[0]["employee_id"]), 4)

    def test_reinstated_retry_swipes_count_once_and_not_when_a_valid_meal_exists(self):
        meal = "Món Việt (Vietnamese Meal)"
        no_log = dict(status="Không hợp lệ", reason="Chưa có log điểm danh", meal_type_raw=meal, amount=44000)
        raw = self._raw([
            # NV 1: quẹt lại 3 lần cách nhau 1 phút -> chỉ 1 suất
            dict(employee_id=1, checkin_time="9/10/2025 09:32", **no_log),
            dict(employee_id=1, checkin_time="9/10/2025 09:33", **no_log),
            dict(employee_id=1, checkin_time="9/10/2025 09:34", **no_log),
            # NV 2: 1 lần thiếu log + 1 suất hợp lệ cùng ngày -> chỉ giữ suất hợp lệ
            dict(employee_id=2, checkin_time="9/10/2025 09:32", **no_log),
            dict(employee_id=2, checkin_time="9/10/2025 12:00", meal_type_raw=meal, amount=44000, status="Hợp lệ", reason=np.nan),
        ])
        (clean, removed), _ = _quiet(dc.clean_log, raw)
        self.assertEqual(clean.groupby("employee_id").size().to_dict(), {1: 1, 2: 1})
        self.assertEqual(clean[clean.employee_id == 1]["checkin_time"].iloc[0], pd.Timestamp("2025-09-10 09:32"))
        self.assertFalse(clean[clean.employee_id == 2]["reinstated_attendance_sync"].iloc[0])
        self.assertEqual(removed["drop_reason"].value_counts().to_dict(), {"quet_lai_khi_thieu_log_diem_danh": 3})

    def test_reinstated_row_without_meal_or_amount_is_still_dropped(self):
        raw = self._raw([
            dict(employee_id=5, checkin_time="9/10/2025 11:00", meal_type_raw=np.nan, amount=np.nan,
                 status="Không hợp lệ", reason="Chưa có log điểm danh"),
        ])
        (clean, removed), _ = _quiet(dc.clean_log, raw)
        self.assertEqual(len(clean), 0)
        self.assertIn("khong_nhan_dien_duoc_loai_mon", removed["drop_reason"].tolist())

    def test_warmup_period_removed_and_start_date_can_be_disabled(self):
        meal = "Món Việt (Vietnamese Meal)"
        raw = self._raw([
            dict(employee_id=1, checkin_time="7/21/2025 11:00", meal_type_raw=meal, amount=44000, status="Hợp lệ", reason=np.nan),
            dict(employee_id=1, checkin_time="8/1/2025 11:00", meal_type_raw=meal, amount=44000, status="Hợp lệ", reason=np.nan),
        ])
        (clean, removed), _ = _quiet(dc.clean_log, raw)  # mặc định bỏ trước 2025-08-01
        self.assertEqual(len(clean), 1)
        self.assertEqual(removed["drop_reason"].tolist(), ["giai_doan_khoi_dong_he_thong"])
        (clean_all, _), _ = _quiet(dc.clean_log, raw, start_date=None)
        self.assertEqual(len(clean_all), 2)


class MonthlyEdgeTests(unittest.TestCase):
    @staticmethod
    def _grid(start, end):
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame({
            "employee_id": 1, "department": "D", "date": dates,
            "n_viet": 1, "n_chay": 0, "n_au": 0, "n_meals": 1, "total_amount": 44000.0, "has_checkin": 1,
        })

    def test_partial_first_month_warns_and_can_be_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            grid_path, out_path = Path(tmp) / "g.parquet", Path(tmp) / "m.parquet"
            self._grid("2025-07-21", "2025-10-31").to_parquet(grid_path)
            _, log = _quiet(ma.create_monthly_dataset, grid_path, out_path)
            self.assertIn("Tháng đầu 2025-07", log)
            monthly, _ = _quiet(ma.create_monthly_dataset, grid_path, out_path, "2025-08")
            self.assertEqual(str(monthly["month"].min()), "2025-08")

    def test_full_first_month_has_no_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            grid_path, out_path = Path(tmp) / "g.parquet", Path(tmp) / "m.parquet"
            self._grid("2025-08-01", "2025-10-31").to_parquet(grid_path)
            _, log = _quiet(ma.create_monthly_dataset, grid_path, out_path)
            self.assertNotIn("Tháng đầu", log)


class ForecastTests(unittest.TestCase):
    def test_meal_cap_is_calendar_days_not_weekdays(self):
        self.assertEqual(pnm.max_meals_in_month(pd.Period("2026-09", freq="M")), 30)
        self.assertEqual(pnm.max_meals_in_month(pd.Period("2026-10", freq="M")), 31)
        # 24 suất (nhân viên đi cả thứ Bảy) không bị ép về 20 như trần cũ
        n44, n70 = pnm.cap_and_round_meals([24.2], [0.0], [pnm.max_meals_in_month(pd.Period("2026-09", freq="M"))])
        self.assertEqual((int(n44[0]), int(n70[0])), (24, 0))

    def test_hr_roster_split_and_safety_guard(self):
        forecast = pd.DataFrame({"employee_id": [1, 2, 3, 4], "pred_cost": [1.0, 2.0, 3.0, 4.0]})
        with tempfile.TemporaryDirectory() as tmp:
            hr_path = Path(tmp) / "hr.xlsx"
            pd.DataFrame({"Mã NV": [1, 2, 3]}).to_excel(hr_path, index=False)
            kept, dropped = pnm.split_by_hr_roster(forecast, hr_path, max_excluded_share=0.5)
            self.assertEqual(kept["employee_id"].tolist(), [1, 2, 3])
            self.assertEqual(dropped["employee_id"].tolist(), [4])
            with self.assertRaisesRegex(ValueError, "vượt ngưỡng"):
                pnm.split_by_hr_roster(forecast, hr_path, max_excluded_share=0.1)

    def test_holiday_coverage_fails_when_calendar_expired(self):
        original = pnm.PUBLIC_HOLIDAYS
        try:
            pnm.PUBLIC_HOLIDAYS = pd.DatetimeIndex(["2027-01-01", "2027-02-04"])
            with self.assertRaisesRegex(ValueError, "chưa có năm 2028"):
                pnm.check_holiday_coverage(pd.Period("2028-01", freq="M"))
            _quiet(pnm.check_holiday_coverage, pd.Period("2027-10", freq="M"))  # không lỗi
        finally:
            pnm.PUBLIC_HOLIDAYS = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
