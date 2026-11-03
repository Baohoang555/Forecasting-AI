import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class DashboardIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.employee = pd.read_csv(ROOT / "results/forecast/forecast_2026-09_employee.csv")
        cls.department = pd.read_csv(ROOT / "results/forecast/forecast_2026-09_department.csv")

    def test_app_syntax_and_single_streamlit_page_config(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertEqual(source.count("st.set_page_config("), 1)
        canonical_priority = source.index(
            'if (base_dir / f"results/metrics/{file_old}").exists():'
        )
        fallback_priority = source.index(
            'if (base_dir / f"results/metrics/{file_new}").exists():'
        )
        self.assertLess(canonical_priority, fallback_priority)
        for stale_text in ["5.89%", "1,607", "177,627", "10.57%", "259%", "VardCanteen2026"]:
            self.assertNotIn(stale_text, source)

    def test_employee_forecast_has_unique_ids_and_valid_cost_formula(self):
        df = self.employee
        self.assertEqual(len(df), df["employee_id"].nunique())
        self.assertTrue(df["employee_id"].notna().all())
        self.assertTrue((df[["pred_N44", "pred_N70", "pred_cost"]] >= 0).all().all())
        expected_cost = df["pred_N44"] * 44_000 + df["pred_N70"] * 70_000
        self.assertTrue(np.allclose(df["pred_cost"], expected_cost))

    def test_department_forecast_matches_employee_rollup(self):
        emp_rollup = (
            self.employee.groupby("department")
            .agg(so_nhan_vien=("employee_id", "count"), tong_du_bao=("pred_cost", "sum"))
            .sort_index()
        )
        dept = self.department.set_index("department").sort_index()
        self.assertEqual(len(emp_rollup), len(dept))
        pd.testing.assert_series_equal(emp_rollup["so_nhan_vien"], dept["so_nhan_vien"], check_names=False)
        self.assertTrue(np.allclose(emp_rollup["tong_du_bao"], dept["tong_du_bao"]))

    @unittest.skipUnless(
        (ROOT / "data" / "hr.xlsx").exists(),
        "requires the private data/hr.xlsx employee roster",
    )
    def test_hr_department_mapping_uses_data_workbook(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('mapping_path = base_dir / "data" / "hr.xlsx"', source)

        hr = pd.read_excel(ROOT / "data" / "hr.xlsx")
        mapped = self.employee.merge(
            hr[["Mã NV", "Phòng ban"]],
            left_on="employee_id",
            right_on="Mã NV",
            how="left",
        )
        self.assertGreater(mapped["Phòng ban"].notna().sum(), 0)
        self.assertEqual(
            mapped.loc[mapped["employee_id"] == 1003, "Phòng ban"].iloc[0],
            "1000. General",
        )

    def test_leaderboard_has_common_schema_and_dynamic_count(self):
        paths = [
            ("results/metrics/baselines_summary.csv", "Model"),
            ("results/metrics/ml_models_summary.csv", "Model"),
            ("results/metrics/dl_models_summary.csv", "Model"),
            ("results/metrics/ensemble_multi_summary.csv", "Method"),
        ]
        frames = [pd.read_csv(ROOT / path).rename(columns={key: "Model"}) for path, key in paths]
        leaderboard = pd.concat(frames, ignore_index=True)
        self.assertEqual(len(leaderboard), 12)
        self.assertTrue({"Model", "MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70"}.issubset(leaderboard.columns))
        self.assertEqual(leaderboard.sort_values("MAE_Cost").iloc[0]["Model"], "MA 3M")

    def test_daily_time_series_metrics_are_displayed_separately(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('time_series_path = base_dir / "results" / "metrics" / "time_series_models_summary.csv"', source)
        self.assertIn('"time_series": df_time_series', source)
        self.assertIn("Daily company-wide time-series benchmark", source)
        self.assertIn("do not compare directly with the employee-month metrics below", source)

    def test_same_split_budget_comparison_is_loaded_and_displayed(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('"same_split_comparison": df_same_split', source)
        self.assertIn("same_split_budget_model_comparison.csv", source)
        self.assertIn("Fair comparison: same data and test months", source)


if __name__ == "__main__":
    unittest.main()
