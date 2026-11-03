import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

spec = importlib.util.spec_from_file_location(
    "time_series_models", ROOT / "src" / "models" / "time_series_models.py"
)
time_series = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_series)
comparison_spec = importlib.util.spec_from_file_location(
    "compare_monthly_budget_models",
    ROOT / "src" / "models" / "compare_monthly_budget_models.py",
)
comparison = importlib.util.module_from_spec(comparison_spec)
comparison_spec.loader.exec_module(comparison)


class TimeSeriesModelsTests(unittest.TestCase):
    def test_daily_aggregation_and_complete_months(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dates = pd.date_range("2026-01-01", "2026-03-31", freq="D")
            grid = pd.DataFrame(
                {
                    "date": dates,
                    "n_viet": np.ones(len(dates)),
                    "n_chay": np.zeros(len(dates)),
                    "n_au": np.full(len(dates), 2.0),
                }
            )
            monthly = pd.DataFrame(
                {
                    "month": pd.PeriodIndex(["2026-01", "2026-02", "2026-03"], freq="M"),
                    "is_complete_month": [True, True, True],
                }
            )
            grid_path = tmp / "grid.parquet"
            monthly_path = tmp / "monthly.parquet"
            grid.to_parquet(grid_path, index=False)
            monthly.to_parquet(monthly_path, index=False)

            daily, test_months = time_series.load_daily_demand(
                grid_path, monthly_path
            )

            self.assertEqual(test_months, list(pd.period_range("2026-01", "2026-03", freq="M")))
            self.assertEqual(len(daily), len(dates))
            self.assertEqual(daily.loc["2026-01-01", "N44"], 1)
            self.assertEqual(daily.loc["2026-01-01", "N70"], 2)

    def test_mlp_recursively_forecasts_nonnegative_meal_counts(self):
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        history = pd.DataFrame(
            {
                "N44": 10 + np.sin(np.arange(90) * 2 * np.pi / 7),
                "N70": 2 + np.cos(np.arange(90) * 2 * np.pi / 7),
            },
            index=dates,
        )
        future = pd.date_range("2026-04-01", periods=5, freq="D")

        prediction = time_series.forecast_mlp(history, future)

        self.assertEqual(prediction.shape, (5, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())

    def test_arima_uses_documented_order_and_returns_two_targets(self):
        dates = pd.date_range("2026-01-01", periods=45, freq="D")
        history = pd.DataFrame(
            {
                "N44": 10 + np.arange(45) % 4,
                "N70": 2 + np.arange(45) % 2,
            },
            index=dates,
        )
        future = pd.date_range("2026-02-15", periods=3, freq="D")

        prediction = time_series.forecast_arima(history, future)

        self.assertEqual(time_series.ARIMA_ORDER, (6, 0, 1))
        self.assertEqual(prediction.shape, (3, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())

    def test_prophet_calls_fit_and_predict_for_both_meal_types(self):
        calls = {"fit": 0, "predict": 0}

        class FakeProphet:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, frame):
                calls["fit"] += 1
                self.train = frame
                return self

            def predict(self, frame):
                calls["predict"] += 1
                return pd.DataFrame({"yhat": np.full(len(frame), 3.0)})

        fake_module = types.ModuleType("prophet")
        fake_module.Prophet = FakeProphet
        dates = pd.date_range("2026-01-01", periods=35, freq="D")
        history = pd.DataFrame({"N44": 5.0, "N70": 1.0}, index=dates)
        future = pd.date_range("2026-02-05", periods=4, freq="D")

        with patch.dict(sys.modules, {"prophet": fake_module}), patch.object(
            time_series, "_holiday_frame", return_value=pd.DataFrame()
        ):
            prediction = time_series.forecast_prophet(history, future)

        self.assertEqual(calls, {"fit": 2, "predict": 2})
        np.testing.assert_array_equal(prediction, np.full((4, 2), 3.0))

    def test_metrics_report_mape_only_for_positive_actual_days(self):
        actual = np.array([[0.0, 0.0], [10.0, 2.0]])
        predicted = np.array([[4.0, 1.0], [8.0, 1.0]])

        metrics = time_series.calculate_daily_metrics(actual, predicted)

        self.assertEqual(metrics["MAPE_N44_positive_%"], 20.0)
        self.assertEqual(metrics["MAPE_N70_positive_%"], 50.0)
        self.assertTrue(np.isfinite(metrics["WAPE_Cost_%"]))

    def test_same_split_budget_metrics_use_common_months(self):
        actual = pd.DataFrame(
            {
                "month": pd.PeriodIndex(["2026-06", "2026-07"], freq="M"),
                "N44": [100.0, 120.0],
                "N70": [10.0, 12.0],
            }
        )
        predicted = pd.DataFrame(
            {
                "month": pd.PeriodIndex(["2026-06", "2026-07"], freq="M"),
                "N44": [100.0, 100.0],
                "N70": [10.0, 10.0],
            }
        )

        metrics = comparison._budget_metrics(actual, predicted)

        expected_error = (20 * 44_000 + 2 * 70_000)
        self.assertEqual(metrics["MAE_Budget"], expected_error / 2)
        self.assertEqual(metrics["WAPE_Budget_%"], 100 * expected_error / (
            (100 * 44_000 + 10 * 70_000)
            + (120 * 44_000 + 12 * 70_000)
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
