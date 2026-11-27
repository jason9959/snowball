import datetime as dt
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

if "yfinance" not in sys.modules:
    try:
        __import__("yfinance")
    except ModuleNotFoundError:
        sys.modules["yfinance"] = types.SimpleNamespace(download=None)

from common.functions.analytics import allocation_sweep, comparison, monte_carlo, periodic, static_portfolio


class FunctionsAnalyticsTests(unittest.TestCase):
    def setUp(self):
        index = pd.bdate_range("2020-01-02", "2024-12-31")
        self.frame = pd.DataFrame(
            {
                "SPY": 100 * np.exp(np.arange(len(index)) * 0.0003),
                "IEF": 100 * np.exp(np.arange(len(index)) * 0.00008),
                "GLD": 100 * np.exp(np.arange(len(index)) * 0.00015),
            },
            index=index,
        )
        self.start = dt.date(2020, 1, 2)
        self.end = dt.date(2024, 12, 31)

    def _prices(self, tickers, start, end, field="Close"):
        return self.frame[list(tickers)].copy()

    def test_comparison_and_periodic_keep_original_layout_data(self):
        with patch("common.functions.analytics.prices", side_effect=self._prices):
            compared = comparison(["SPY", "IEF"], self.start, self.end)
            periodic_result = periodic("SPY", self.start, self.end, "반기")
        self.assertEqual(list(compared.table["종목"]), ["SPY", "IEF"])
        self.assertEqual(compared.metrics, {})
        self.assertIn("SPY · 시작값 100", periodic_result.series)
        self.assertFalse(periodic_result.table.empty)

    def test_comparison_rejects_partially_missing_tickers(self):
        with patch(
            "common.functions.analytics.prices",
            return_value=self.frame[["SPY"]].copy(),
        ):
            with self.assertRaisesRegex(ValueError, "IEF"):
                comparison(["SPY", "IEF"], self.start, self.end)

    def test_static_portfolio_supports_contributions(self):
        with patch("common.functions.analytics.prices", side_effect=self._prices):
            result = static_portfolio(
                ["SPY", "IEF"], [60, 40], self.start, self.end, 10_000,
                "매월", contribution=100, contribution_frequency="매월",
            )
        self.assertIn("누적 투입금", result.series)
        self.assertGreater(result.series["누적 투입금"].iloc[-1], 10_000)

    def test_static_portfolio_adds_selected_benchmark_and_drawdown(self):
        with patch("common.functions.analytics.prices", side_effect=self._prices):
            result = static_portfolio(
                ["IEF", "GLD"], [60, 40], self.start, self.end, 10_000,
                "매월", contribution=100, contribution_frequency="매월",
                benchmark_ticker="SPY", benchmark_label="S&P 500",
            )
        self.assertIn("벤치마크 · S&P 500", result.series)
        self.assertIn("S&P 500 낙폭", result.series)
        self.assertEqual(result.metrics["벤치마크"], "S&P 500")
        self.assertIn("벤치마크 MDD", result.metrics)
        self.assertEqual(
            result.series["벤치마크 · S&P 500"].index.tolist(),
            result.series["포트폴리오"].index.tolist(),
        )

    def test_fixed_allocation_and_monte_carlo_options(self):
        with patch("common.functions.analytics.prices", side_effect=self._prices):
            sweep = allocation_sweep(
                ["SPY", "IEF", "GLD"], self.start, self.end, 5,
                fixed_index=0, fixed_ratio=40,
            )
            simulated = monte_carlo(
                "SPY", self.start, self.end, 2, 100, 10_000, "bootstrap", 100,
            )
        self.assertTrue((sweep.table["SPY 비율"] == 40).all())
        self.assertEqual(len(sweep.table), 13)
        self.assertIn("총 예정 투입금", simulated.metrics)


if __name__ == "__main__":
    unittest.main()
