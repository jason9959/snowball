import unittest

import pandas as pd

from features.snowball06_multi_split_trading.engine import (
    AssetRule,
    MultiSplitConfig,
    run_multi_split_backtest,
)
from features.snowball06_multi_split_trading.persistence import (
    deserialize_multi_result,
    serialize_multi_result,
)


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-02", periods=len(values)))


class MultiSplitTradingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = {
            "AAA": series([100, 90, 80, 100, 120]),
            "BBB": series([50, 55, 45, 40, 60]),
        }
        self.rules = [
            AssetRule("AAA", 100, 3, .10, .10, 300),
            AssetRule("BBB", 100, 3, .10, .10, 300),
        ]

    def test_limited_mode_shares_one_cash_balance(self) -> None:
        result = run_multi_split_backtest(
            self.prices, self.rules,
            MultiSplitConfig(600, "limited", "fractional_6"),
        )
        self.assertEqual(set(result["asset_summary"]["종목"]), {"AAA", "BBB"})
        self.assertEqual(set(result["trades"]["종목"]), {"AAA", "BBB"})
        self.assertGreaterEqual(result["cash_nav"].min(), -1e-6)
        self.assertEqual(result["initial_capital"], 600)
        self.assertEqual(result["max_slots"], 6)

    def test_required_mode_reverses_portfolio_level_minimum_capital(self) -> None:
        result = run_multi_split_backtest(
            self.prices, self.rules,
            MultiSplitConfig(0, "required", "fractional_6"),
        )
        self.assertGreater(result["minimum_required_capital"], 0)
        self.assertAlmostEqual(
            result["initial_capital"], result["minimum_required_capital"], places=6,
        )
        self.assertGreaterEqual(result["cash_nav"].min(), -1e-6)
        self.assertEqual(result["recommended_capital_20"], result["initial_capital"] * 1.2)

    def test_result_round_trip_keeps_asset_summary_and_shared_nav(self) -> None:
        result = run_multi_split_backtest(
            self.prices, self.rules,
            MultiSplitConfig(600, "limited", "fractional_6"),
        )
        restored = deserialize_multi_result(serialize_multi_result(result))
        pd.testing.assert_series_equal(
            restored["split_nav"], result["split_nav"], check_freq=False,
        )
        self.assertEqual(
            restored["asset_summary"]["종목"].tolist(),
            result["asset_summary"]["종목"].tolist(),
        )

    def test_requires_multiple_unique_assets(self) -> None:
        with self.assertRaisesRegex(ValueError, "2개 이상"):
            run_multi_split_backtest(
                {"AAA": self.prices["AAA"]}, [self.rules[0]],
                MultiSplitConfig(300),
            )
        with self.assertRaisesRegex(ValueError, "중복"):
            run_multi_split_backtest(
                self.prices, [self.rules[0], self.rules[0]],
                MultiSplitConfig(600),
            )


if __name__ == "__main__":
    unittest.main()
