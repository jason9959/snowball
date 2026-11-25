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

    def test_limited_cash_prioritizes_the_largest_decline_without_partial_scaling(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=2)
        prices = {
            "AAA": pd.Series([100, 90], index=dates),
            "BBB": pd.Series([100, 80], index=dates),
        }
        rules = [
            AssetRule("AAA", 100, 2, .10, .10, 200),
            AssetRule("BBB", 100, 2, .20, .20, 200),
        ]
        result = run_multi_split_backtest(
            prices, rules, MultiSplitConfig(300, "limited", "fractional_6"),
        )
        second_buys = result["trades"][
            (result["trades"]["매수/매도"] == "매수")
            & (result["trades"]["차수"] == 2)
        ]
        self.assertEqual(second_buys["종목"].tolist(), ["BBB"])
        self.assertAlmostEqual(float(second_buys.iloc[0]["체결금액"]), 100.0, places=4)
        self.assertEqual(result["pending_orders"], 1)

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

    def test_missed_step_uses_dynamic_recovery_base_then_restores_original_ladder(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=10)
        prices = {
            "AAA": pd.Series(
                [45, 36, 28.8, 23.04, 20, 22, 17.6, 21.12, 23.04, 18.432],
                index=dates,
            ),
            "BBB": pd.Series(
                [100, 90, 81, 81, 99, 99, 99, 99, 99, 99],
                index=dates,
            ),
        }
        rules = [
            AssetRule("AAA", 100, 6, .20, .20, 600),
            AssetRule("BBB", 100, 3, .10, .10, 300),
        ]

        result = run_multi_split_backtest(
            prices, rules, MultiSplitConfig(600, "limited", "fractional_6"),
        )
        aaa = result["trades"][result["trades"]["종목"] == "AAA"]
        buys = aaa[aaa["매수/매도"] == "매수"]["시뮬레이션 체결가"].tolist()
        sells = aaa[aaa["매수/매도"] == "매도"]["시뮬레이션 체결가"].tolist()

        # $23.04 was missed without cash and later filled at the actual $20.
        self.assertIn(20.0, buys)
        # The temporary first position follows the $22 high: 22×80%=17.60.
        self.assertIn(17.6, buys)
        # Its child exits at 17.60×120%=21.12, before restoration at $23.04.
        self.assertIn(21.12, sells)
        # Once restored, the original ladder resumes: 23.04×80%=18.432.
        self.assertTrue(any(abs(value - 18.432) < 1e-9 for value in buys))
        self.assertEqual(result["recovery_started"], 1)
        self.assertEqual(result["recovery_completed"], 1)
        self.assertEqual(result["active_recoveries"], 0)


if __name__ == "__main__":
    unittest.main()
