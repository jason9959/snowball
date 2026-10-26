import unittest
from unittest.mock import patch

import pandas as pd

from snowball.core.engine import MomentumConfig, _apply_canary_overlay


class CanaryRoundingTests(unittest.TestCase):
    def _config(self, digits):
        return {
            "attack_mode": "상대 모멘텀",
            "attack_relative": MomentumConfig(),
            "risk_avoidance": {
                "enabled": True,
                "tickers": ["TIP"],
                "apply_mode": "방어형",
                "direction": "마이너스(-)",
                "same_momentum": False,
                "custom_momentum": MomentumConfig("이동평균", 11),
                "decision_percent_digits": digits,
            },
        }

    @patch("snowball.core.engine.momentum_detail")
    def test_second_decimal_rounding_keeps_raw_value_but_uses_zero_for_sign(self, detail):
        detail.return_value = {"계산값": -0.000160015, "기준 기간": "11개월"}
        prices = pd.DataFrame(
            {"TIP": [100.0]}, index=pd.to_datetime(["2016-11-30"])
        )
        adjusted, moved, summary = _apply_canary_overlay(
            prices, pd.Timestamp("2016-11-30"), {"QQQ": 1.0}, self._config(1)
        )
        row = summary["audit"].iloc[0]
        self.assertAlmostEqual(row["모멘텀 값"], -0.000160015)
        self.assertEqual(row["판정값"], 0.0)
        self.assertFalse(row["위험 판정"])
        self.assertEqual(adjusted, {"QQQ": 1.0})
        self.assertEqual(moved, 0.0)

    @patch("snowball.core.engine.momentum_detail")
    def test_none_restores_raw_value_decision(self, detail):
        detail.return_value = {"계산값": -0.000160015, "기준 기간": "11개월"}
        prices = pd.DataFrame(
            {"TIP": [100.0]}, index=pd.to_datetime(["2016-11-30"])
        )
        adjusted, moved, summary = _apply_canary_overlay(
            prices, pd.Timestamp("2016-11-30"), {"QQQ": 1.0}, self._config(None)
        )
        row = summary["audit"].iloc[0]
        self.assertAlmostEqual(row["판정값"], -0.000160015)
        self.assertTrue(row["위험 판정"])
        self.assertEqual(adjusted, {"QQQ": 0.0})
        self.assertEqual(moved, 1.0)


if __name__ == "__main__":
    unittest.main()
