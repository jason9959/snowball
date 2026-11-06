from __future__ import annotations

import unittest

import pandas as pd

from snowball.rebalance_compare import build_rebalance_comparison, parse_target_selection


class RebalanceComparisonTest(unittest.TestCase):
    def test_parses_domestic_mailing_selection(self) -> None:
        parsed = parse_target_selection(
            "329750 · TIGER 미국달러단기채권액티브 50.0%, "
            "130680 · TIGER 원유선물Enhanced(H) 50.0%"
        )
        self.assertEqual(set(parsed), {"329750", "130680"})
        self.assertEqual(parsed["329750"][1], 0.5)

    def test_combines_duplicate_target_codes(self) -> None:
        parsed = parse_target_selection("329750 · 달러채권 25.0%, 329750 · 달러채권 75.0%")
        self.assertEqual(parsed["329750"][1], 1.0)

    def test_comparison_includes_buy_sell_and_cash(self) -> None:
        holdings = pd.DataFrame([
            {"종목코드": "329750", "종목명": "달러채권", "평가금액": 600_000},
            {"종목코드": "999999", "종목명": "기존종목", "평가금액": 300_000},
        ])
        result = build_rebalance_comparison(
            holdings, cash=100_000, total_value=1_000_000,
            target_selection="329750 · 달러채권 50.0%, 130680 · 원자재 50.0%",
        ).set_index("종목코드")
        self.assertEqual(result.loc["329750", "조정"], "매도")
        self.assertEqual(result.loc["130680", "조정"], "매수")
        self.assertEqual(result.loc["999999", "목표 비중"], 0.0)
        self.assertEqual(result.loc["현금", "조정"], "매도")
        self.assertAlmostEqual(result["차이 금액"].sum(), 0.0)

    def test_rejects_incomplete_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "100%"):
            parse_target_selection("QQQ 50.0%")


if __name__ == "__main__":
    unittest.main()
