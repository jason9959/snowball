from __future__ import annotations

import unittest

import pandas as pd

from snowball.accounting import (
    calculate_account_valuation,
    required_price_symbols,
    transaction_frame,
    yahoo_symbol,
)


class AccountingTest(unittest.TestCase):
    def test_yahoo_symbol_preserves_domestic_leading_zero(self):
        self.assertEqual(yahoo_symbol("69500", "한국"), "069500.KS")
        self.assertEqual(yahoo_symbol("QQQ", "미국"), "QQQ")

    def test_cash_holdings_profit_and_return(self):
        records = [
            {
                "거래ID": "deposit", "입력시각": "2026-11-01 09:00:00",
                "거래일": "2026-11-01", "유형": "입금", "시장": "현금",
                "종목코드": "현금", "종목명": "", "수량": 0, "체결단가": 0,
                "수수료·세금": 0, "금액": 1_000_000, "통화": "KRW", "환율": 1,
                "메모": "", "상태": "정상", "수정시각": "",
            },
            {
                "거래ID": "buy", "입력시각": "2026-11-02 09:00:00",
                "거래일": "2026-11-02", "유형": "매수", "시장": "한국",
                "종목코드": "069500", "종목명": "KODEX 200", "수량": 10,
                "체결단가": 10_000, "수수료·세금": 100, "금액": 100_000,
                "통화": "KRW", "환율": 1, "메모": "", "상태": "정상", "수정시각": "",
            },
        ]
        frame = transaction_frame(records)
        self.assertEqual(required_price_symbols(frame), ("069500.KS",))
        closes = pd.DataFrame(
            {"069500.KS": [10_000.0, 12_000.0]},
            index=pd.to_datetime(["2026-11-02", "2026-11-03"]),
        )
        valuation = calculate_account_valuation(
            frame, closes, end_date=pd.Timestamp("2026-11-03"),
        )
        self.assertAlmostEqual(valuation.cash, 899_900)
        self.assertAlmostEqual(valuation.holdings_value, 120_000)
        self.assertAlmostEqual(valuation.total_value, 1_019_900)
        self.assertAlmostEqual(valuation.profit, 19_900)
        self.assertAlmostEqual(valuation.return_rate or 0, 0.0199)
        self.assertEqual(float(valuation.holdings.iloc[0]["수량"]), 10)

    def test_cancelled_records_are_excluded(self):
        record = {
            "거래ID": "cancel", "입력시각": "2026-11-01 09:00:00",
            "거래일": "2026-11-01", "유형": "입금", "시장": "현금",
            "종목코드": "현금", "종목명": "", "수량": 0, "체결단가": 0,
            "수수료·세금": 0, "금액": 1_000_000, "통화": "KRW", "환율": 1,
            "메모": "", "상태": "취소", "수정시각": "",
        }
        self.assertTrue(transaction_frame([record]).empty)


if __name__ == "__main__":
    unittest.main()
