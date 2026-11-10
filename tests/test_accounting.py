from __future__ import annotations

import unittest

import pandas as pd

from snowball.accounting import (
    account_period_metrics,
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

    def test_manual_sheet_row_accepts_blank_status_and_derives_trade_amount(self):
        record = {
            "거래ID": "", "입력시각": "", "거래일": "2026-11-01",
            "유형": "매수", "시장": "한국", "종목코드": "069500",
            "종목명": "KODEX 200", "수량": 3, "체결단가": 10_000,
            "수수료·세금": 0, "금액": "", "통화": "KRW", "환율": 1,
            "메모": "", "상태": "", "수정시각": "",
        }
        frame = transaction_frame([record])
        self.assertEqual(len(frame), 1)
        self.assertEqual(float(frame.iloc[0]["금액"]), 30_000)

    def test_external_deposit_does_not_create_investment_return(self):
        records = [
            {
                "거래ID": identifier, "입력시각": f"2026-11-0{day} 09:00:00",
                "거래일": f"2026-11-0{day}", "유형": "입금", "시장": "현금",
                "종목코드": "현금", "종목명": "", "수량": 0, "체결단가": 0,
                "수수료·세금": 0, "금액": amount, "통화": "KRW", "환율": 1,
                "메모": "", "상태": "정상", "수정시각": "",
            }
            for identifier, day, amount in (("first", 1, 1_000_000), ("second", 2, 2_000_000))
        ]
        valuation = calculate_account_valuation(
            transaction_frame(records), end_date=pd.Timestamp("2026-11-02"),
        )
        metrics = account_period_metrics(
            valuation.history, pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-02"),
        )
        self.assertAlmostEqual(metrics.period_return or 0, 0.0)
        self.assertAlmostEqual(metrics.max_drawdown or 0, 0.0)

    def test_period_metrics_rebase_and_measure_drawdown(self):
        history = pd.DataFrame(
            {
                "총평가액": [100.0, 110.0, 110.0, 88.0],
                "입출금": [100.0, 0.0, 0.0, 0.0],
                "일수익률": [0.0, 0.10, 0.0, -0.20],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="D"),
        )
        metrics = account_period_metrics(
            history, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-04"),
        )
        self.assertAlmostEqual(metrics.period_return or 0, -0.12)
        self.assertAlmostEqual(metrics.max_drawdown or 0, -0.20)
        self.assertAlmostEqual(metrics.current_drawdown or 0, -0.20)

    def test_currency_asset_history_tracks_cost_and_native_market_value(self):
        records = [
            {
                "입력날짜": "2026-11-01", "입력시각": "09:00:00",
                "거래일": "2026-11-01", "유형": "매수", "시장": "한국",
                "종목코드": "069500", "종목명": "KODEX 200", "수량": 10,
                "체결단가": 10_000, "거래금액": 100_000, "수수료·세금": 100,
                "결제금액": 100_100, "통화": "KRW", "환율": 1, "메모": "",
            },
            {
                "입력날짜": "2026-11-01", "입력시각": "23:00:00",
                "거래일": "2026-11-01", "유형": "매수", "시장": "미국",
                "종목코드": "QQQ", "종목명": "Invesco QQQ", "수량": 2,
                "체결단가": 500, "거래금액": 1_000, "수수료·세금": 1,
                "결제금액": 1_001, "통화": "USD", "환율": 1_350, "메모": "",
            },
        ]
        closes = pd.DataFrame(
            {"069500.KS": [10_000.0, 11_000.0], "QQQ": [500.0, 525.0]},
            index=pd.to_datetime(["2026-11-01", "2026-11-02"]),
        )
        valuation = calculate_account_valuation(
            transaction_frame(records), closes, pd.Series(
                [1350.0, 1360.0], index=closes.index,
            ), end_date=pd.Timestamp("2026-11-02"),
        )
        krw = valuation.currency_assets["KRW"].iloc[-1]
        usd = valuation.currency_assets["USD"].iloc[-1]
        self.assertAlmostEqual(float(krw["매입원가"]), 100_100)
        self.assertAlmostEqual(float(krw["시장평가액"]), 110_000)
        self.assertAlmostEqual(float(usd["매입원가"]), 1_001)
        self.assertAlmostEqual(float(usd["시장평가액"]), 1_050)


if __name__ == "__main__":
    unittest.main()
