from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from snowball.account_dashboard import account_date_bounds, build_account_dashboard


def _record(identifier: str, day: str, kind: str, amount: float) -> dict[str, object]:
    return {
        "거래ID": identifier, "입력시각": f"{day} 09:00:00", "거래일": day,
        "유형": kind, "시장": "현금", "종목코드": "현금", "종목명": "",
        "수량": 0, "체결단가": 0, "수수료·세금": 0, "금액": amount,
        "통화": "KRW", "환율": 1, "메모": "", "상태": "정상", "수정시각": "",
    }


class AccountDashboardTest(unittest.TestCase):
    def test_bounds_use_first_active_transaction(self) -> None:
        records = [
            {**_record("cancel", "2026-10-01", "입금", 10), "상태": "취소"},
            _record("active", "2026-11-01", "입금", 100),
        ]
        self.assertEqual(
            account_date_bounds(records, today=date(2026, 11, 6)),
            (date(2026, 11, 1), date(2026, 11, 6)),
        )

    def test_dashboard_summarizes_flows_and_ledger(self) -> None:
        records = [
            _record("deposit", "2026-11-01", "입금", 1_000_000),
            _record("withdraw", "2026-11-03", "출금", 100_000),
        ]
        dashboard = build_account_dashboard(
            records, start_date=date(2026, 11, 2), end_date=date(2026, 11, 4),
        )
        self.assertEqual(dashboard.cumulative_deposits, 1_000_000)
        self.assertEqual(dashboard.cumulative_withdrawals, 100_000)
        self.assertEqual(dashboard.period_deposits, 0)
        self.assertEqual(dashboard.period_withdrawals, 100_000)
        self.assertEqual(dashboard.ledger.iloc[0]["유형"], "출금")
        self.assertTrue(dashboard.performance_valid)
        self.assertAlmostEqual(dashboard.valuation.total_value, 900_000)

    def test_dashboard_rejects_reversed_period(self) -> None:
        with self.assertRaisesRegex(ValueError, "시작일"):
            build_account_dashboard(
                [_record("deposit", "2026-11-01", "입금", 1_000)],
                start_date=date(2026, 11, 3), end_date=date(2026, 11, 2),
            )


if __name__ == "__main__":
    unittest.main()
