import datetime as dt
import unittest
from pathlib import Path

import pandas as pd

from snowball.mailing import (
    PerformanceSnapshot,
    RebalanceRecommendation,
    first_xnys_session,
    is_month_end_xnys_session,
    load_mailing_portfolios,
    next_xkrx_session,
    next_xnys_session,
    render_email,
)
from snowball.mailing import _domestic_equivalent
from snowball.mailing import _performance_snapshot


class MailingTests(unittest.TestCase):
    def test_first_xnys_session_skips_new_year_holiday(self):
        self.assertEqual(first_xnys_session(dt.date(2026, 1, 8)), dt.date(2026, 1, 2))

    def test_first_xnys_session_regular_month(self):
        self.assertEqual(first_xnys_session(dt.date(2026, 7, 20)), dt.date(2026, 7, 1))

    def test_month_end_close_targets_next_month_first_session(self):
        self.assertTrue(is_month_end_xnys_session(dt.date(2026, 10, 30)))
        self.assertEqual(next_xnys_session(dt.date(2026, 10, 30)), dt.date(2026, 11, 2))
        self.assertEqual(next_xkrx_session(dt.date(2026, 10, 30)), dt.date(2026, 11, 2))
        self.assertFalse(is_month_end_xnys_session(dt.date(2026, 10, 29)))

    def test_loads_blended_portfolio(self):
        path = Path(__file__).parents[1] / "config" / "mailing.json"
        records = load_mailing_portfolios(path)
        self.assertEqual(records[0].title, "3. 프리셋 결합 (1 + 2, 50:50)")
        self.assertEqual(records[0].secondary_preset.name, "2. 집중투자 공격형")
        self.assertEqual(records[0].first_weight, 0.5)
        self.assertEqual(records[0].tracking_start, dt.date(2026, 11, 2))

    def test_email_contains_weights_and_decision_date(self):
        recommendation = RebalanceRecommendation(
            title="테스트",
            trade_date=dt.date(2026, 11, 2),
            domestic_trade_date=dt.date(2026, 11, 2),
            decision_date=dt.date(2026, 10, 30),
            selection="QQQ 50.0%, GLD 50.0%",
            domestic_selection="449190 · KODEX 미국나스닥100(H) 50.0%, 411060 · ACE KRX금현물 50.0%",
        )
        subject, text, html = render_email([recommendation])
        self.assertIn("2026-11", subject)
        self.assertIn("2026-10-30", text)
        self.assertIn("QQQ 50.0%, GLD 50.0%", html)
        self.assertIn("ACE KRX금현물", html)

    def test_email_contains_model_performance_and_drawdown_status(self):
        recommendation = RebalanceRecommendation(
            title="3. 프리셋 결합",
            trade_date=dt.date(2026, 12, 1),
            domestic_trade_date=dt.date(2026, 12, 1),
            decision_date=dt.date(2026, 11, 30),
            selection="BIL 50.0%, PDBC 50.0%",
            domestic_selection="329750 50.0%, 130680 50.0%",
            performance=PerformanceSnapshot(
                tracking_start=dt.date(2026, 11, 2),
                as_of=dt.date(2026, 11, 30),
                observations=20,
                cumulative_return=0.0312,
                monthly_return=0.0312,
                current_drawdown=-0.0125,
                max_drawdown=-0.0478,
                max_drawdown_date=dt.date(2026, 11, 18),
                drawdown_peak_date=dt.date(2026, 11, 25),
                drawdown_days=5,
            ),
        )
        _, text, html = render_email([recommendation])
        self.assertIn("누적 수익률: 3.12%", text)
        self.assertIn("고점 이후 5일", text)
        self.assertIn("운용 중 MDD", html)
        self.assertIn("-4.78%", html)

    def test_performance_snapshot_tracks_current_and_max_drawdown(self):
        result = {
            "nav": pd.Series(
                [1.02, 1.05, 1.00, 1.03],
                index=pd.to_datetime([
                    "2026-11-02", "2026-11-10", "2026-11-18", "2026-11-30",
                ]),
            ),
        }
        snapshot = _performance_snapshot(
            result, dt.date(2026, 11, 2), dt.date(2026, 11, 30),
        )
        self.assertAlmostEqual(snapshot.cumulative_return, 0.03)
        self.assertAlmostEqual(snapshot.monthly_return, 0.03)
        self.assertAlmostEqual(snapshot.current_drawdown, 1.03 / 1.05 - 1.0)
        self.assertAlmostEqual(snapshot.max_drawdown, 1.00 / 1.05 - 1.0)
        self.assertEqual(snapshot.max_drawdown_date, dt.date(2026, 11, 18))
        self.assertEqual(snapshot.drawdown_days, 20)

    def test_diversified_commodity_keeps_requested_crude_oil_proxy(self):
        mapped = _domestic_equivalent("PDBC 50.0%, QQQ 50.0%")
        self.assertIn("130680 · TIGER 원유선물Enhanced(H) 50.0%", mapped)
        self.assertNotIn("국내 대체 보류", mapped)


if __name__ == "__main__":
    unittest.main()
