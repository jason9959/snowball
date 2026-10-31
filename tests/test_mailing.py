import datetime as dt
import unittest
from pathlib import Path

from snowball.mailing import (
    RebalanceRecommendation,
    first_xnys_session,
    is_month_end_xnys_session,
    load_mailing_portfolios,
    next_xkrx_session,
    next_xnys_session,
    render_email,
)
from snowball.mailing import _domestic_equivalent


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
        self.assertEqual(records[0].title, "10. 프리셋 1 + 9 결합 (50:50)")
        self.assertEqual(records[0].secondary_preset.name, "9. MDD 개선 공격형")
        self.assertEqual(records[0].first_weight, 0.5)

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

    def test_diversified_commodity_is_not_mapped_to_crude_oil(self):
        mapped = _domestic_equivalent("PDBC 50.0%, QQQ 50.0%")
        self.assertIn("PDBC 국내 대체 보류", mapped)
        self.assertNotIn("130680", mapped)


if __name__ == "__main__":
    unittest.main()
