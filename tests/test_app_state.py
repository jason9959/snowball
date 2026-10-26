import datetime as dt
import unittest

from snowball.app_state import SavedRun
from snowball.presets import PRESETS


class AppStateTests(unittest.TestCase):
    def test_saved_run_validates(self):
        saved = SavedRun(
            preset=PRESETS["1. 분산투자 방어형"],
            start_date=dt.date(2010, 1, 1),
            end_date=dt.date(2026, 10, 23),
            initial_investment=10_000.0,
            transaction_cost_rate=0.001,
        )
        saved.validate()

    def test_invalid_period_is_rejected(self):
        saved = SavedRun(
            preset=PRESETS["1. 분산투자 방어형"],
            start_date=dt.date(2026, 1, 1),
            end_date=dt.date(2026, 1, 1),
            initial_investment=10_000.0,
            transaction_cost_rate=0.0,
        )
        with self.assertRaises(ValueError):
            saved.validate()


if __name__ == "__main__":
    unittest.main()
