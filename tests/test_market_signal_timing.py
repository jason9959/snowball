import unittest

import pandas as pd

from snowball.core.engine import _market_signal_details


class MarketSignalTimingTests(unittest.TestCase):
    def test_dividend_yield_uses_latest_month_end_available_to_rebalance(self):
        dividend = pd.Series(
            [1.20, 1.40],
            index=pd.to_datetime(["2021-12-31", "2022-01-31"]),
            name="SP500_DIV_YIELD",
        )
        config = {
            "market_risk": {
                "signals": ["S&P500 배당수익률"],
                "dividend_threshold": 1.32,
            },
            "_market_data": {"SP500_DIV_YIELD": dividend},
        }
        rows = _market_signal_details(
            pd.DataFrame(), pd.Timestamp("2022-01-31"), config
        )
        self.assertEqual(rows[0]["관찰일"], pd.Timestamp("2022-01-31"))
        self.assertEqual(rows[0]["원시값"], 1.40)
        self.assertFalse(rows[0]["위험 판정"])


if __name__ == "__main__":
    unittest.main()
