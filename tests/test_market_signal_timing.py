import unittest

import pandas as pd

from snowball.core.engine import _market_signal_details


class MarketSignalTimingTests(unittest.TestCase):
    def test_dividend_yield_uses_latest_month_end_available_to_rebalance(self):
        dividend = pd.Series(
            [1.20, 1.40],
            index=pd.to_datetime(["2021-09-30", "2021-10-31"]),
            name="SP500_DIV_YIELD",
        )
        dividend.attrs["source"] = "test snapshot"
        config = {
            "market_risk": {
                "signals": ["S&P500 배당수익률"],
                "dividend_threshold": 1.32,
            },
            "_market_data": {"SP500_DIV_YIELD": dividend},
        }
        rows = _market_signal_details(
            # October ended on Sunday. The final market session was Friday,
            # but the October monthly observation must still be selected.
            pd.DataFrame(), pd.Timestamp("2021-10-29"), config
        )
        self.assertEqual(rows[0]["관찰일"], pd.Timestamp("2021-10-31"))
        self.assertEqual(rows[0]["원시값"], 1.40)
        self.assertEqual(rows[0]["임계값"], 1.32)
        self.assertEqual(rows[0]["신호 출처"], "test snapshot")
        self.assertFalse(rows[0]["위험 판정"])


if __name__ == "__main__":
    unittest.main()
