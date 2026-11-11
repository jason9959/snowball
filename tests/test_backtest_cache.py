import unittest
from unittest.mock import patch

import pandas as pd

from common.snowball.backtest_cache import (
    cached_download_yahoo_prices,
    cached_result,
    clear_backtest_caches,
)
from snowball.data.prices import PriceBundle


class BacktestCacheTests(unittest.TestCase):
    def setUp(self):
        clear_backtest_caches()

    def test_result_cache_reuses_identical_calculation(self):
        calls = []

        def calculate():
            calls.append(True)
            return {"value": 7}

        first, first_hit = cached_result(("same", 1), calculate)
        second, second_hit = cached_result(("same", 1), calculate)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_price_cache_is_separate_and_normalizes_duplicate_symbols(self):
        index = pd.bdate_range("2025-01-01", periods=3)
        prices = pd.DataFrame({"QQQ": [1.0, 2.0, 3.0]}, index=index)
        bundle = PriceBundle(prices, prices.copy(), source="synthetic")
        with patch(
            "common.snowball.backtest_cache.download_yahoo_prices",
            return_value=bundle,
        ) as loader:
            first = cached_download_yahoo_prices(
                ("QQQ", "QQQ"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")
            )
            second = cached_download_yahoo_prices(
                ("QQQ",), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")
            )
        self.assertIs(first, second)
        loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
