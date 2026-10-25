import datetime as dt
import unittest

import numpy as np
import pandas as pd

from snowball.data.prices import PriceBundle
from snowball.presets import PRESETS
from snowball.service import run_preset


class PresetExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = pd.bdate_range("2008-01-01", "2013-12-31")
        symbols = sorted({symbol for preset in PRESETS.values() for symbol in preset.symbols})
        values = {}
        for position, symbol in enumerate(symbols, start=1):
            daily_rate = 0.00005 + position * 0.000005
            values[symbol] = 100.0 * np.exp(np.arange(len(cls.index)) * daily_rate)
        prices = pd.DataFrame(values, index=cls.index)
        cls.bundle = PriceBundle(prices, prices.copy(), source="synthetic")
        monthly = pd.date_range("2007-01-31", "2013-12-31", freq="ME")
        cls.market_data = {
            "SP500_DIV_YIELD": pd.Series(5.0, index=monthly),
            "T10Y3M": pd.Series(1.0, index=monthly),
        }

    def test_each_research_preset_runs_through_the_real_engine(self):
        for preset in PRESETS.values():
            with self.subTest(preset=preset.name):
                result = run_preset(
                    preset, self.bundle, self.market_data,
                    dt.date(2010, 1, 1), dt.date(2012, 12, 31),
                )
                self.assertFalse(result["nav"].empty)
                self.assertGreater(float(result["nav"].iloc[-1]), 0.0)
                self.assertEqual(result["preset_name"], preset.name)
                self.assertIn("metrics", result)


if __name__ == "__main__":
    unittest.main()
