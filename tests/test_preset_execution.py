import datetime as dt
import unittest

import numpy as np
import pandas as pd

from snowball.data.prices import PriceBundle
from snowball.presets import PRESETS
from snowball.service import run_blended_presets, run_preset


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

    def test_half_and_half_blend_combines_nav_and_rebalancing_weights(self):
        first = run_preset(
            PRESETS["1. 분산투자 방어형"], self.bundle, self.market_data,
            dt.date(2010, 1, 1), dt.date(2012, 12, 31),
        )
        second = run_preset(
            PRESETS["2. 집중투자 공격형"], self.bundle, self.market_data,
            dt.date(2010, 1, 1), dt.date(2012, 12, 31),
        )
        blended = run_blended_presets(
            PRESETS["1. 분산투자 방어형"], PRESETS["2. 집중투자 공격형"], 0.5,
            self.bundle, self.market_data,
            dt.date(2010, 1, 1), dt.date(2012, 12, 31),
        )
        self.assertEqual(blended["preset_name"], "3. 프리셋 결합")
        self.assertAlmostEqual(blended["history"].iloc[0]["프리셋 1 비중"], 0.5)
        self.assertAlmostEqual(blended["history"].iloc[0]["프리셋 2 비중"], 0.5)
        self.assertNotEqual(
            blended["history"].iloc[0]["프리셋 1 선택"],
            blended["history"].iloc[0]["프리셋 2 선택"],
        )
        selected = blended["history"].iloc[0]["선택된 종목"]
        total = sum(float(item.rsplit(" ", 1)[1].rstrip("%")) for item in selected.split(", "))
        self.assertAlmostEqual(total, 100.0)
        self.assertGreater(float(blended["nav"].iloc[-1]), 0.0)

    def test_blend_endpoints_match_their_component_nav(self):
        for weight, preset_name in ((1.0, "1. 분산투자 방어형"), (0.0, "2. 집중투자 공격형")):
            with self.subTest(weight=weight):
                blended = run_blended_presets(
                    PRESETS["1. 분산투자 방어형"], PRESETS["2. 집중투자 공격형"], weight,
                    self.bundle, self.market_data,
                    dt.date(2010, 1, 1), dt.date(2012, 12, 31),
                )
                component = run_preset(
                    PRESETS[preset_name], self.bundle, self.market_data,
                    dt.date(2010, 1, 1), dt.date(2012, 12, 31),
                )
                pd.testing.assert_series_equal(blended["nav"], component["nav"])


if __name__ == "__main__":
    unittest.main()
