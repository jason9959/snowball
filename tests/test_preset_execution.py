import datetime as dt
import unittest

import numpy as np
import pandas as pd

from snowball.data.prices import PriceBundle
from snowball.domestic import DOMESTIC_EXECUTION_MAP
from snowball.presets import PRESETS
from snowball.service import (
    run_blended_presets,
    run_domestic_blended_presets,
    run_domestic_preset,
    run_preset,
)


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
        domestic_values = {}
        for position, symbol in enumerate(sorted(set(DOMESTIC_EXECUTION_MAP.values())), start=1):
            daily_rate = 0.00004 + position * 0.000004
            domestic_values[symbol] = 10_000.0 * np.exp(np.arange(len(cls.index)) * daily_rate)
        domestic_prices = pd.DataFrame(domestic_values, index=cls.index)
        cls.domestic_bundle = PriceBundle(
            domestic_prices, domestic_prices.copy(), source="synthetic domestic",
        )
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

    def test_domestic_presets_keep_us_signals_and_execute_mapped_etfs(self):
        for index, preset_name in enumerate(("1. 분산투자 방어형", "2. 집중투자 공격형"), start=4):
            with self.subTest(preset=preset_name):
                result = run_domestic_preset(
                    PRESETS[preset_name], self.bundle, self.domestic_bundle,
                    self.market_data, dt.date(2010, 1, 1), dt.date(2012, 12, 31),
                    preset_name=f"{index}. 국내 실행형",
                )
                self.assertIn("미국 신호 종목", result["history"].columns)
                self.assertTrue(
                    all(
                        ticker == "현금" or ticker.endswith(".KS")
                        for ticker in result["final_weights"]
                    )
                )
                first = result["history"].iloc[0]
                self.assertGreater(pd.Timestamp(first["적용일"]), pd.Timestamp(first["미국 적용일"]))

    def test_domestic_combined_preset_preserves_sleeve_weight(self):
        result = run_domestic_blended_presets(
            PRESETS["1. 분산투자 방어형"], PRESETS["2. 집중투자 공격형"], 0.5,
            self.bundle, self.domestic_bundle, self.market_data,
            dt.date(2010, 1, 1), dt.date(2012, 12, 31),
        )
        self.assertEqual(result["preset_name"], "6. 국내 프리셋 결합")
        self.assertIn("미국 신호 종목", result["history"].columns)
        self.assertAlmostEqual(result["history"].iloc[0]["프리셋 1 비중"], 0.5)

    def test_confirmed_gold_and_oil_replacements_are_locked(self):
        self.assertEqual(DOMESTIC_EXECUTION_MAP["GLD"], "411060.KS")
        self.assertEqual(DOMESTIC_EXECUTION_MAP["DBC"], "130680.KS")
        self.assertEqual(DOMESTIC_EXECUTION_MAP["PDBC"], "130680.KS")

    def test_domestic_presets_have_mappings_for_every_investable_asset(self):
        for preset_name in ("1. 분산투자 방어형", "2. 집중투자 공격형"):
            preset = PRESETS[preset_name]
            for ticker in preset.attack_tickers + preset.defense_tickers:
                with self.subTest(preset=preset_name, ticker=ticker):
                    self.assertIn(ticker, DOMESTIC_EXECUTION_MAP)
                    self.assertTrue(DOMESTIC_EXECUTION_MAP[ticker].endswith(".KS"))

    def test_preset_eight_combines_one_and_seven(self):
        result = run_blended_presets(
            PRESETS["1. 분산투자 방어형"],
            PRESETS["7. 집중투자 공격형 1-3-6-9-12"],
            0.5, self.bundle, self.market_data,
            dt.date(2010, 1, 1), dt.date(2012, 12, 31),
            preset_name="8. 프리셋 1 + 7 결합",
            second_label="프리셋 7",
        )
        self.assertEqual(result["preset_name"], "8. 프리셋 1 + 7 결합")
        self.assertEqual(result["sleeve_labels"], ("프리셋 1", "프리셋 7"))
        self.assertIn("프리셋 7 선택", result["history"].columns)
        self.assertAlmostEqual(result["history"].iloc[0]["프리셋 7 비중"], 0.5)

    def test_preset_ten_combines_one_and_nine(self):
        result = run_blended_presets(
            PRESETS["1. 분산투자 방어형"],
            PRESETS["9. MDD 개선 공격형"],
            0.5, self.bundle, self.market_data,
            dt.date(2010, 1, 1), dt.date(2012, 12, 31),
            preset_name="10. 프리셋 1 + 9 결합",
            second_label="프리셋 9",
        )
        self.assertEqual(result["preset_name"], "10. 프리셋 1 + 9 결합")
        self.assertEqual(result["sleeve_labels"], ("프리셋 1", "프리셋 9"))
        self.assertIn("프리셋 9 선택", result["history"].columns)
        self.assertAlmostEqual(result["history"].iloc[0]["프리셋 9 비중"], 0.5)


if __name__ == "__main__":
    unittest.main()
