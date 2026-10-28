import datetime as dt
import unittest

from snowball.presets import PRESETS, get_preset


class PresetTests(unittest.TestCase):
    def test_all_presets_are_complete_and_build_engine_configs(self):
        for preset in PRESETS.values():
            preset.validate()
            config = preset.engine_config(dt.date(2010, 1, 1), dt.date(2019, 12, 31))
            self.assertTrue(config["attack_tickers"])
            self.assertIn("fixed_defense_weight", config)
            self.assertIn("risk_avoidance", config)
            self.assertIn("market_risk", config)

    def test_defensive_research_preset_matches_agreed_definition(self):
        preset = get_preset("1. 분산투자 방어형")
        self.assertEqual(preset.attack_tickers, ("QQQ", "SOXX", "XLE", "XLK", "XLV", "EDV", "GLD", "DBC"))
        self.assertEqual(preset.attack_select_count, 4)
        self.assertEqual(preset.negative_threshold, 5)
        self.assertEqual(preset.canary_tickers, ("SPY", "TIP"))
        self.assertIsNone(preset.canary_decision_percent_digits)
        self.assertAlmostEqual(preset.dividend_threshold, 1.32)
        self.assertEqual(preset.market_signals, ("S&P500 배당수익률",))

    def test_aggressive_research_preset_matches_agreed_definition(self):
        preset = get_preset("2. 집중투자 공격형")
        self.assertEqual(preset.attack_tickers, ("QQQ", "SCHD", "VWO"))
        self.assertEqual(preset.attack_select_count, 1)
        self.assertEqual(preset.defense_tickers, ("GLD", "HYG", "PDBC"))
        self.assertIsNone(preset.canary_decision_percent_digits)
        self.assertIsNotNone(preset.spread_10y3m)
        self.assertEqual(preset.spread_10y3m.as_dict(), {"lag": 7, "active": 8, "threshold": -0.5})
        self.assertAlmostEqual(preset.dividend_threshold, 1.6)

    def test_alternative_aggressive_preset_only_changes_attack_score(self):
        original = get_preset("2. 집중투자 공격형")
        alternative = get_preset("7. 집중투자 공격형 1-3-6-9-12")
        self.assertEqual(alternative.attack_tickers, original.attack_tickers)
        self.assertEqual(alternative.defense_tickers, original.defense_tickers)
        self.assertEqual(alternative.attack_relative.score_type, "1-3-6-9-12개월 단순")
        self.assertEqual(alternative.attack_absolute.score_type, "1-3-6-9-12개월 단순")
        self.assertEqual(alternative.defense_relative, original.defense_relative)
        self.assertEqual(alternative.canary_tickers, original.canary_tickers)
        self.assertEqual(alternative.market_signals, original.market_signals)
        self.assertEqual(alternative.spread_10y3m, original.spread_10y3m)
        self.assertEqual(alternative.dividend_threshold, original.dividend_threshold)

    def test_unknown_preset_is_actionable_value_error(self):
        with self.assertRaisesRegex(ValueError, "알 수 없는 프리셋"):
            get_preset("없는 전략")


if __name__ == "__main__":
    unittest.main()
