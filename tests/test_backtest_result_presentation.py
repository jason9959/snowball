import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class BacktestResultPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_nicegui = types.ModuleType("nicegui")
        fake_nicegui.ui = types.SimpleNamespace()
        with patch.dict(sys.modules, {"nicegui": fake_nicegui}):
            cls.module = importlib.import_module("common.snowball.backtest_results")

    def setUp(self):
        self.monthly = pd.DataFrame(
            {
                "적용 월": ["2025-01", "2025-02", "2025-03", "2025-04"],
                "수익률 원시값": [0.04, -0.03, 0.01, -0.005],
                "수익률 (%)": [4.0, -3.0, 1.0, -0.5],
            }
        )

    def test_monthly_heatmap_keeps_sign_and_calendar_position(self):
        options = self.module._monthly_heatmap_options(self.monthly)
        data = options["series"][0]["data"]
        self.assertEqual([item["value"][0] for item in data], [0, 1, 2, 3])
        self.assertEqual([item["value"][2] for item in data], [4.0, -3.0, 1.0, -0.5])
        self.assertEqual(options["xAxis"]["data"][0], "1월")
        self.assertEqual(options["yAxis"]["data"], ["2025"])

    def test_monthly_distribution_counts_every_observation(self):
        options = self.module._monthly_distribution_options(self.monthly)
        counts = [item["value"] for item in options["series"][0]["data"]]
        self.assertEqual(sum(counts), len(self.monthly))

    def test_mix_page_does_not_warm_prices_before_result_cache(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "features" / "snowball02_portfolio_mix" / "page.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("prepare_mix_prices", source)
        self.assertIn("저장된 동일 결과 사용", source)

    def test_result_tabs_are_built_only_when_selected(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "common" / "snowball" / "backtest_results.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("ui.tab_panels", source)
        self.assertIn('render_selected("returns")', source)
        self.assertIn("renderers[name]()", source)


if __name__ == "__main__":
    unittest.main()
