import datetime as dt
import unittest

import numpy as np
import pandas as pd

from snowball.builtin_library import (
    BUILTIN_PRESET_SPECS,
    get_builtin_spec,
    run_builtin_backtest,
)
from snowball.data.prices import PriceBundle


class BuiltinLibraryTests(unittest.TestCase):
    def test_library_contains_titles_one_through_ten(self):
        self.assertEqual(len(BUILTIN_PRESET_SPECS), 10)
        self.assertEqual(
            [int(title.split(".", 1)[0]) for title in BUILTIN_PRESET_SPECS],
            list(range(1, 11)),
        )

    def test_combined_and_domestic_definitions_are_preserved(self):
        third = get_builtin_spec("3. 프리셋 결합")
        sixth = get_builtin_spec("6. 국내 프리셋 결합")
        self.assertEqual(third.primary.name, "1. 분산투자 방어형")
        self.assertEqual(third.secondary.name, "2. 집중투자 공격형")
        self.assertEqual(third.primary_weight, 0.5)
        self.assertTrue(sixth.domestic_execution)

    def test_unknown_title_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "알 수 없는"):
            get_builtin_spec("없는 프리셋")

    def test_combined_runner_uses_explicit_prior_signal_month(self):
        symbols = sorted({
            symbol
            for spec in BUILTIN_PRESET_SPECS.values()
            for preset in (spec.primary, spec.secondary)
            if preset is not None
            for symbol in preset.symbols
        } | {"QQQ"})
        index = pd.bdate_range("2008-01-01", "2012-12-31")
        prices = pd.DataFrame({
            symbol: 100.0 * np.exp(np.arange(len(index)) * (0.0001 + offset * 0.000001))
            for offset, symbol in enumerate(symbols)
        }, index=index)
        bundle = PriceBundle(prices, prices.copy(), source="synthetic")
        captured: dict[str, str] = {}

        def market_loader(signals, start_iso, end_iso, *, mode, signal_end_iso):
            captured["signal_end"] = signal_end_iso
            monthly = pd.date_range("2007-01-31", "2012-12-31", freq="ME")
            return {
                "SP500_DIV_YIELD": pd.Series(5.0, index=monthly),
                "T10Y3M": pd.Series(1.0, index=monthly),
            }

        result, _ = run_builtin_backtest(
            "3. 프리셋 결합", dt.date(2010, 1, 1), dt.date(2011, 12, 31),
            initial_investment=10_000,
            price_loader=lambda symbols, start, end: bundle,
            market_loader=market_loader,
        )
        self.assertEqual(captured["signal_end"], "2011-11-30")
        self.assertEqual(result["preset_name"], "3. 프리셋 결합")
        self.assertFalse(result["nav"].empty)


if __name__ == "__main__":
    unittest.main()
