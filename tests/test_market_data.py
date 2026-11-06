import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from snowball.data.market_data import load_strategy_market_data, load_treasury_spreads


class _Response:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M"

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class MarketDataTests(unittest.TestCase):
    def test_strategy_loader_only_requires_prior_completed_month(self):
        captured: dict[str, str] = {}

        def fake_spreads(series_ids, start_iso, end_iso, *, mode):
            captured.update({"start": start_iso, "end": end_iso, "mode": mode})
            return {
                "T10Y3M": pd.Series(
                    [0.78], index=pd.to_datetime(["2026-10-31"]), name="T10Y3M",
                )
            }

        with patch(
            "snowball.data.market_data.load_treasury_spreads",
            side_effect=fake_spreads,
        ):
            result = load_strategy_market_data(
                ("장단기 금리차 역전 (10년-3개월)",),
                "2010-01-01", "2026-11-06", mode="fixed",
            )
        self.assertEqual(captured["end"], "2026-10-31")
        self.assertEqual(captured["mode"], "fixed")
        self.assertEqual(result["T10Y3M"].iloc[-1], 0.78)

    def test_single_month_strategy_uses_prior_month_dividend_observation(self):
        dividend = pd.Series(
            [1.4], index=pd.to_datetime(["2026-10-31"]), name="SP500_DIV_YIELD",
        )
        dividend.attrs["estimate_months"] = []
        with (
            patch(
                "snowball.data.market_data.load_sp500_dividend_yield",
                return_value=dividend,
            ) as multpl,
            patch(
                "snowball.data.market_data.load_spy_distribution_yield",
                side_effect=ValueError("test fallback"),
            ),
        ):
            result = load_strategy_market_data(
                ("S&P500 배당수익률",),
                "2026-11-01", "2026-11-06", mode="fixed",
            )
        self.assertEqual(multpl.call_args.args[1], "2026-10-31")
        self.assertEqual(result["SP500_DIV_YIELD"].index[-1], pd.Timestamp("2026-10-31"))
        self.assertEqual(result["SP500_DIV_YIELD"].iloc[-1], 1.4)

    def test_packaged_months_do_not_call_network(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            packaged = Path(directory) / "packaged.csv"
            cache = Path(directory) / "cache.csv"
            pd.DataFrame([
                {"Date": "2024-01-31", "T10Y2Y": -0.2, "T10Y3M": -1.1},
                {"Date": "2024-02-29", "T10Y2Y": -0.1, "T10Y3M": -1.0},
            ]).to_csv(packaged, index=False)
            result = load_treasury_spreads(
                {"T10Y3M"}, "2024-01-01", "2024-02-29",
                packaged_path=packaged, cache_path=cache,
                request_get=lambda *args, **kwargs: self.fail("network must not be called"),
            )
            self.assertEqual(result["T10Y3M"].tolist(), [-1.1, -1.0])

    def test_missing_month_is_updated_from_official_fred_csv(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            packaged = Path(directory) / "packaged.csv"
            cache = Path(directory) / "runtime" / "cache.csv"
            pd.DataFrame([
                {"Date": "2024-01-31", "T10Y2Y": -0.2, "T10Y3M": -1.1},
            ]).to_csv(packaged, index=False)

            def request_get(url, **kwargs):
                self.assertIn("fred.stlouisfed.org", url)
                return _Response("DATE,T10Y3M\n2024-02-01,-0.9\n2024-02-29,-0.8\n")

            result = load_treasury_spreads(
                {"T10Y3M"}, "2024-01-01", "2024-02-29",
                packaged_path=packaged, cache_path=cache, request_get=request_get,
            )
            self.assertEqual(result["T10Y3M"].tolist(), [-1.1, -0.8])
            self.assertTrue(cache.exists())

    def test_latest_mode_refreshes_existing_months_and_keeps_version(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            packaged = Path(directory) / "packaged.csv"
            cache = Path(directory) / "runtime" / "cache.csv"
            pd.DataFrame([
                {"Date": "2024-01-31", "T10Y2Y": -0.2, "T10Y3M": -1.1},
                {"Date": "2024-02-29", "T10Y2Y": -0.1, "T10Y3M": -1.0},
            ]).to_csv(packaged, index=False)

            result = load_treasury_spreads(
                {"T10Y3M"}, "2024-01-01", "2024-02-29",
                packaged_path=packaged, cache_path=cache, mode="latest",
                request_get=lambda *args, **kwargs: _Response(
                    "DATE,T10Y3M\n2024-01-31,-1.2\n2024-02-29,-0.8\n"
                ),
            )
            self.assertEqual(result["T10Y3M"].tolist(), [-1.2, -0.8])
            self.assertTrue(result["T10Y3M"].attrs["snapshot_id"].startswith("fred-T10Y3M-"))
            self.assertEqual(len(list((cache.parent / "snapshots").glob("treasury_spreads_*.csv"))), 1)


if __name__ == "__main__":
    unittest.main()
