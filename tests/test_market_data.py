import tempfile
import unittest
from pathlib import Path

import pandas as pd

from snowball.data.market_data import load_treasury_spreads


class _Response:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M"

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class MarketDataTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
