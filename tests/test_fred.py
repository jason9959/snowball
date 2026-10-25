import unittest

import pandas as pd

from snowball.data.fred import download_fred_series, parse_fred_csv


class _Response:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M"

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class FredTests(unittest.TestCase):
    def test_parse_fred_csv_handles_missing_dot(self):
        series = parse_fred_csv(
            "DATE,T10Y3M\n2024-01-01,-1.2\n2024-01-02,.\n2024-01-03,-1.0\n",
            "T10Y3M",
        )
        self.assertEqual(series.tolist(), [-1.2, -1.0])
        self.assertEqual(series.index[-1], pd.Timestamp("2024-01-03"))

    def test_download_records_official_source_and_hash(self):
        calls = []

        def request_get(url, **kwargs):
            calls.append((url, kwargs))
            return _Response("DATE,T10Y3M\n2024-01-01,-1.2\n2024-01-31,-0.8\n")

        series, provenance = download_fred_series(
            "T10Y3M", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"),
            request_get=request_get,
        )
        self.assertEqual(series.iloc[-1], -0.8)
        self.assertIn("fred.stlouisfed.org", calls[0][0])
        self.assertEqual(provenance.vintage_mode, "current_history_not_alfred")
        self.assertEqual(len(provenance.sha256), 64)


if __name__ == "__main__":
    unittest.main()
