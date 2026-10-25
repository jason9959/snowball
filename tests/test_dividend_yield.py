import tempfile
import unittest
from pathlib import Path

import pandas as pd

from snowball.data.market_data import load_sp500_dividend_yield, parse_multpl_dividend_yield


class _Response:
    url = "https://www.multpl.com/s-p-500-dividend-yield/table/by-month"

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class DividendYieldTests(unittest.TestCase):
    HTML = """
    <table id="datatable">
      <tr><td>Feb 29, 2024</td><td>1.41%</td></tr>
      <tr><td>Mar 15, 2024</td><td><span title="Estimate">1.39%</span></td></tr>
      <tr><td>Mar 31, 2024</td><td><span title="Estimate">1.37%</span></td></tr>
    </table>
    """

    def test_parser_keeps_latest_row_and_estimate_flag(self):
        frame = parse_multpl_dividend_yield(self.HTML)
        self.assertEqual(frame["Date"].tolist(), [pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")])
        self.assertEqual(frame["SP500_DIV_YIELD"].tolist(), [1.41, 1.37])
        self.assertEqual(frame["Estimate"].tolist(), [False, True])

    def test_packaged_snapshot_needs_no_scrape(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            packaged = Path(directory) / "packaged.csv"
            cache = Path(directory) / "cache.csv"
            pd.DataFrame([
                {"Date": "2024-02-29", "SP500_DIV_YIELD": 1.41, "Estimate": False},
                {"Date": "2024-03-31", "SP500_DIV_YIELD": 1.37, "Estimate": False},
            ]).to_csv(packaged, index=False)
            result = load_sp500_dividend_yield(
                "2024-02-01", "2024-03-31", packaged_path=packaged, cache_path=cache,
                request_get=lambda *args, **kwargs: self.fail("scrape must not be called"),
            )
            self.assertEqual(result.tolist(), [1.41, 1.37])
            self.assertEqual(result.attrs["estimate_months"], [])

    def test_missing_month_scrape_records_estimate_and_hash(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            packaged = Path(directory) / "packaged.csv"
            cache = Path(directory) / "runtime" / "cache.csv"
            pd.DataFrame([
                {"Date": "2024-02-29", "SP500_DIV_YIELD": 1.41, "Estimate": False},
            ]).to_csv(packaged, index=False)
            result = load_sp500_dividend_yield(
                "2024-02-01", "2024-03-31", packaged_path=packaged, cache_path=cache,
                request_get=lambda *args, **kwargs: _Response(self.HTML),
            )
            self.assertEqual(result.tolist(), [1.41, 1.37])
            self.assertEqual(result.attrs["estimate_months"], ["2024-03"])
            self.assertEqual(len(result.attrs["sha256"]), 64)
            self.assertEqual(result.attrs["vintage_mode"], "current_history_not_point_in_time")


if __name__ == "__main__":
    unittest.main()
