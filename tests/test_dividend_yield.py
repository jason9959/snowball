import tempfile
import unittest
from pathlib import Path

import pandas as pd

from snowball.data.market_data import (
    calculate_spy_distribution_yield,
    compare_dividend_snapshots,
    correct_estimated_sp500_dividend_yield,
    load_sp500_dividend_yield,
    parse_multpl_dividend_yield,
)


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

    def test_fixed_mode_ignores_changed_runtime_cache(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            packaged = Path(directory) / "packaged.csv"
            cache = Path(directory) / "cache.csv"
            pd.DataFrame([
                {"Date": "2024-02-29", "SP500_DIV_YIELD": 1.41, "Estimate": False},
            ]).to_csv(packaged, index=False)
            pd.DataFrame([
                {"Date": "2024-02-29", "SP500_DIV_YIELD": 9.99, "Estimate": False},
            ]).to_csv(cache, index=False)
            result = load_sp500_dividend_yield(
                "2024-02-01", "2024-02-29", packaged_path=packaged,
                cache_path=cache, mode="fixed",
                request_get=lambda *args, **kwargs: self.fail("fixed mode must not call network"),
            )
            self.assertEqual(result.tolist(), [1.41])
            self.assertEqual(result.attrs["snapshot_mode"], "fixed")
            self.assertTrue(result.attrs["snapshot_id"].startswith("bundled-dividend-"))

    def test_latest_mode_preserves_hashed_candidate_version(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            packaged = Path(directory) / "packaged.csv"
            cache = Path(directory) / "runtime" / "cache.csv"
            pd.DataFrame([
                {"Date": "2024-02-29", "SP500_DIV_YIELD": 1.40, "Estimate": False},
                {"Date": "2024-03-31", "SP500_DIV_YIELD": 1.38, "Estimate": False},
            ]).to_csv(packaged, index=False)
            result = load_sp500_dividend_yield(
                "2024-02-01", "2024-03-31", packaged_path=packaged,
                cache_path=cache, mode="latest",
                request_get=lambda *args, **kwargs: _Response(self.HTML),
            )
            self.assertEqual(result.tolist(), [1.41, 1.37])
            versions = list((cache.parent / "snapshots").glob("sp500_dividend_yield_*.csv"))
            self.assertEqual(len(versions), 1)
            self.assertEqual(result.attrs["snapshot_mode"], "latest")

    def test_snapshot_comparison_identifies_changes_and_new_months(self):
        baseline = pd.Series(
            [1.41], index=pd.to_datetime(["2024-02-29"]), name="SP500_DIV_YIELD",
        )
        candidate = pd.DataFrame(
            {"SP500_DIV_YIELD": [1.42, 1.37]},
            index=pd.to_datetime(["2024-02-29", "2024-03-31"]),
        )
        compared = compare_dividend_snapshots(
            baseline, candidate, "2024-02-01", "2024-03-31",
        )
        self.assertEqual(compared["상태"].tolist(), ["변경", "신규"])

    def test_spy_yield_uses_raw_close_and_trailing_365_day_distributions(self):
        history = pd.DataFrame(
            {
                "Close": [100.0, 110.0, 120.0, 125.0],
                "Dividends": [1.0, 1.0, 1.0, 0.0],
            },
            index=pd.to_datetime(["2023-03-31", "2023-09-29", "2024-03-28", "2024-03-29"]),
        )
        result = calculate_spy_distribution_yield(history)
        self.assertAlmostEqual(result.loc["2024-03-31", "SPY_TTM_DISTRIBUTION"], 2.0)
        self.assertAlmostEqual(result.loc["2024-03-31", "SPY_CLOSE"], 125.0)
        self.assertAlmostEqual(result.loc["2024-03-31", "SPY_DISTRIBUTION_YIELD"], 1.6)

    def test_only_estimates_are_replaced_by_prior_confirmed_spy_spread(self):
        index = pd.date_range("2023-01-31", periods=15, freq="ME")
        multpl = pd.Series([1.50] * 12 + [1.40, 1.39, 1.38], index=index, name="SP500_DIV_YIELD")
        multpl.attrs.update({
            "estimate_months": [str(period) for period in index[-3:].to_period("M")],
            "snapshot_id": "multpl-test",
        })
        spy = pd.Series([1.40] * 15, index=index, name="SPY_DISTRIBUTION_YIELD")
        spy.attrs["snapshot_id"] = "spy-test"
        corrected = correct_estimated_sp500_dividend_yield(multpl, spy)
        self.assertEqual(corrected.iloc[:12].tolist(), multpl.iloc[:12].tolist())
        self.assertEqual(corrected.iloc[-3:].round(6).tolist(), [1.50, 1.50, 1.50])
        self.assertEqual(corrected.attrs["corrected_months"], ["2024-01", "2024-02", "2024-03"])
        self.assertEqual(corrected.attrs["raw_multpl_values"]["2024-03"], 1.38)

    def test_estimate_falls_back_when_calibration_history_is_too_short(self):
        index = pd.date_range("2024-01-31", periods=3, freq="ME")
        multpl = pd.Series([1.5, 1.4, 1.3], index=index)
        multpl.attrs.update({"estimate_months": ["2024-03"], "snapshot_id": "m"})
        spy = pd.Series([1.4, 1.3, 1.2], index=index)
        spy.attrs["snapshot_id"] = "s"
        corrected = correct_estimated_sp500_dividend_yield(multpl, spy)
        self.assertEqual(corrected.iloc[-1], 1.3)
        self.assertEqual(corrected.attrs["fallback_estimate_months"], ["2024-03"])


if __name__ == "__main__":
    unittest.main()
