import datetime as dt
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

if "yfinance" not in sys.modules:
    try:
        __import__("yfinance")
    except ModuleNotFoundError:
        sys.modules["yfinance"] = types.SimpleNamespace(download=None)

from common.functions.analytics import ToolResult
from common.functions.specialized import (
    dollar_compass,
    dynamic_portfolio,
    fear_greed,
    inflation_audit,
    real_estate_vs_stock,
    reb_statistics_multi,
)


class SpecializedAdapterTests(unittest.TestCase):
    def test_inflation_audit_reads_reproducible_summary(self):
        result = inflation_audit("일별")
        self.assertEqual(result.title, "인플레이션 나침반 검증 결과")
        self.assertEqual(result.metrics["위험 측정 주기"], "일별")
        self.assertIn("검증 조건", result.table.columns)

    def test_dynamic_portfolio_exposes_native_engine_outputs(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        native = {
            "nav": pd.Series([1.0, 1.02, 1.01], index=dates),
            "values": pd.Series([10_000, 10_200, 10_100], index=dates),
            "benchmark_nav": pd.Series([1.0, 1.01, 1.02], index=dates),
            "equal_weight_nav": pd.Series([1.0, 1.00, 1.01], index=dates),
            "drawdown": pd.Series([0.0, 0.0, -0.01], index=dates),
            "annual_turnover": 0.5,
            "final_weights": {"SPY": 1.0},
            "history": pd.DataFrame({"적용일": dates, "선택": ["SPY"] * 3}),
            "common_start": dates[0],
            "common_end": dates[-1],
        }
        page = types.ModuleType("features.feature04_dynamic_portfolio.page")
        page._run = lambda _config: native
        results = types.ModuleType("features.feature04_dynamic_portfolio.results")
        results._metric_summary = lambda *_args: {
            "기간 수익률": 0.01,
            "연환산 수익률": 0.10,
            "연 변동성": 0.12,
            "최대 낙폭 (MDD)": -0.01,
        }
        with patch.dict(
            sys.modules,
            {
                "features.feature04_dynamic_portfolio.page": page,
                "features.feature04_dynamic_portfolio.results": results,
            },
        ):
            result = dynamic_portfolio({"currency": "USD"})
        self.assertEqual(result.title, "동적 포트폴리오 백테스트")
        self.assertEqual(result.metrics["현재 비중"], "SPY 100.0%")
        self.assertEqual(result.series["포트폴리오"].iloc[-1], 1.01)

    def test_dollar_backtest_returns_the_original_result(self):
        dates = pd.date_range("2024-01-01", periods=3)
        market = pd.DataFrame({"date": dates, "usdkrw": [1300, 1310, 1320]})
        curve = pd.DataFrame(
            {
                "date": dates,
                "total_value": [10_000, 10_100, 10_200],
                "return": [0.0, 0.01, 0.02],
                "usdkrw": [1300, 1310, 1320],
            }
        )
        switches = pd.DataFrame({"date": [dates[1]], "action": ["switch"]})
        page = types.ModuleType("features.feature09_dollar.page")
        page.load_market_data = lambda: market
        page.add_indicators = lambda value: value
        page.backtest_switching = lambda *_args: (curve, switches)
        page.max_drawdown = lambda _value: -0.05
        with patch.dict(sys.modules, {"features.feature09_dollar.page": page}):
            result = dollar_compass(dt.date(2024, 1, 1), dt.date(2024, 1, 3), 10_000, 10, 0.15)
        self.assertEqual(result.title, "달러 환율 나침반")
        self.assertEqual(result.metrics["환전 횟수"], "1회")
        self.assertEqual(result.series["포트폴리오"].iloc[-1], 10_200)

    def test_fear_greed_backtest_returns_values_and_events(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        indicators = pd.DataFrame({"fear_and_greed_historical": [20, 30, 40]}, index=dates)
        prices = pd.DataFrame({"SPY": [100, 101, 102]}, index=dates)
        output = {
            "values": pd.Series([10_000, 10_100, 10_200], index=dates),
            "drawdown": pd.Series([0.0, -0.01, 0.0], index=dates),
            "indicator": indicators.iloc[:, 0],
            "events": pd.DataFrame({"date": [dates[0]], "event": ["rebalance"]}),
        }
        data = types.ModuleType("features.feature14_fear_greed.data")
        data.fetch_fear_greed = lambda *_args: indicators
        data.fetch_prices = lambda *_args: prices
        backtest = types.ModuleType("features.feature14_fear_greed.backtest")
        backtest.run_backtest = lambda *_args: output
        with patch.dict(
            sys.modules,
            {
                "features.feature14_fear_greed.data": data,
                "features.feature14_fear_greed.backtest": backtest,
            },
        ):
            result = fear_greed(
                dt.date(2024, 1, 2), dt.date(2024, 1, 4), ["SPY"], [100], 25, 10_000
            )
        self.assertEqual(result.title, "공포·탐욕 리밸런싱")
        self.assertEqual(result.metrics["최종 평가액"], "10,200")
        self.assertEqual(len(result.table), 1)

    def test_real_estate_simulation_exposes_percentile_table(self):
        dates = pd.bdate_range("2020-01-02", "2022-01-03")
        price = pd.DataFrame({"VOO": np.linspace(100, 130, len(dates))}, index=dates)
        with patch("common.functions.specialized.prices", return_value=price):
            result = real_estate_vs_stock(
                "VOO", dt.date(2020, 1, 2), dt.date(2022, 1, 3),
                500_000_000, 300_000_000, 0, 1, 2, 4,
                "원리금균등상환", 1_000_000, 24, 5, 30,
                50_000_000, "Bootstrap", 100,
            )
        self.assertIn("주식 평가액(P50)", result.metrics)
        self.assertEqual(result.metrics["경로 수"], "100개")
        self.assertEqual(list(result.table.columns), ["개월", "주택 순자산", "주식 P5", "주식 P50", "주식 P95"])

    def test_multi_region_requires_selection_and_merges_results(self):
        with self.assertRaisesRegex(ValueError, "지역"):
            reb_statistics_multi("table", "MM", [], "item", 2020, 2024)

        def one_region(_table, _cycle, region, _item, _start, _end, _kind):
            frame = pd.DataFrame({"지역명": [region], "값": [1.0]})
            return ToolResult(region, {}, {"지수": pd.Series([1.0])}, frame)

        with patch("common.functions.specialized.reb_statistics", side_effect=one_region):
            result = reb_statistics_multi("table", "MM", ["A", "B"], "item", 2020, 2024)
        self.assertEqual(result.metrics["비교 지역"], "2개")
        self.assertEqual(set(result.series), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
