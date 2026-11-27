import unittest
import sys
import types

import pandas as pd
import numpy as np

from common.functions.analytics import ToolResult

if "nicegui" not in sys.modules:
    try:
        __import__("nicegui")
    except ModuleNotFoundError:
        sys.modules["nicegui"] = types.SimpleNamespace(ui=types.SimpleNamespace())

from common.functions.presentation import (
    deserialize_tool_result,
    drawdown_series,
    serialize_tool_result,
)


class FunctionsPresentationTests(unittest.TestCase):
    def test_explicit_drawdown_is_kept_out_of_general_series_and_converted_to_percent(self):
        dates = pd.bdate_range("2026-01-02", periods=3)
        result = ToolResult(
            "정적 포트폴리오", {"MDD": "-10.00%"},
            {
                "포트폴리오": pd.Series([100, 90, 110], index=dates),
                "누적 투입금": pd.Series([100, 100, 100], index=dates),
                "낙폭": pd.Series([0, -.10, 0], index=dates),
            },
        )
        drawdowns = drawdown_series(result.series)
        self.assertEqual(list(drawdowns), ["포트폴리오 낙폭"])
        self.assertEqual(drawdowns["포트폴리오 낙폭"].tolist(), [0.0, -10.0, 0.0])

    def test_drawdown_is_derived_for_positive_performance_series(self):
        dates = pd.bdate_range("2026-01-02", periods=3)
        drawdowns = drawdown_series({
            "QQQ": pd.Series([100, 80, 120], index=dates),
        })
        self.assertAlmostEqual(float(drawdowns["QQQ 낙폭"].min()), -20.0)

    def test_tool_result_round_trip_preserves_dates_series_table_and_korean(self):
        dates = pd.bdate_range("2026-01-02", periods=3)
        source = ToolResult(
            "정적 포트폴리오", {"최종 평가액": "12,345"},
            {"포트폴리오": pd.Series([10_000, 11_000, 12_345], index=dates)},
            pd.DataFrame({"날짜": dates, "유형": ["매수", "유지", "매도"]}),
            "직전 결과",
        )
        restored = deserialize_tool_result(serialize_tool_result(source))
        self.assertEqual(restored.title, source.title)
        self.assertEqual(restored.metrics, source.metrics)
        self.assertEqual(restored.note, "직전 결과")
        pd.testing.assert_series_equal(
            restored.series["포트폴리오"], source.series["포트폴리오"],
            check_freq=False, check_dtype=False,
        )
        self.assertEqual(restored.table["유형"].tolist(), ["매수", "유지", "매도"])

    def test_persisted_metrics_are_json_safe(self):
        source = ToolResult(
            "안전한 저장", {"기준일": pd.Timestamp("2026-11-26"), "값": np.float64(3.5)},
        )
        payload = serialize_tool_result(source)
        self.assertEqual(payload["metrics"]["기준일"], "2026-11-26T00:00:00")
        self.assertEqual(payload["metrics"]["값"], 3.5)


if __name__ == "__main__":
    unittest.main()
