import unittest

import pandas as pd

from features.snowball04_split_trading.engine import SplitTradingConfig, run_split_backtest
from features.snowball04_split_trading.sheets import export_result


def closes(values):
    return pd.Series(values, index=pd.date_range("2026-01-02", periods=len(values), freq="B"))


class SplitTradingTests(unittest.TestCase):
    def test_high_watermark_sets_second_boundary(self):
        result = run_split_backtest(
            closes([100, 130, 140, 135, 125]),
            SplitTradingConfig(300, 100, .10, .10),
        )
        buys = result["trades"].query("`매수/매도` == '매수'")
        self.assertEqual(buys["차수"].tolist(), [1, 2])
        self.assertAlmostEqual(buys.iloc[-1]["시뮬레이션 체결가"], 126.0)

    def test_documented_multi_level_example(self):
        result = run_split_backtest(
            closes([100, 130, 140, 135, 110, 142]),
            SplitTradingConfig(300, 100, .10, .10),
        )
        frame = result["trades"]
        self.assertEqual(frame["매수/매도"].tolist(), ["매수", "매수", "매수", "매도", "매도"])
        self.assertEqual(frame["차수"].tolist(), [1, 2, 3, 3, 2])
        self.assertAlmostEqual(frame.iloc[1]["시뮬레이션 체결가"], 126.0)
        self.assertAlmostEqual(frame.iloc[2]["시뮬레이션 체결가"], 113.4)
        self.assertAlmostEqual(frame.iloc[3]["시뮬레이션 체결가"], 124.74)
        self.assertAlmostEqual(frame.iloc[4]["시뮬레이션 체결가"], 138.6)
        self.assertAlmostEqual(result["final_cash"], 220.0)
        self.assertAlmostEqual(result["split_final"], 362.0)
        self.assertAlmostEqual(result["split_return"], 0.2066666667)
        self.assertAlmostEqual(result["full_final"], 426.0)

    def test_remaining_cash_and_slot_count(self):
        result = run_split_backtest(
            closes([100, 70]),
            SplitTradingConfig(1_000_000, 300_000, .10, .10),
        )
        self.assertEqual(result["max_slots"], 3)
        self.assertAlmostEqual(result["final_cash"], 100_000)

    def test_one_slot_only_buys_first_position(self):
        result = run_split_backtest(
            closes([100, 50, 120]),
            SplitTradingConfig(100, 100, .10, .10),
        )
        self.assertEqual(len(result["trades"]), 1)
        self.assertAlmostEqual(result["split_final"], 120)

    def test_sheet_export_uses_next_available_number_and_numeric_cells(self):
        class Worksheet:
            def __init__(self, title, sheet_id=7):
                self.title, self.id, self.rows = title, sheet_id, []
            def update(self, *, range_name, values, raw):
                self.rows = values
            def freeze(self, **kwargs):
                pass
            def format(self, *args, **kwargs):
                pass

        class Spreadsheet:
            id = "sheet-id"
            def __init__(self):
                self.items = [Worksheet("기존1"), Worksheet("기존2"), Worksheet("3. 테스트")]
            def worksheets(self):
                return self.items
            def add_worksheet(self, *, title, rows, cols):
                sheet = Worksheet(title, 99)
                self.items.append(sheet)
                return sheet

        result = run_split_backtest(
            closes([100, 90, 110]), SplitTradingConfig(300, 100, .10, .10),
        )
        book = Spreadsheet()
        title, url = export_result(
            book, symbol_name="테스트", conditions={"국가": "한국"}, result=result,
        )
        self.assertEqual(title, "4. 테스트")
        self.assertIn("#gid=99", url)
        self.assertIsInstance(book.items[-1].rows[5][1], float)


if __name__ == "__main__":
    unittest.main()
