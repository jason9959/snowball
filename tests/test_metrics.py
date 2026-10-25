import unittest

import pandas as pd

from snowball.metrics import summarize_nav


class MetricTests(unittest.TestCase):
    def test_metrics_use_unrounded_nav(self):
        index = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        nav = pd.Series(1.0 + pd.Series(range(len(index)), index=index) / 10_000.0, index=index)
        metrics = summarize_nav(nav)
        self.assertGreater(metrics["total_return"], 0)
        self.assertEqual(metrics["max_drawdown"], 0)


if __name__ == "__main__":
    unittest.main()
