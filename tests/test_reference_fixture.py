import re
import unittest
from pathlib import Path


class ReferenceFixtureTests(unittest.TestCase):
    def test_reference_table_is_complete_and_weights_sum_to_one_hundred(self):
        path = Path(__file__).resolve().parents[1] / "reference_rebalancing_2010_2026.md"
        text = path.read_text(encoding="utf-8")
        rows = re.findall(r"\| (20\d{2}-\d{2}) \| ([^|]+) \|", text)
        months = [month for month, _ in rows]
        self.assertEqual(len(rows), 202)
        self.assertEqual(len(set(months)), 202)
        self.assertEqual(months[0], "2010-01")
        self.assertEqual(months[-1], "2026-10")
        for month, holdings in rows:
            weights = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%", holdings)]
            self.assertAlmostEqual(sum(weights), 100.0, places=8, msg=month)


if __name__ == "__main__":
    unittest.main()
