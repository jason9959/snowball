import datetime as dt
import json
import unittest

import numpy as np
import pandas as pd

from common.snowball.serialization import dataframe_records, json_safe


class PresentationSerializationTests(unittest.TestCase):
    def test_dataframe_records_convert_timestamp_and_numpy_values(self) -> None:
        frame = pd.DataFrame({
            "날짜": [pd.Timestamp("2026-11-25")],
            "값": [np.float64(1.25)],
            "누락": [np.nan],
        })
        rows = dataframe_records(frame)
        self.assertEqual(rows[0]["날짜"], "2026-11-25T00:00:00")
        self.assertEqual(rows[0]["값"], 1.25)
        self.assertIsNone(rows[0]["누락"])
        json.dumps(rows, ensure_ascii=False)

    def test_recursive_json_safe_handles_dates_and_non_finite_values(self) -> None:
        value = json_safe({
            "date": dt.date(2026, 11, 25),
            "timestamp": pd.Timestamp("2026-11-25 12:30"),
            "nan": float("nan"),
        })
        self.assertEqual(value["date"], "2026-11-25")
        self.assertEqual(value["timestamp"], "2026-11-25T12:30:00")
        self.assertIsNone(value["nan"])
        json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
