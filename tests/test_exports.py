import unittest
from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from snowball.exports import diagnostic_frames, diagnostic_zip, monthly_return_export


class ExportTests(unittest.TestCase):
    def setUp(self):
        index = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-29"])
        self.result = {
            "nav": pd.Series([100.0, 110.0, 99.0], index=index),
            "history": pd.DataFrame([{"적용일": "2024-02-01", "종목": "QQQ"}]),
            "candidate_snapshots": {"2024-02-01": pd.DataFrame([{"티커": "QQQ", "계산값": 0.1}])},
            "canary_audits": {"2024-02-01": pd.DataFrame([{"티커": "TIP", "원시값": 0.01}])},
            "market_audits": {"2024-02-01": pd.DataFrame([{"시장 신호": "배당", "원시값": 1.4}])},
        }

    def test_monthly_return_export_keeps_raw_and_percent_values(self):
        frame = monthly_return_export(self.result["nav"])
        self.assertEqual(frame["적용 월"].tolist(), ["2024-02"])
        self.assertAlmostEqual(frame.iloc[0]["수익률 원시값"], -0.1)
        self.assertAlmostEqual(frame.iloc[0]["수익률 (%)"], -10.0)

    def test_diagnostic_frames_include_applied_date(self):
        frames = diagnostic_frames(self.result)
        self.assertEqual(frames["03_asset_signals.csv"].iloc[0]["리밸런싱 적용일"], "2024-02-01")
        self.assertIn("낙폭", frames["06_daily_nav_drawdown.csv"].columns)

    def test_zip_contains_all_diagnostic_csv_files(self):
        with ZipFile(BytesIO(diagnostic_zip(self.result))) as archive:
            self.assertEqual(set(archive.namelist()), set(diagnostic_frames(self.result)))


if __name__ == "__main__":
    unittest.main()
