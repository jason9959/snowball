import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snowball.settings import google_sheet_settings, runtime_secret


class RuntimeSettingsTests(unittest.TestCase):
    def test_environment_value_has_priority(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, patch.dict(
            os.environ, {"GOOGLE_SHEET_ID": "from-environment"}
        ):
            root = Path(folder)
            (root / ".streamlit").mkdir()
            (root / ".streamlit" / "secrets.toml").write_text(
                'GOOGLE_SHEET_ID = "from-file"\n', encoding="utf-8"
            )
            self.assertEqual(runtime_secret("GOOGLE_SHEET_ID", root=root), "from-environment")

    def test_local_streamlit_secrets_are_reused(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, patch.dict(
            os.environ, {}, clear=True
        ):
            root = Path(folder)
            (root / ".streamlit").mkdir()
            (root / ".streamlit" / "secrets.toml").write_text(
                'GOOGLE_SHEET_ID = "sheet-id"\n'
                'GOOGLE_SERVICE_ACCOUNT_JSON = "service-json"\n',
                encoding="utf-8",
            )
            self.assertEqual(google_sheet_settings(root=root), ("sheet-id", "service-json"))


if __name__ == "__main__":
    unittest.main()
