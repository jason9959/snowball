import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModularLayoutTests(unittest.TestCase):
    def test_numbered_snowball_feature_packages_exist(self):
        expected = (
            "snowball01_individual_backtest",
            "snowball02_portfolio_mix",
            "snowball03_account_dashboard",
        )
        for name in expected:
            with self.subTest(feature=name):
                folder = ROOT / "features" / name
                self.assertTrue((folder / "__init__.py").is_file())
                self.assertTrue((folder / "page.py").is_file())

    def test_entry_point_only_registers_features_and_starts_server(self):
        entry = ROOT / "nicegui_app.py"
        source = entry.read_text(encoding="utf-8")
        tree = ast.parse(source)
        page_functions = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == "page"
                for item in node.decorator_list
            )
        ]
        self.assertEqual(page_functions, [])
        self.assertLessEqual(len(source.splitlines()), 45)
        for name in (
            "snowball01_individual_backtest",
            "snowball02_portfolio_mix",
            "snowball03_account_dashboard",
        ):
            self.assertIn(name, source)

    def test_snowball_common_is_namespaced(self):
        common = ROOT / "common" / "snowball"
        for name in ("layout.py", "styles.py", "backtest_results.py", "preset_store.py"):
            self.assertTrue((common / name).is_file())


if __name__ == "__main__":
    unittest.main()
