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

    def test_all_functions_sources_and_native_route_are_local(self):
        for number in range(1, 16):
            matches = list((ROOT / "features").glob(f"feature{number:02d}_*"))
            self.assertEqual(len(matches), 1, f"feature{number:02d} source folder")
        layout = (ROOT / "common" / "snowball" / "layout.py").read_text(encoding="utf-8")
        self.assertIn('target = f"/functions/{feature_id}"', layout)
        self.assertNotIn("streamlit.app", layout)
        route = (ROOT / "features" / "functions_nicegui" / "page.py").read_text(encoding="utf-8")
        self.assertIn('@ui.page("/functions/{feature_id}")', route)
        for feature_id in (
            "return_comparison", "periodic_returns", "portfolio_backtest",
            "dynamic_portfolio", "allocation_sweep", "monte_carlo_normal",
            "monte_carlo_bootstrap", "inflation_compass", "dollar_compass",
            "financial_dashboard", "moving_average", "laoer_infinite",
            "real_estate_vs_stock", "fear_greed_rebalance", "reb_statistics",
        ):
            self.assertIn(f'"{feature_id}"', route)


if __name__ == "__main__":
    unittest.main()
