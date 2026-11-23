import unittest
from unittest.mock import Mock, patch
import sys
import types

try:
    import requests
except ModuleNotFoundError:  # lightweight Codex test runtime
    requests = types.ModuleType("requests")
    requests.RequestException = type("RequestException", (Exception,), {})
    requests.get = Mock()
    sys.modules["requests"] = requests

from common.snowball.symbols import search_symbols


class SymbolSearchTests(unittest.TestCase):
    @patch("common.snowball.symbols.requests.get")
    def test_us_direct_ticker_remains_available_when_yahoo_fails(self, get: Mock) -> None:
        get.side_effect = requests.RequestException("offline")
        self.assertEqual(search_symbols("qqq", "미국"), {"QQQ": "QQQ · 티커 직접 사용"})

    @patch("common.snowball.symbols.requests.get")
    def test_search_filters_results_by_country(self, get: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "quotes": [
                {"symbol": "SPY", "exchange": "PCX", "quoteType": "ETF", "longname": "SPDR S&P 500 ETF"},
                {"symbol": "360750.KS", "exchange": "KSC", "quoteType": "ETF", "longname": "TIGER 미국S&P500"},
            ]
        }
        get.return_value = response
        self.assertIn("SPY", search_symbols("S&P500", "미국"))
        self.assertNotIn("360750.KS", search_symbols("S&P500", "미국"))
        self.assertIn("360750.KS", search_symbols("S&P500", "한국"))


if __name__ == "__main__":
    unittest.main()
