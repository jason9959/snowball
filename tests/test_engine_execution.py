import datetime as dt
import unittest

import pandas as pd

from snowball.core.engine import MomentumConfig, run_backtest


class EngineExecutionTests(unittest.TestCase):
    def test_rebalance_open_to_close_move_is_counted_once(self):
        index = pd.DatetimeIndex([
            "2019-11-29", "2019-12-02", "2019-12-31",
            "2020-01-02", "2020-01-03", "2020-02-03",
        ])
        prices = pd.DataFrame({"SPY": [80.0, 90.0, 100.0, 110.0, 121.0, 133.1]}, index=index)
        adjusted_open = pd.DataFrame({"SPY": [80.0, 90.0, 100.0, 100.0, 110.0, 121.0]}, index=index)
        momentum = MomentumConfig(kind="수익률", months=1)
        config = {
            "start_date": dt.date(2020, 1, 2), "end_date": dt.date(2020, 2, 3),
            "initial_investment": 10_000.0, "rebalance_frequency": "월별",
            "contribution_amount": 0.0, "contribution_frequency": "매월",
            "attack_tickers": ["SPY"], "defense_tickers": [],
            "attack_mode": "상대 모멘텀", "attack_select_count": 1,
            "attack_relative": momentum, "attack_absolute": momentum, "attack_use_bil": False,
            "defense_enabled": False, "defense_mode": "전체 동일 비중", "defense_select_count": 1,
            "defense_relative": momentum, "defense_absolute": momentum, "defense_use_bil": False,
            "include_cash": True, "weight_mode": "고정 범위", "fixed_defense_weight": 0.0,
            "negative_threshold": 1, "negative_same_momentum": True, "positive_same_momentum": True,
            "risk_avoidance": {"enabled": False}, "market_risk": {"enabled": False, "signals": []},
            "cash_replacement_ticker": None, "bil_ticker": "BIL",
            "use_transaction_cost": False, "transaction_cost_rate": 0.0,
            "execution_prices": adjusted_open,
        }
        result = run_backtest(prices, config)
        self.assertAlmostEqual(float(result["values"].iloc[-1]), 13_310.0)
        self.assertAlmostEqual(float(result["nav"].iloc[-1]), 1.331)


if __name__ == "__main__":
    unittest.main()
