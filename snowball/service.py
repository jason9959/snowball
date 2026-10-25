"""Application service connecting validated settings, supplied data and engine."""

from __future__ import annotations

import datetime as dt
from typing import Mapping

import pandas as pd

from .core.engine import run_backtest
from .data.prices import PriceBundle
from .metrics import summarize_nav
from .presets import PortfolioPreset


def run_preset(
    preset: PortfolioPreset,
    prices: PriceBundle,
    market_data: Mapping[str, pd.Series],
    start_date: dt.date,
    end_date: dt.date,
    *,
    initial_investment: float = 10_000.0,
    transaction_cost_rate: float = 0.0,
) -> dict:
    if start_date >= end_date:
        raise ValueError("시작일은 종료일보다 앞서야 합니다.")
    required = tuple(dict.fromkeys(preset.symbols + (("BIL",) if preset.cash_replacement_ticker else ())))
    prices.validate(required)
    config = preset.engine_config(
        start_date, end_date, initial_investment=initial_investment,
        transaction_cost_rate=transaction_cost_rate,
    )
    config["execution_prices"] = prices.adjusted_open
    config["_market_data"] = dict(market_data)
    result = run_backtest(prices.adjusted_close, config)
    result["metrics"] = summarize_nav(result["nav"])
    result["preset_name"] = preset.name
    result["engine_config"] = config
    return result
