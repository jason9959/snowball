"""Execution service for Snowball 02."""

from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd

from common.snowball.backtest_cache import cached_download_yahoo_prices, cached_result
from snowball.builtin_library import run_mixed_backtest
from snowball.domestic import domestic_symbols
from snowball.user_presets import UserPreset, export_user_presets


def prepare_mix_prices(
    first: UserPreset, second: UserPreset, start_date: dt.date, end_date: dt.date,
) -> None:
    """Populate shared signal and optional domestic execution price caches."""
    warmup = pd.Timestamp(start_date) - pd.DateOffset(months=18)
    symbols = tuple(dict.fromkeys(
        first.strategy.symbols + second.strategy.symbols + ("QQQ",)
    ))
    cached_download_yahoo_prices(symbols, warmup, pd.Timestamp(end_date))
    if first.domestic_execution and second.domestic_execution:
        execution_us = tuple(dict.fromkeys(
            first.strategy.attack_tickers + first.strategy.defense_tickers
            + second.strategy.attack_tickers + second.strategy.defense_tickers
            + ("QQQ",)
        ))
        cached_download_yahoo_prices(
            domestic_symbols(execution_us), warmup, pd.Timestamp(end_date),
        )


def run_portfolio_mix(
    first: UserPreset,
    second: UserPreset,
    first_weight: float,
    start_date: dt.date,
    end_date: dt.date,
    *,
    initial_investment: float,
    transaction_cost_rate: float,
    market_data_mode: str,
) -> tuple[dict, object, bool]:
    fingerprint = hashlib.sha256(export_user_presets([first, second])).hexdigest()
    key = (
        "snowball02", fingerprint, float(first_weight), start_date.isoformat(),
        end_date.isoformat(), float(initial_investment), float(transaction_cost_rate),
        market_data_mode,
    )

    def calculate():
        return run_mixed_backtest(
            first, second, first_weight, start_date, end_date,
            initial_investment=initial_investment,
            transaction_cost_rate=transaction_cost_rate,
            market_data_mode=market_data_mode,
            price_loader=cached_download_yahoo_prices,
        )

    (result, bundle), hit = cached_result(key, calculate)
    return result, bundle, hit
