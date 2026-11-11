"""Short-lived in-memory caches used only by Snowball backtests.

Do not merge this with the account dashboard cache: backtests require adjusted
open/close prices, while account valuation intentionally downloads unadjusted
closes and FX data.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Callable, Hashable, TypeVar

import pandas as pd

from snowball.data.prices import PriceBundle, download_yahoo_prices


PRICE_CACHE_SECONDS = 900
RESULT_CACHE_SECONDS = 1800
MAX_PRICE_ENTRIES = 8
MAX_RESULT_ENTRIES = 4

T = TypeVar("T")

_lock = Lock()
_prices: OrderedDict[
    tuple[tuple[str, ...], str, str], tuple[float, PriceBundle]
] = OrderedDict()
_results: OrderedDict[Hashable, tuple[float, object]] = OrderedDict()


def _trim(cache: OrderedDict, maximum: int) -> None:
    while len(cache) > maximum:
        cache.popitem(last=False)


def cached_download_yahoo_prices(
    symbols: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp,
) -> PriceBundle:
    normalized = tuple(dict.fromkeys(symbols))
    key = (normalized, pd.Timestamp(start).isoformat(), pd.Timestamp(end).isoformat())
    now = time.monotonic()
    with _lock:
        cached = _prices.get(key)
        if cached and now - cached[0] < PRICE_CACHE_SECONDS:
            _prices.move_to_end(key)
            return cached[1]
    bundle = download_yahoo_prices(normalized, pd.Timestamp(start), pd.Timestamp(end))
    with _lock:
        _prices[key] = (time.monotonic(), bundle)
        _prices.move_to_end(key)
        _trim(_prices, MAX_PRICE_ENTRIES)
    return bundle


def cached_result(key: Hashable, factory: Callable[[], T]) -> tuple[T, bool]:
    """Return ``(value, cache_hit)`` for an immutable-by-convention result."""
    now = time.monotonic()
    with _lock:
        cached = _results.get(key)
        if cached and now - cached[0] < RESULT_CACHE_SECONDS:
            _results.move_to_end(key)
            return cached[1], True  # type: ignore[return-value]
    value = factory()
    with _lock:
        _results[key] = (time.monotonic(), value)
        _results.move_to_end(key)
        _trim(_results, MAX_RESULT_ENTRIES)
    return value, False


def clear_backtest_caches() -> None:
    with _lock:
        _prices.clear()
        _results.clear()
