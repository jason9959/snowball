"""Built-in preset library and UI-neutral backtest execution."""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .data.market_data import load_strategy_market_data
from .data.prices import PriceBundle, download_yahoo_prices
from .domestic import domestic_symbols
from .presets import PRESETS, PortfolioPreset
from .service import (
    run_blended_presets,
    run_domestic_blended_presets,
    run_domestic_preset,
    run_preset,
)
from .user_presets import UserPreset


@dataclass(frozen=True)
class BuiltinPresetSpec:
    title: str
    primary: PortfolioPreset
    secondary: PortfolioPreset | None = None
    primary_weight: float = 1.0
    domestic_execution: bool = False


_P1 = PRESETS["1. 분산투자 방어형"]
_P2 = PRESETS["2. 집중투자 공격형"]
_P7 = PRESETS["7. 집중투자 공격형 1-3-6-9-12"]
_P9 = PRESETS["9. MDD 개선 공격형"]

BUILTIN_PRESET_SPECS: dict[str, BuiltinPresetSpec] = {
    spec.title: spec
    for spec in (
        BuiltinPresetSpec(_P1.name, _P1),
        BuiltinPresetSpec(_P2.name, _P2),
        BuiltinPresetSpec("3. 프리셋 결합", _P1, _P2, 0.5),
        BuiltinPresetSpec("4. 국내 분산투자 방어형", _P1, domestic_execution=True),
        BuiltinPresetSpec("5. 국내 집중투자 공격형", _P2, domestic_execution=True),
        BuiltinPresetSpec("6. 국내 프리셋 결합", _P1, _P2, 0.5, True),
        BuiltinPresetSpec(_P7.name, _P7),
        BuiltinPresetSpec("8. 프리셋 1 + 7 결합", _P1, _P7, 0.5),
        BuiltinPresetSpec(_P9.name, _P9),
        BuiltinPresetSpec("10. 프리셋 1 + 9 결합", _P1, _P9, 0.5),
    )
}


def _attach_data_snapshot(
    result: dict,
    bundle: PriceBundle,
    market: dict[str, pd.Series],
    market_data_mode: str,
) -> None:
    price_hash = hashlib.sha256(
        pd.util.hash_pandas_object(bundle.adjusted_close, index=True).values.tobytes()
        + pd.util.hash_pandas_object(bundle.adjusted_open, index=True).values.tobytes()
    ).hexdigest()[:12]
    result["data_snapshot"] = {
        "price_source": bundle.source,
        "price_snapshot_id": f"yahoo-{price_hash}",
        "market_data_mode": market_data_mode,
        "market_snapshots": {
            name: series.attrs.get("snapshot_id", "-") for name, series in market.items()
        },
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def get_builtin_spec(title: str) -> BuiltinPresetSpec:
    try:
        return BUILTIN_PRESET_SPECS[title]
    except KeyError as error:
        raise ValueError(f"알 수 없는 기본 프리셋: {title}") from error


def run_builtin_backtest(
    title: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    initial_investment: float,
    transaction_cost_rate: float = 0.0,
    market_data_mode: str = "fixed",
    primary_weight: float | None = None,
    price_loader: Callable[[tuple[str, ...], pd.Timestamp, pd.Timestamp], PriceBundle] = download_yahoo_prices,
    market_loader: Callable[..., dict[str, pd.Series]] = load_strategy_market_data,
) -> tuple[dict, PriceBundle]:
    spec = get_builtin_spec(title)
    if start_date >= end_date:
        raise ValueError("시작일은 종료일보다 앞서야 합니다.")
    weight = spec.primary_weight if primary_weight is None else float(primary_weight)
    if spec.secondary is not None and not 0.0 <= weight <= 1.0:
        raise ValueError("프리셋 1 비중은 0~1 범위여야 합니다.")

    warmup = pd.Timestamp(start_date) - pd.DateOffset(months=18)
    secondary_symbols = spec.secondary.symbols if spec.secondary else ()
    symbols = tuple(dict.fromkeys(spec.primary.symbols + secondary_symbols + ("QQQ",)))
    signal_bundle = price_loader(symbols, warmup, pd.Timestamp(end_date))
    secondary_signals = spec.secondary.market_signals if spec.secondary else ()
    signals = tuple(dict.fromkeys(spec.primary.market_signals + secondary_signals))
    signal_end = (pd.Timestamp(end_date).to_period("M") - 1).to_timestamp("M")
    market = market_loader(
        signals,
        warmup.date().isoformat(),
        end_date.isoformat(),
        mode=market_data_mode,
        signal_end_iso=signal_end.date().isoformat(),
    )

    if spec.domestic_execution:
        execution_us = tuple(dict.fromkeys(
            spec.primary.attack_tickers + spec.primary.defense_tickers
            + (() if spec.secondary is None else (
                spec.secondary.attack_tickers + spec.secondary.defense_tickers
            ))
            + ("QQQ",)
        ))
        execution_bundle = price_loader(
            domestic_symbols(execution_us), warmup, pd.Timestamp(end_date),
        )
        if spec.secondary is None:
            result = run_domestic_preset(
                spec.primary, signal_bundle, execution_bundle, market,
                start_date, end_date, initial_investment=initial_investment,
                transaction_cost_rate=transaction_cost_rate, preset_name=spec.title,
            )
        else:
            result = run_domestic_blended_presets(
                spec.primary, spec.secondary, weight, signal_bundle,
                execution_bundle, market, start_date, end_date,
                initial_investment=initial_investment,
                transaction_cost_rate=transaction_cost_rate,
            )
            result["preset_name"] = spec.title
        _attach_data_snapshot(result, execution_bundle, market, market_data_mode)
        return result, execution_bundle

    if spec.secondary is None:
        result = run_preset(
            spec.primary, signal_bundle, market, start_date, end_date,
            initial_investment=initial_investment,
            transaction_cost_rate=transaction_cost_rate,
        )
        _attach_data_snapshot(result, signal_bundle, market, market_data_mode)
        return result, signal_bundle
    second_label = (
        "프리셋 7" if spec.secondary.name == _P7.name
        else "프리셋 9" if spec.secondary.name == _P9.name
        else "프리셋 2"
    )
    result = run_blended_presets(
        spec.primary, spec.secondary, weight, signal_bundle, market,
        start_date, end_date, initial_investment=initial_investment,
        transaction_cost_rate=transaction_cost_rate,
        preset_name=spec.title, second_label=second_label,
    )
    _attach_data_snapshot(result, signal_bundle, market, market_data_mode)
    return result, signal_bundle


def run_user_backtest(
    record: UserPreset,
    start_date: dt.date,
    end_date: dt.date,
    *,
    initial_investment: float,
    transaction_cost_rate: float = 0.0,
    market_data_mode: str = "fixed",
    price_loader: Callable[[tuple[str, ...], pd.Timestamp, pd.Timestamp], PriceBundle] = download_yahoo_prices,
    market_loader: Callable[..., dict[str, pd.Series]] = load_strategy_market_data,
) -> tuple[dict, PriceBundle]:
    """Run one portable user preset through the same engine as built-ins."""
    record.validate()
    if start_date >= end_date:
        raise ValueError("시작일은 종료일보다 앞서야 합니다.")
    warmup = pd.Timestamp(start_date) - pd.DateOffset(months=18)
    symbols = tuple(dict.fromkeys(record.strategy.symbols + ("QQQ",)))
    signal_bundle = price_loader(symbols, warmup, pd.Timestamp(end_date))
    signal_end = (pd.Timestamp(end_date).to_period("M") - 1).to_timestamp("M")
    market = market_loader(
        record.strategy.market_signals,
        warmup.date().isoformat(), end_date.isoformat(), mode=market_data_mode,
        signal_end_iso=signal_end.date().isoformat(),
    )
    if not record.domestic_execution:
        result = run_preset(
            record.strategy, signal_bundle, market, start_date, end_date,
            initial_investment=initial_investment,
            transaction_cost_rate=transaction_cost_rate,
        )
        result["preset_name"] = record.title
        _attach_data_snapshot(result, signal_bundle, market, market_data_mode)
        return result, signal_bundle

    execution_us = tuple(dict.fromkeys(
        record.strategy.attack_tickers + record.strategy.defense_tickers + ("QQQ",)
    ))
    execution_bundle = price_loader(
        domestic_symbols(execution_us), warmup, pd.Timestamp(end_date),
    )
    result = run_domestic_preset(
        record.strategy, signal_bundle, execution_bundle, market,
        start_date, end_date, initial_investment=initial_investment,
        transaction_cost_rate=transaction_cost_rate, preset_name=record.title,
    )
    _attach_data_snapshot(result, execution_bundle, market, market_data_mode)
    return result, execution_bundle
