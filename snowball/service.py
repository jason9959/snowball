"""Application service connecting validated settings, supplied data and engine."""

from __future__ import annotations

import datetime as dt
from typing import Mapping

import pandas as pd

from .core.engine import run_backtest
from .data.prices import PriceBundle
from .domestic import DOMESTIC_EXECUTION_MAP, domestic_symbols
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


def _scaled_selection(selection: object, scale: float) -> dict[str, float]:
    """Convert an engine selection string into whole-portfolio weights."""
    weights: dict[str, float] = {}
    if not isinstance(selection, str):
        return weights
    for item in selection.split(", "):
        try:
            ticker, percent = item.rsplit(" ", 1)
            weight = float(percent.rstrip("%")) / 100.0 * scale
        except (ValueError, AttributeError):
            continue
        weights[ticker] = weights.get(ticker, 0.0) + weight
    return weights


def _format_selection(weights: Mapping[str, float]) -> str:
    return ", ".join(
        f"{ticker} {weight:.1%}" for ticker, weight in weights.items() if weight > 1e-12
    )


def _map_domestic_history(history: pd.DataFrame, execution_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map US decisions to domestic instruments and the next KRX session."""
    rows: list[dict[str, object]] = []
    for _, source in history.iterrows():
        us_weights = _scaled_selection(source.get("선택된 종목"), 1.0)
        domestic: dict[str, float] = {}
        for ticker, weight in us_weights.items():
            mapped = ticker if ticker == "현금" else DOMESTIC_EXECUTION_MAP.get(ticker)
            if mapped is None:
                raise ValueError(f"{ticker}의 국내 실행 ETF가 정해지지 않았습니다.")
            domestic[mapped] = domestic.get(mapped, 0.0) + weight
        signal_date = pd.Timestamp(source["적용일"])
        position = execution_index.searchsorted(signal_date, side="right")
        if position >= len(execution_index):
            continue
        row = source.to_dict()
        row["미국 신호 종목"] = source.get("선택된 종목", "-")
        row["선택된 종목"] = _format_selection(domestic)
        row["미국 적용일"] = str(signal_date.date())
        row["적용일"] = str(execution_index[position].date())
        rows.append(row)
    mapped = pd.DataFrame(rows)
    if mapped.empty:
        raise ValueError("입력 기간에 국내 ETF로 실행할 리밸런싱 신호가 없습니다.")
    return mapped.drop_duplicates(subset=["적용일"], keep="last").reset_index(drop=True)


def _domestic_common_start(
    history: pd.DataFrame, prices: PriceBundle, requested_start: dt.date,
) -> pd.Timestamp:
    used = sorted({
        ticker for selection in history["선택된 종목"]
        for ticker in _scaled_selection(selection, 1.0) if ticker != "현금"
    })
    if not used:
        return pd.Timestamp(requested_start)
    first_dates = []
    for ticker in used:
        first = prices.adjusted_close[ticker].first_valid_index()
        if first is None:
            raise ValueError(f"{ticker}의 국내 가격 데이터가 없습니다.")
        first_dates.append(pd.Timestamp(first))
    return max(pd.Timestamp(requested_start), max(first_dates))


def _run_domestic_from_signal_result(
    signal_result: dict,
    execution_prices: PriceBundle,
    start_date: dt.date,
    end_date: dt.date,
    initial_investment: float,
    transaction_cost_rate: float,
    preset_name: str,
) -> dict:
    history = _map_domestic_history(signal_result["history"], execution_prices.adjusted_close.index)
    common_start = max(
        _domestic_common_start(history, execution_prices, start_date),
        pd.to_datetime(history["적용일"]).min(),
    )
    evaluation_index = execution_prices.adjusted_close.index[
        (execution_prices.adjusted_close.index >= common_start)
        & (execution_prices.adjusted_close.index <= pd.Timestamp(end_date))
    ]
    if evaluation_index.empty:
        raise ValueError("국내 ETF의 공통 거래 가능 기간이 선택한 종료일 이후입니다.")
    prior = history[pd.to_datetime(history["적용일"]) <= evaluation_index[0]]
    future = history[pd.to_datetime(history["적용일"]) > evaluation_index[0]]
    if prior.empty:
        first = history.iloc[0].copy()
    else:
        first = prior.iloc[-1].copy()
    first["적용일"] = str(evaluation_index[0].date())
    history = pd.concat([pd.DataFrame([first]), future], ignore_index=True)
    nav, values, history, turnover = _simulate_blended_portfolio(
        history, execution_prices, evaluation_index, initial_investment, transaction_cost_rate,
    )
    drawdown = (nav / nav.cummax() - 1.0).rename("포트폴리오")
    final_weights = _scaled_selection(history.iloc[-1]["선택된 종목"], 1.0)
    result = dict(signal_result)
    result.update({
        "values": values,
        "invested": pd.Series(initial_investment, index=nav.index, name="누적 투입금"),
        "flows": pd.Series(0.0, index=nav.index, name="외부 현금흐름"),
        "nav": nav,
        "drawdown": drawdown,
        "history": history,
        "final_weights": final_weights,
        "annual_turnover": turnover,
        "common_start": nav.index[0],
        "common_end": nav.index[-1],
        "preset_name": preset_name,
        "metrics": summarize_nav(nav),
        "benchmark_ticker": DOMESTIC_EXECUTION_MAP["QQQ"],
    })
    config = dict(signal_result["engine_config"])
    config["실행 시장"] = "한국거래소 국내 상장 ETF"
    config["체결 규칙"] = "미국 신호 적용일 다음 국내 거래일 수정시가"
    config["국내 실행 매핑"] = dict(DOMESTIC_EXECUTION_MAP)
    result["engine_config"] = config
    return result


def _blend_history(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_weight: float,
    first_label: str = "프리셋 1",
    second_label: str = "프리셋 2",
) -> pd.DataFrame:
    second_weight = 1.0 - first_weight
    left = first.add_prefix(f"{first_label} · ")
    right = second.add_prefix(f"{second_label} · ")
    left["적용일"] = first["적용일"].astype(str).to_numpy()
    right["적용일"] = second["적용일"].astype(str).to_numpy()
    merged = left.merge(right, on="적용일", how="outer", sort=True)
    rows: list[dict[str, object]] = []
    for _, source in merged.iterrows():
        combined: dict[str, float] = {}
        for ticker, weight in _scaled_selection(source.get(f"{first_label} · 선택된 종목"), first_weight).items():
            combined[ticker] = combined.get(ticker, 0.0) + weight
        for ticker, weight in _scaled_selection(source.get(f"{second_label} · 선택된 종목"), second_weight).items():
            combined[ticker] = combined.get(ticker, 0.0) + weight
        selected = ", ".join(
            f"{ticker} {weight:.1%}" for ticker, weight in combined.items() if weight > 1e-12
        )
        rows.append({
            "예정일": source.get(f"{first_label} · 예정일", source.get(f"{second_label} · 예정일")),
            "적용일": source["적용일"],
            "판단 기준일": source.get(f"{first_label} · 판단 기준일", source.get(f"{second_label} · 판단 기준일")),
            "선택된 종목": selected,
            f"{first_label} 비중": first_weight,
            f"{first_label} 선택": source.get(f"{first_label} · 선택된 종목", "-"),
            f"{second_label} 비중": second_weight,
            f"{second_label} 선택": source.get(f"{second_label} · 선택된 종목", "-"),
            "턴오버": (
                first_weight * float(source.get(f"{first_label} · 턴오버", 0.0) or 0.0)
                + second_weight * float(source.get(f"{second_label} · 턴오버", 0.0) or 0.0)
            ),
            "거래비용": (
                first_weight * float(source.get(f"{first_label} · 거래비용", 0.0) or 0.0)
                + second_weight * float(source.get(f"{second_label} · 거래비용", 0.0) or 0.0)
            ),
            "평가금액": (
                first_weight * float(source.get(f"{first_label} · 평가금액", 0.0) or 0.0)
                + second_weight * float(source.get(f"{second_label} · 평가금액", 0.0) or 0.0)
            ),
        })
    return pd.DataFrame(rows)


def _blend_snapshots(
    first: dict[str, pd.DataFrame], second: dict[str, pd.DataFrame],
    first_label: str = "프리셋 1", second_label: str = "프리셋 2",
) -> dict[str, pd.DataFrame]:
    combined: dict[str, pd.DataFrame] = {}
    for date in sorted(set(first) | set(second)):
        frames = []
        for label, source in ((first_label, first), (second_label, second)):
            frame = source.get(date)
            if frame is None or frame.empty:
                continue
            copy = frame.copy()
            copy.insert(0, "구성 전략", label)
            frames.append(copy)
        combined[date] = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined


def _display_config(config: dict) -> dict:
    copy = dict(config)
    copy.pop("execution_prices", None)
    copy.pop("_market_data", None)
    return copy


def _simulate_blended_portfolio(
    history: pd.DataFrame,
    prices: PriceBundle,
    evaluation_index: pd.DatetimeIndex,
    initial_investment: float,
    transaction_cost_rate: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, float]:
    """Trade the merged target weights at adjusted open and value at adjusted close."""
    targets: dict[pd.Timestamp, dict[str, float]] = {}
    for _, row in history.iterrows():
        targets[pd.Timestamp(row["적용일"])] = _scaled_selection(row["선택된 종목"], 1.0)
    used_tickers = sorted({ticker for weights in targets.values() for ticker in weights if ticker != "현금"})
    close = prices.adjusted_close.reindex(evaluation_index)[used_tickers].ffill()
    open_price = prices.adjusted_open.reindex(evaluation_index)[used_tickers].ffill()
    shares = pd.Series(0.0, index=used_tickers)
    cash = float(initial_investment)
    values: list[float] = []
    returns: list[float] = []
    turnover_sum = 0.0
    history = history.copy()
    row_by_date = {pd.Timestamp(value): index for index, value in history["적용일"].items()}

    for date in evaluation_index:
        if date in targets:
            target = targets[date]
            execution = open_price.loc[date]
            if execution.isna().any() or (execution <= 0).any():
                raise ValueError(f"{date.date()}의 결합 포트폴리오 수정 시가 데이터가 부족합니다.")
            current_values = shares * execution
            before_value = float(current_values.sum() + cash)
            current_weights = {
                ticker: float(value / before_value) for ticker, value in current_values.items()
            }
            current_weights["현금"] = cash / before_value if before_value else 0.0
            all_assets = set(current_weights) | set(target)
            turnover = 0.5 * sum(
                abs(target.get(asset, 0.0) - current_weights.get(asset, 0.0))
                for asset in all_assets
            )
            if date != evaluation_index[0]:
                turnover_sum += turnover
            gross_trade = sum(
                abs(before_value * target.get(ticker, 0.0) - current_values.get(ticker, 0.0))
                for ticker in used_tickers
            )
            fee = gross_trade * transaction_cost_rate
            investable = max(0.0, before_value - fee)
            for ticker in used_tickers:
                shares[ticker] = investable * target.get(ticker, 0.0) / float(execution[ticker])
            cash = investable * target.get("현금", 0.0)
            after_value = float((shares * close.loc[date]).sum() + cash)
            history_index = row_by_date[date]
            history.at[history_index, "턴오버"] = 0.0 if date == evaluation_index[0] else turnover
            history.at[history_index, "거래비용"] = fee
            history.at[history_index, "평가금액"] = after_value
        else:
            after_value = float((shares * close.loc[date]).sum() + cash)
        period_return = (
            after_value / initial_investment - 1.0
            if not values else after_value / values[-1] - 1.0
        )
        values.append(after_value)
        returns.append(period_return)

    value_series = pd.Series(values, index=evaluation_index, name="포트폴리오 평가액")
    nav = (1.0 + pd.Series(returns, index=evaluation_index).fillna(0.0)).cumprod().rename("포트폴리오")
    years = max((evaluation_index[-1] - evaluation_index[0]).days / 365.25, 1 / 365.25)
    return nav, value_series, history, turnover_sum / years


def run_blended_presets(
    first: PortfolioPreset,
    second: PortfolioPreset,
    first_weight: float,
    prices: PriceBundle,
    market_data: Mapping[str, pd.Series],
    start_date: dt.date,
    end_date: dt.date,
    *,
    initial_investment: float = 10_000.0,
    transaction_cost_rate: float = 0.0,
    preset_name: str = "3. 프리셋 결합",
    first_label: str = "프리셋 1",
    second_label: str = "프리셋 2",
) -> dict:
    """Run two independent sleeves and combine their daily values and holdings."""
    if not 0 <= first_weight <= 1:
        raise ValueError("프리셋 1 비중은 0% 이상 100% 이하여야 합니다.")
    second_weight = 1.0 - first_weight
    first_result = run_preset(
        first, prices, market_data, start_date, end_date,
        initial_investment=initial_investment,
        transaction_cost_rate=transaction_cost_rate,
    )
    second_result = run_preset(
        second, prices, market_data, start_date, end_date,
        initial_investment=initial_investment,
        transaction_cost_rate=transaction_cost_rate,
    )
    navs = pd.concat(
        [first_result["nav"].rename("first"), second_result["nav"].rename("second")],
        axis=1, join="inner",
    ).dropna()
    history = _blend_history(
        first_result["history"], second_result["history"], first_weight,
        first_label, second_label,
    )
    nav, blended_values, history, annual_turnover = _simulate_blended_portfolio(
        history, prices, navs.index, initial_investment, transaction_cost_rate,
    )
    drawdown = (nav / nav.cummax() - 1.0).rename("포트폴리오")
    final_weights: dict[str, float] = {}
    for scale, result in ((first_weight, first_result), (second_weight, second_result)):
        for ticker, weight in result["final_weights"].items():
            final_weights[ticker] = final_weights.get(ticker, 0.0) + scale * weight

    result = {
        "values": blended_values,
        "invested": pd.Series(initial_investment, index=nav.index, name="누적 투입금"),
        "flows": pd.Series(0.0, index=nav.index, name="외부 현금흐름"),
        "nav": nav,
        "drawdown": drawdown,
        "history": history,
        "candidate_snapshots": _blend_snapshots(
            first_result["candidate_snapshots"], second_result["candidate_snapshots"],
            first_label, second_label,
        ),
        "canary_audits": _blend_snapshots(
            first_result["canary_audits"], second_result["canary_audits"],
            first_label, second_label,
        ),
        "market_audits": _blend_snapshots(
            first_result["market_audits"], second_result["market_audits"],
            first_label, second_label,
        ),
        "final_weights": final_weights,
        "annual_turnover": annual_turnover,
        "common_start": nav.index[0],
        "common_end": nav.index[-1],
        "preset_name": preset_name,
        "sleeve_labels": (first_label, second_label),
        "metrics": summarize_nav(nav),
        "engine_config": {
            "전략": f"{first_label} + {second_label} 혼합",
            f"{first_label} 비중": first_weight,
            f"{second_label} 비중": second_weight,
            first_label: _display_config(first_result["engine_config"]),
            second_label: _display_config(second_result["engine_config"]),
            "_market_data": dict(market_data),
        },
        "component_results": {first_label: first_result, second_label: second_result},
    }
    return result


def run_domestic_preset(
    preset: PortfolioPreset,
    signal_prices: PriceBundle,
    execution_prices: PriceBundle,
    market_data: Mapping[str, pd.Series],
    start_date: dt.date,
    end_date: dt.date,
    *,
    initial_investment: float = 10_000_000.0,
    transaction_cost_rate: float = 0.0,
    preset_name: str = "4. 국내 분산투자 방어형",
) -> dict:
    """Calculate signals in USD and execute the selected domestic-listed ETFs."""
    domestic_symbols(preset.attack_tickers + preset.defense_tickers)
    signal_result = run_preset(
        preset, signal_prices, market_data, start_date, end_date,
        initial_investment=initial_investment,
        transaction_cost_rate=transaction_cost_rate,
    )
    return _run_domestic_from_signal_result(
        signal_result, execution_prices, start_date, end_date,
        initial_investment, transaction_cost_rate, preset_name,
    )


def run_domestic_blended_presets(
    first: PortfolioPreset,
    second: PortfolioPreset,
    first_weight: float,
    signal_prices: PriceBundle,
    execution_prices: PriceBundle,
    market_data: Mapping[str, pd.Series],
    start_date: dt.date,
    end_date: dt.date,
    *,
    initial_investment: float = 10_000_000.0,
    transaction_cost_rate: float = 0.0,
) -> dict:
    """Domestic execution counterpart of preset 3."""
    domestic_symbols(
        first.attack_tickers + first.defense_tickers
        + second.attack_tickers + second.defense_tickers
    )
    signal_result = run_blended_presets(
        first, second, first_weight, signal_prices, market_data, start_date, end_date,
        initial_investment=initial_investment,
        transaction_cost_rate=transaction_cost_rate,
    )
    return _run_domestic_from_signal_result(
        signal_result, execution_prices, start_date, end_date,
        initial_investment, transaction_cost_rate, "6. 국내 프리셋 결합",
    )
