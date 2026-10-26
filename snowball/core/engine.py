"""Calculation engine for the dynamic portfolio backtest."""

from __future__ import annotations

import datetime as dt
import calendar
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CASH = "현금"
SCORE_PERIODS = {
    "1-3-6개월 단순": ((1, 3, 6), None),
    "3-6-12개월 단순": ((3, 6, 12), None),
    "1-3-6-12개월 단순": ((1, 3, 6, 12), None),
    "1-3-6-9-12개월 단순": ((1, 3, 6, 9, 12), None),
    "1-3-6-12개월 가중": ((1, 3, 6, 12), (12.0, 4.0, 2.0, 1.0)),
}
FREQUENCY_MONTHS = {"월별": 1, "분기": 3, "반기": 6, "연별": 12}
CONTRIBUTION_MONTHS = {"매월": 1, "매분기": 3, "매년": 12}
MARKET_SIGNAL_TESTED = {
    "장단기 금리차 역전 (10년-2년)": True,
    "S&P500 배당수익률": True,
    "장단기 금리차 역전 (10년-3개월)": True,
    "실업률": False,
    "실업률 & SPY (GTT-UE)": False,
    "삼의 법칙 (Sahm's Rule)": False,
    "삼의 법칙과 SPY 모멘텀": False,
}
# Calculated from the current research presets. Required-by-preset options
# stay selectable even when their standalone test coverage is incomplete.
MARKET_SIGNAL_REQUIRED_BY_PRESETS = {"S&P500 배당수익률", "장단기 금리차 역전 (10년-3개월)"}
MARKET_SIGNAL_SUPPORT = {
    signal: tested or signal in MARKET_SIGNAL_REQUIRED_BY_PRESETS
    for signal, tested in MARKET_SIGNAL_TESTED.items()
}


def normalize_market_signals(signals: list[str] | tuple[str, ...], *, allow_unverified: bool = False) -> list[str]:
    """Validate market signals at the engine boundary.

    The internal escape hatch is intentionally environment-driven and is not
    exposed in the normal Streamlit UI.
    """
    selected = list(dict.fromkeys(signals))
    unknown = [signal for signal in selected if signal not in MARKET_SIGNAL_SUPPORT]
    unsupported = [signal for signal in selected if not MARKET_SIGNAL_SUPPORT.get(signal, False)]
    if (unknown or (unsupported and not allow_unverified)):
        names = ", ".join(unknown + unsupported)
        raise ValueError(f"아직 검증되지 않아 일반 실행에서 지원하지 않는 시장 신호입니다: {names}")
    return selected


def validate_preset_market_signals(
    presets: dict[str, dict[str, Any]],
    *,
    required_by_presets: set[str] | None = None,
) -> None:
    """Fail fast if a research preset contains an unsupported signal."""
    for name, preset in presets.items():
        signals = preset.get("market_risk_signals", [])
        required = required_by_presets or set()
        invalid = [
            signal for signal in signals
            if not MARKET_SIGNAL_TESTED.get(signal, False) and signal not in required
        ]
        if invalid:
            raise ValueError(f"프리셋 {name}에 미검증 시장 신호가 포함되어 있습니다: {', '.join(invalid)}")


def resolve_recent_period(today: dt.date, years: int) -> tuple[dt.date, dt.date]:
    """Return a calendar-safe recent period without mutating UI dates."""
    try:
        start = today.replace(year=today.year - int(years))
    except ValueError:
        start = today.replace(year=today.year - int(years), day=28)
    return start, today


@dataclass(frozen=True)
class MomentumConfig:
    kind: str = "수익률"
    months: int = 12
    score_type: str = "1-3-6-12개월 단순"


def _asof_value(series: pd.Series, when: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    valid = series.loc[:when].dropna()
    if valid.empty:
        raise ValueError("비교 시점 이전의 가격 데이터가 없습니다.")
    return pd.Timestamp(valid.index[-1]), float(valid.iloc[-1])


def momentum_detail(
    series: pd.Series, signal_date: pd.Timestamp, config: MomentumConfig
) -> dict[str, Any]:
    """Return the score and intermediate values used for one asset."""
    series = series.dropna().sort_index()
    current_date, current = _asof_value(series, signal_date)
    detail: dict[str, Any] = {
        "판단일": current_date,
        "수정종가": current,
        "산출 기준": config.kind,
    }

    def period_return(months: int) -> tuple[float, pd.Timestamp, float]:
        base_date, base = _asof_value(series, signal_date - pd.DateOffset(months=months))
        return current / base - 1.0, base_date, base

    if config.kind == "수익률":
        value, base_date, base = period_return(config.months)
        detail.update({"기준 기간": f"{config.months}개월", "기준일": base_date, "기준가": base})
    elif config.kind == "이동평균":
        window_start = signal_date - pd.DateOffset(months=config.months)
        window = series.loc[window_start:signal_date].dropna()
        if window.empty:
            raise ValueError("이동평균을 계산할 가격 데이터가 없습니다.")
        average = float(window.mean())
        value = current / average - 1.0
        detail.update(
            {
                "기준 기간": f"{config.months}개월",
                "관측 수": int(len(window)),
                "이동평균": average,
                "P/MA": current / average,
            }
        )
    else:
        periods, weights = SCORE_PERIODS[config.score_type]
        returns = []
        for months in periods:
            period_value, _, _ = period_return(months)
            returns.append(period_value)
            detail[f"R{months}"] = period_value
        if weights is None:
            value = float(np.mean(returns))
        else:
            value = float(np.dot(np.asarray(returns), np.asarray(weights)))
            for months, weight, result in zip(periods, weights, returns):
                detail[f"{int(weight)}×R{months}"] = result * weight
        detail["기준 기간"] = config.score_type

    detail["계산값"] = float(value)
    return detail


def scheduled_trade_dates(
    index: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Map calendar-period starts to the first available market date.

    The schedule preserves the requested start-day-of-month. If that calendar
    day is a holiday or weekend, the first available session after it is used.
    Each actual session is constrained to its own planned period, so a later
    period's close can never be pulled backward into an earlier decision.
    """
    months = FREQUENCY_MONTHS[frequency]
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    anchor = start_date

    def add_months_preserving_day(value: pd.Timestamp, offset: int) -> pd.Timestamp:
        month_index = value.year * 12 + (value.month - 1) + offset
        year, month_zero_based = divmod(month_index, 12)
        month = month_zero_based + 1
        day = min(value.day, calendar.monthrange(year, month)[1])
        return pd.Timestamp(year=year, month=month, day=day)

    dates: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    step = 0
    while True:
        planned = add_months_preserving_day(anchor, months * step)
        if planned > end_date:
            break
        next_planned = add_months_preserving_day(anchor, months * (step + 1))
        candidates = index[(index >= planned) & (index < next_planned)]
        if len(candidates):
            actual = pd.Timestamp(candidates[0])
            if not dates or actual != dates[-1][1]:
                dates.append((planned, actual))
        step += 1
    return dates


def _ranked(
    values: dict[str, float], previous: set[str], registration_order: list[str]
) -> list[str]:
    order = {ticker: i for i, ticker in enumerate(registration_order)}
    return sorted(
        values,
        key=lambda ticker: (
            -values[ticker],
            0 if ticker in previous else 1,
            order.get(ticker, 10_000),
        ),
    )


def _absolute_pass(
    value: float, benchmark_value: float | None
) -> bool:
    threshold = benchmark_value if benchmark_value is not None else 0.0
    return value >= threshold


_ATTACK_ABSOLUTE_MODES = {
    "절대 모멘텀",
    "절대 모멘텀 + 방어 스위치",
    "상대 & 절대 모멘텀",
    "상대 & 절대 모멘텀 + 방어 스위치",
}


def _weight_momentum_config(config: dict[str, Any], mode: str) -> tuple[MomentumConfig, str]:
    """Choose the momentum calculation used by the portfolio-weight option."""
    same_key = "negative_same_momentum" if mode == "마이너스 절대 모멘텀 수" else "positive_same_momentum"
    if not config.get(same_key, True):
        custom_key = "negative_momentum" if mode == "마이너스 절대 모멘텀 수" else "positive_momentum"
        return config[custom_key], "독립 기준"
    if config.get("attack_mode") in _ATTACK_ABSOLUTE_MODES:
        return config["attack_absolute"], "투자자산 절대 기준"
    return config["attack_relative"], "투자자산 상대 기준"


def _evaluate_candidates(
    prices: pd.DataFrame,
    candidates: list[str],
    signal_date: pd.Timestamp,
    relative_config: MomentumConfig,
    absolute_config: MomentumConfig,
    use_bil: bool,
    bil_ticker: str,
    role: str,
) -> tuple[dict[str, dict[str, Any]], float | None]:
    rows: dict[str, dict[str, Any]] = {}
    bil_score = None
    if use_bil:
        bil_score = momentum_detail(prices[bil_ticker], signal_date, absolute_config)["계산값"]
    for ticker in candidates:
        relative = momentum_detail(prices[ticker], signal_date, relative_config)
        absolute = momentum_detail(prices[ticker], signal_date, absolute_config)
        rows[ticker] = {
            "역할": role,
            "티커": ticker,
            "판단 기준 종가": relative["수정종가"],
            "상대 계산값": relative["계산값"],
            "절대 계산값": absolute["계산값"],
            "BIL 기준값": bil_score,
            "데이터 유효": True,
            "상대 상세": relative,
            "절대 상세": absolute,
        }
    return rows, bil_score


def _attack_allocation(
    mode: str,
    amount: float,
    rows: dict[str, dict[str, Any]],
    select_count: int,
    previous: set[str],
    registration_order: list[str],
    bil_score: float | None,
) -> tuple[dict[str, float], float, list[dict[str, Any]]]:
    scores = {ticker: row["상대 계산값"] for ticker, row in rows.items()}
    ranked = _ranked(scores, previous, registration_order)
    for rank, ticker in enumerate(ranked, 1):
        rows[ticker]["순위"] = rank
    uses_relative = mode in {"상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치"}
    uses_absolute = mode != "상대 모멘텀"
    selected = ranked[:select_count] if uses_relative else list(registration_order)
    passed = [
        ticker for ticker in selected
        if not uses_absolute or _absolute_pass(rows[ticker]["절대 계산값"], bil_score)
    ]
    switched = mode in {"절대 모멘텀 + 방어 스위치", "상대 & 절대 모멘텀 + 방어 스위치"}
    weights: dict[str, float] = {}
    defense_amount = 0.0
    audit: list[dict[str, Any]] = []
    initial_amount = amount / len(selected) if selected else 0.0
    if switched:
        base = initial_amount
        for ticker in selected:
            if ticker in passed:
                weights[ticker] = base
            else:
                defense_amount += base
    elif passed:
        weights = {ticker: amount / len(passed) for ticker in passed}
    else:
        defense_amount = 0.0

    for ticker, row in rows.items():
        relative_selected = ticker in selected if uses_relative else True
        absolute_passed = _absolute_pass(row["절대 계산값"], bil_score) if uses_absolute else None
        if ticker in passed:
            reason = "선택 및 기준 통과" if uses_absolute else "상대 상위 자산 선택"
        elif not relative_selected:
            reason = "상대 순위 미달"
        elif uses_absolute and not absolute_passed:
            reason = "절대 기준 탈락 · 방어 전환" if switched else "절대 기준 탈락 · 현금 전환"
        else:
            reason = "미선택"
        audit.append(
            {
                **{key: value for key, value in row.items() if not key.endswith("상세")},
                "상대 선택": relative_selected,
                "절대 통과": absolute_passed,
                "선택 여부": ticker in passed,
                "초기 배분 비중": initial_amount / amount if amount > 0 else 0.0,
                "최종 목표 비중": weights.get(ticker, 0.0),
                "결과 사유": reason,
            }
        )
    cash_amount = 0.0 if switched else max(0.0, amount - sum(weights.values()))
    return weights, defense_amount + 0.0, audit + ([{
        "역할": "공격", "티커": CASH, "선택 여부": True,
        "초기 배분 비중": cash_amount / amount if amount > 0 else 0.0,
        "최종 목표 비중": cash_amount, "결과 사유": "공격자산 기준 미달",
    }] if cash_amount else [])


def _defense_allocation(
    mode: str,
    amount: float,
    rows: dict[str, dict[str, Any]],
    select_count: int,
    include_cash: bool,
    previous: set[str],
    registration_order: list[str],
    bil_score: float | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if amount <= 1e-12:
        return {}, []
    candidates = list(registration_order)
    score_map = {ticker: row["상대 계산값"] for ticker, row in rows.items()}
    if include_cash:
        score_map[CASH] = 0.0
        candidates.append(CASH)
    if mode == "전체 동일 비중":
        chosen = candidates
        passed = chosen
        uses_absolute = False
        cash_switch = False
    else:
        ranked = _ranked(score_map, previous, candidates)
        for rank, ticker in enumerate(ranked, 1):
            if ticker in rows:
                rows[ticker]["순위"] = rank
        uses_relative = mode in {"상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 현금 스위치"}
        uses_absolute = mode in {"절대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 현금 스위치"}
        chosen = ranked[:select_count] if uses_relative else ranked
        passed = []
        for ticker in chosen:
            value = 0.0 if ticker == CASH else rows[ticker]["절대 계산값"]
            if not uses_absolute or _absolute_pass(value, bil_score):
                passed.append(ticker)
        cash_switch = mode == "상대 & 절대 모멘텀 + 현금 스위치"

    weights: dict[str, float] = {}
    initial_amount = amount / len(chosen) if chosen else 0.0
    if cash_switch:
        base = initial_amount
        for ticker in chosen:
            target = ticker if ticker in passed else CASH
            weights[target] = weights.get(target, 0.0) + base
    elif passed:
        weights = {ticker: amount / len(passed) for ticker in passed}
    else:
        weights = {CASH: amount}

    audit = []
    for ticker in candidates:
        selected = ticker in passed
        row = rows.get(ticker, {"역할": "방어", "티커": CASH, "상대 계산값": 0.0, "절대 계산값": 0.0})
        audit.append(
            {
                **{key: value for key, value in row.items() if not key.endswith("상세")},
                "선택 여부": selected,
                "초기 배분 비중": initial_amount / amount if amount > 0 else 0.0,
                "최종 목표 비중": weights.get(ticker, 0.0),
                "결과 사유": "방어자산 선택" if selected else "방어자산 기준 미달",
            }
        )
    return weights, audit


def _canary_momentum_config(config: dict[str, Any], risk: dict[str, Any]) -> MomentumConfig:
    """Resolve the canary signal calculation from the attack settings."""
    if not risk.get("same_momentum", True):
        return risk["custom_momentum"]

    attack_mode = config.get("attack_mode", "")
    if risk.get("momentum_type") == "절대 모멘텀" and attack_mode in _ATTACK_ABSOLUTE_MODES:
        return config["attack_absolute"]
    if risk.get("momentum_type") == "상대 모멘텀" and attack_mode in {
        "상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치",
    }:
        return config["attack_relative"]
    # If the selected attack mode has no corresponding calculation, reuse the
    # available attack setting instead of inventing a second formula.
    return config["attack_absolute"] if attack_mode in _ATTACK_ABSOLUTE_MODES else config["attack_relative"]


def _apply_canary_overlay(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    attack_weights: dict[str, float],
    config: dict[str, Any],
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Apply the canary risk overlay to the remaining attack allocation."""
    risk = config.get("risk_avoidance") or {}
    if not risk.get("enabled"):
        return attack_weights, 0.0, {"enabled": False, "audit": pd.DataFrame()}

    tickers = list(dict.fromkeys(risk.get("tickers", [])))
    if not tickers:
        raise ValueError("위험 회피 옵션을 켠 경우 카나리아 ETF를 한 개 이상 선택해주세요.")

    momentum = _canary_momentum_config(config, risk)
    direction = risk.get("direction", "마이너스(-)")
    audit: list[dict[str, Any]] = []
    for ticker in tickers:
        if ticker not in prices.columns:
            raise ValueError(f"카나리아 {ticker}의 가격 데이터가 없습니다.")
        series = prices[ticker].dropna()
        try:
            detail = momentum_detail(series, signal_date, momentum)
        except ValueError as error:
            available = "데이터 없음"
            if not series.empty:
                available = f"{series.index.min():%Y-%m-%d} ~ {series.index.max():%Y-%m-%d}"
            raise ValueError(
                f"카나리아 {ticker}의 모멘텀 계산에 필요한 과거 데이터가 부족합니다. "
                f"필요 기준: {momentum.kind} {momentum.months if momentum.kind != '모멘텀 스코어' else momentum.score_type}, "
                f"확보 기간: {available}"
            ) from error
        value = float(detail["계산값"])
        danger = value < 0 if direction == "마이너스(-)" else value > 0
        audit.append(
            {
                "티커": ticker,
                "산출 기준": momentum.kind,
                "기준 기간": detail.get("기준 기간", "-"),
                "모멘텀 값": value,
                "위험 판정": danger,
                "데이터 유효": True,
            }
        )

    risk_count = sum(bool(row["위험 판정"]) for row in audit)
    valid_count = len(audit)
    apply_mode = risk.get("apply_mode", "중립형")
    if apply_mode == "방어형":
        conversion = 1.0 if risk_count > 0 else 0.0
    elif apply_mode == "공격형":
        conversion = 1.0 if risk_count == valid_count else 0.0
    else:
        conversion = risk_count / valid_count if valid_count else 0.0

    before_attack = sum(weight for ticker, weight in attack_weights.items() if ticker != CASH)
    moved = before_attack * conversion
    adjusted = {
        ticker: weight * (1.0 - conversion) if ticker != CASH else weight
        for ticker, weight in attack_weights.items()
    }
    summary = {
        "enabled": True,
        "audit": pd.DataFrame(audit),
        "risk_count": risk_count,
        "valid_count": valid_count,
        "apply_mode": apply_mode,
        "conversion": conversion,
        "before_attack": before_attack,
        "after_attack": sum(weight for ticker, weight in adjusted.items() if ticker != CASH),
        "moved": moved,
    }
    return adjusted, moved, summary


def _market_asof(series: pd.Series, when: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    valid = series.loc[:when].dropna()
    if valid.empty:
        raise ValueError("시장 데이터의 판단 기준일 이전 값이 없습니다.")
    return pd.Timestamp(valid.index[-1]), float(valid.iloc[-1])


def _market_signal_details(
    prices: pd.DataFrame, signal_date: pd.Timestamp, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate the public market-data signals selected in the B option."""
    risk = config.get("market_risk") or {}
    data = config.get("_market_data") or {}
    labels = list(dict.fromkeys(risk.get("signals", [])))
    if not labels:
        raise ValueError("시장 데이터 위험 회피 옵션을 켠 경우 신호를 한 개 이상 선택해주세요.")
    rows: list[dict[str, Any]] = []
    spread_settings = risk.get("spread_settings") or {}
    unemployment_months = int(risk.get("unemployment_months", 12))

    def monthly(series: pd.Series) -> pd.Series:
        clean = series.dropna().sort_index()
        periods = clean.index.to_period("M")
        result = clean.groupby(periods).last().dropna()
        result.index = result.index.to_timestamp("M")
        return result

    def unemployment_state() -> tuple[bool, float, float]:
        series = monthly(data["UNRATE"])
        _, current = _market_asof(series, signal_date)
        prior = series.loc[:signal_date].iloc[:-1]
        if len(prior) < unemployment_months:
            raise ValueError(f"실업률 계산에 필요한 {unemployment_months}개월 데이터가 부족합니다.")
        average = float(prior.iloc[-unemployment_months:].mean())
        previous = float(prior.iloc[-1])
        return current > average and current > previous, current, average

    for label in labels:
        active = False
        value: float | None = None
        basis = ""
        threshold: float | None = None
        source = ""
        if "금리차" in label:
            is_three_month = "10년-3개월" in label
            defaults = {"lag": 7, "active": 8, "threshold": -0.5} if is_three_month else {"lag": 0, "active": 3, "threshold": 0.0}
            setting_key = "10년-3개월" if is_three_month else "10년-2년"
            setting = {**defaults, **spread_settings.get(setting_key, {})}
            lag = int(setting["lag"])
            active_months = int(setting["active"])
            threshold = float(setting["threshold"])
            direct_key = "T10Y3M" if is_three_month else "T10Y2Y"
            monthly_direct_key = "T10Y3MM" if is_three_month else None
            short_key = "DGS3MO" if is_three_month else "DGS2"
            short_label = "3개월" if is_three_month else "2년"
            if direct_key in data:
                spread = data[direct_key]
                source = direct_key
            elif monthly_direct_key and monthly_direct_key in data:
                spread = data[monthly_direct_key]
                source = monthly_direct_key
            else:
                dgs10, short_rate = data["DGS10"], data[short_key]
                spread = dgs10.reindex(dgs10.index.union(short_rate.index)).ffill() - short_rate.reindex(dgs10.index.union(short_rate.index)).ffill()
                source = f"DGS10-{short_key}"
            spread_monthly = monthly(spread)
            if is_three_month:
                target = (signal_date.to_period("M") - lag).to_timestamp("M")
            else:
                target = signal_date - pd.DateOffset(months=lag)
            observed_date, value = _market_asof(spread_monthly, target)
            if is_three_month:
                # An event in month E activates from E+lag through the next
                # active_months-1 monthly decisions. At T, inspect T-lag-
                # (active_months-1) through T-lag inclusively.
                dates = [
                    (signal_date.to_period("M") - lag - offset).to_timestamp("M")
                    for offset in range(max(1, active_months))
                ]
            else:
                dates = [target - pd.DateOffset(months=offset) for offset in range(max(1, active_months))]
            observations = []
            for date in dates:
                try:
                    observed, spread_value = _market_asof(spread_monthly, date)
                except ValueError:
                    observations = []
                    break
                observations.append((observed, spread_value))
            active = len(observations) == max(1, active_months) and any(spread_value < threshold for _, spread_value in observations)
            basis = f"{lag}개월 지연 {short_label} 금리차 {value:.2f}%p < {threshold:.2f}%p · {active_months}개월 이벤트 창 · 출처 {source}"
        elif label == "S&P500 배당수익률":
            # The rebalance decision already uses the prior market session as
            # ``signal_date``.  A month-end observation is therefore available
            # to the following month's first-session rebalance without another
            # one-month subtraction.  Subtracting here would create a two-month
            # effective lag (for example, December data at a February trade).
            # Monthly observations are stamped on the calendar month-end. If
            # month-end is a weekend, ``signal_date`` is the preceding Friday;
            # compare by observation month so that value is not skipped.
            observation_month_end = signal_date.to_period("M").to_timestamp("M")
            observed_date, value = _market_asof(
                data["SP500_DIV_YIELD"], observation_month_end
            )
            threshold = float(risk.get("dividend_threshold", 1.32))
            source = str(
                data["SP500_DIV_YIELD"].attrs.get(
                    "source", "S&P500 배당수익률 데이터"
                )
            )
            active = value < threshold
            basis = (
                f"현재 {value:.2f}% < 기준 {threshold:.2f}%"
                f" · 출처 {source}"
            )
        elif label == "실업률":
            active, current, average = unemployment_state()
            observed_date, _ = _market_asof(monthly(data["UNRATE"]), signal_date)
            value = current
            basis = f"현재 {current:.2f}% · 이전 평균 {average:.2f}%"
        elif label == "실업률 & SPY (GTT-UE)":
            unemployment_active, current, average = unemployment_state()
            observed_date, _ = _market_asof(monthly(data["UNRATE"]), signal_date)
            spy = prices["SPY"].dropna()
            _, spy_current = _market_asof(spy, signal_date)
            _, spy_ma = _market_asof(spy.loc[:signal_date].rolling(210, min_periods=100).mean().dropna(), signal_date)
            active = unemployment_active and spy_current < spy_ma
            value = current
            basis = f"실업률 {current:.2f}% / 평균 {average:.2f}% · SPY {spy_current:.2f} < MA {spy_ma:.2f}"
        elif label in {"삼의 법칙 (Sahm's Rule)", "삼의 법칙과 SPY 모멘텀"}:
            series = monthly(data["UNRATE"])
            observed_date, current = _market_asof(series, signal_date)
            rolling = series.rolling(3).mean().dropna()
            current_avg = _market_asof(rolling, signal_date)[1]
            prior = rolling.loc[:signal_date].iloc[:-1]
            if len(prior) < 12:
                raise ValueError("Sahm's Rule 계산에 필요한 실업률 데이터가 부족합니다.")
            prior_min = float(prior.iloc[-12:].min())
            previous = float(series.loc[:signal_date].iloc[:-1].iloc[-1])
            sahm = current_avg >= prior_min + 0.5 and current > previous
            active = sahm
            value = current_avg
            basis = f"3개월 평균 {current_avg:.2f}% / 이전 12개월 최저 {prior_min:.2f}%"
            if label == "삼의 법칙과 SPY 모멘텀":
                spy = prices["SPY"].dropna()
                _, spy_current = _market_asof(spy, signal_date)
                _, spy_ma = _market_asof(spy.loc[:signal_date].rolling(210, min_periods=100).mean().dropna(), signal_date)
                active = sahm and spy_current < spy_ma
                basis += f" · SPY {spy_current:.2f} < MA {spy_ma:.2f}"
        rows.append({
            "시장 신호": label,
            "관찰일": observed_date,
            "원시값": value,
            "위험 판정": bool(active),
            "판정 근거": basis,
            "임계값": threshold,
            "신호 출처": source,
        })
    return rows


def _apply_market_overlay(
    prices: pd.DataFrame, signal_date: pd.Timestamp, attack_weights: dict[str, float], config: dict[str, Any]
) -> tuple[dict[str, float], float, dict[str, Any]]:
    risk = config.get("market_risk") or {}
    if not risk.get("enabled"):
        return attack_weights, 0.0, {"enabled": False, "triggered": False, "audit": pd.DataFrame()}
    audit = _market_signal_details(prices, signal_date, config)
    active = [bool(row["위험 판정"]) for row in audit]
    triggered = any(active) if risk.get("combine", "OR") == "OR" else bool(active) and all(active)
    conversion = 1.0 if triggered else 0.0
    before_attack = sum(weight for ticker, weight in attack_weights.items() if ticker != CASH)
    moved = before_attack * conversion
    adjusted = {
        ticker: weight * (1.0 - conversion) if ticker != CASH else weight
        for ticker, weight in attack_weights.items()
    }
    return adjusted, moved, {
        "enabled": True, "triggered": triggered, "combine": risk.get("combine", "OR"),
        "audit": pd.DataFrame(audit), "active_count": sum(active), "signal_count": len(active),
        "conversion": conversion, "before_attack": before_attack,
        "after_attack": sum(weight for ticker, weight in adjusted.items() if ticker != CASH), "moved": moved,
    }


def _previous_market_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    candidates = index[index < date]
    if not len(candidates):
        raise ValueError("최초 투자 전에 모멘텀을 계산할 과거 데이터가 부족합니다.")
    return pd.Timestamp(candidates[-1])


def run_backtest(prices: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run the strategy and preserve each rebalance decision snapshot."""
    market_risk = dict(config.get("market_risk") or {})
    allow_unverified = os.environ.get("DYNAMIC_ENABLE_UNVERIFIED_SIGNALS") == "1"
    market_risk["signals"] = normalize_market_signals(
        market_risk.get("signals", []), allow_unverified=allow_unverified
    )
    config = dict(config)
    config["market_risk"] = market_risk
    # Keep the unfilled frame for the common-period calculation.  If we filled
    # first, a stale last quote could make a ticker appear to have data beyond
    # its actual end date and the backtest would silently use a shorter/invalid
    # common period.
    raw_prices = prices.sort_index().astype(float)
    raw_execution = config.get("execution_prices")
    if raw_execution is None:
        raw_execution = raw_prices.copy()
    else:
        raw_execution = raw_execution.sort_index().astype(float)
    requested_start = pd.Timestamp(config["start_date"])
    requested_end = pd.Timestamp(config["end_date"])
    cash_replacement = config.get("cash_replacement_ticker")
    required_tickers = list(dict.fromkeys(
        config["attack_tickers"] + config["defense_tickers"]
        + ([cash_replacement] if cash_replacement else [])
    ))
    missing_columns = [ticker for ticker in required_tickers if ticker not in raw_prices.columns]
    if missing_columns:
        raise ValueError("가격 데이터가 없는 종목: " + ", ".join(missing_columns))
    nonempty = {ticker: raw_prices[ticker].dropna() for ticker in required_tickers}
    empty_tickers = [ticker for ticker, series in nonempty.items() if series.empty]
    if empty_tickers:
        raise ValueError("가격 데이터가 없는 종목: " + ", ".join(empty_tickers))
    valid_starts = [series.index.min() for series in nonempty.values()]
    valid_ends = [series.index.max() for series in nonempty.values()]
    common_data_start = max(valid_starts)
    # Start on the first common session on or after the requested date. A
    # holiday start is therefore executed on the next available session.
    first_common = raw_prices.index[
        (raw_prices.index >= requested_start) & (raw_prices.index >= common_data_start)
    ]
    start = pd.Timestamp(first_common[0]) if len(first_common) else max([requested_start, common_data_start])
    end = min([requested_end, *valid_ends])
    evaluation_index = raw_prices.index[(raw_prices.index >= start) & (raw_prices.index <= end)]
    if len(evaluation_index) < 2:
        raise ValueError("평가 기간의 공통 거래일이 부족합니다.")
    # Forward fill only after the common period has been established.  This is
    # for isolated missing observations; it cannot extend a ticker's usable
    # start or end date anymore.
    prices = raw_prices.ffill()
    execution_prices = raw_execution.reindex(prices.index).ffill()
    start_actual, end_actual = pd.Timestamp(evaluation_index[0]), pd.Timestamp(evaluation_index[-1])
    schedules = scheduled_trade_dates(evaluation_index, requested_start, end_actual, config["rebalance_frequency"])
    schedule_map = {actual: planned for planned, actual in schedules}
    contribution_dates = {
        actual
        for _, actual in scheduled_trade_dates(
            evaluation_index,
            requested_start,
            end_actual,
            {"매월": "월별", "매분기": "분기", "매년": "연별"}[config["contribution_frequency"]],
        )[1:]
    } if config["contribution_amount"] > 0 else set()

    tickers = list(dict.fromkeys(
        config["attack_tickers"] + config["defense_tickers"]
        + ([cash_replacement] if cash_replacement else [])
    ))
    shares = pd.Series(0.0, index=tickers)
    cash = float(config["initial_investment"])
    total_invested = cash
    fee_rate = float(config["transaction_cost_rate"]) if config["use_transaction_cost"] else 0.0
    values, invested, flows, nav_returns = [], [], [], []
    history: list[dict[str, Any]] = []
    candidate_snapshots: dict[str, pd.DataFrame] = {}
    canary_audits: dict[str, pd.DataFrame] = {}
    market_audits: dict[str, pd.DataFrame] = {}
    target_weights: dict[str, float] = {CASH: 1.0}
    previous_attack: set[str] = set()
    previous_defense: set[str] = set()
    turnover_sum = 0.0

    for date in evaluation_index:
        contribution = float(config["contribution_amount"]) if date in contribution_dates else 0.0
        if contribution:
            cash += contribution
            total_invested += contribution
        before_value = float((shares * prices.loc[date, tickers]).sum() + cash)
        if date in schedule_map:
            signal_date = _previous_market_date(prices.index, date)
            attack_rows, attack_bil = _evaluate_candidates(
                prices, config["attack_tickers"], signal_date,
                config["attack_relative"], config["attack_absolute"],
                config["attack_use_bil"], config["bil_ticker"], "공격",
            )
            weight_mode = config.get("weight_mode", "고정 범위")
            weight_summary = "고정 범위"
            if weight_mode == "마이너스 절대 모멘텀 수":
                weight_config, weight_basis = _weight_momentum_config(config, weight_mode)
                weight_values = {}
                for ticker in config["attack_tickers"]:
                    if config.get("negative_same_momentum", True):
                        field = "절대 계산값" if config.get("attack_mode") in _ATTACK_ABSOLUTE_MODES else "상대 계산값"
                        value = float(attack_rows[ticker][field])
                    else:
                        value = float(momentum_detail(prices[ticker], signal_date, weight_config)["계산값"])
                    weight_values[ticker] = value
                    attack_rows[ticker]["비중 기준값"] = value
                    attack_rows[ticker]["비중 통과"] = value >= 0.0
                negative_count = sum(value < 0.0 for value in weight_values.values())
                attack_base = 0.0 if negative_count >= int(config["negative_threshold"]) else 1.0
                weight_summary = f"{weight_basis} · 마이너스 {negative_count}개 / 기준 {int(config['negative_threshold'])}개"
            elif weight_mode == "플러스 절대 모멘텀 자산 개수 비중 한도":
                weight_config, weight_basis = _weight_momentum_config(config, weight_mode)
                weight_values = {}
                for ticker in config["attack_tickers"]:
                    if config.get("positive_same_momentum", True):
                        field = "절대 계산값" if config.get("attack_mode") in _ATTACK_ABSOLUTE_MODES else "상대 계산값"
                        value = float(attack_rows[ticker][field])
                    else:
                        value = float(momentum_detail(prices[ticker], signal_date, weight_config)["계산값"])
                    weight_values[ticker] = value
                    attack_rows[ticker]["비중 기준값"] = value
                    attack_rows[ticker]["비중 통과"] = value > 0.0
                valid_count = len(weight_values)
                positive_count = sum(value > 0.0 for value in weight_values.values())
                attack_base = positive_count / valid_count if valid_count else 0.0
                weight_summary = f"{weight_basis} · 플러스 {positive_count}/{valid_count}개"
            else:
                attack_base = 1.0 - float(config.get("fixed_defense_weight", 0.0))
            attack_weights, switched_amount, attack_audit = _attack_allocation(
                config["attack_mode"], attack_base, attack_rows,
                config["attack_select_count"], previous_attack, config["attack_tickers"], attack_bil,
            )
            pre_canary_attack_weights = attack_weights.copy()
            attack_weights, market_moved, market_summary = _apply_market_overlay(
                prices, signal_date, attack_weights, config,
            )
            if market_summary.get("triggered"):
                canary_moved = 0.0
                canary_summary = {
                    "enabled": False,
                    "skipped": True,
                    "skip_reason": "B 우선 발동으로 A 미평가",
                    "audit": pd.DataFrame(),
                }
            else:
                attack_weights, canary_moved, canary_summary = _apply_canary_overlay(
                    prices, signal_date, attack_weights, config,
                )
            for row in attack_audit:
                ticker = row.get("티커")
                before_weight = pre_canary_attack_weights.get(ticker, 0.0)
                after_weight = attack_weights.get(ticker, 0.0)
                row["카나리아 전환 전 비중"] = before_weight
                row["카나리아 전환 비중"] = max(0.0, before_weight - after_weight)
                row["최종 목표 비중"] = after_weight
                if row.get("카나리아 전환 비중", 0.0) > 0:
                    row["결과 사유"] = f"{row.get('결과 사유', '')} · 카나리아 전환"
            base_defense = 1.0 - attack_base
            fixed_and_switched = base_defense + switched_amount + market_moved + canary_moved
            defense_weights: dict[str, float] = {}
            defense_audit: list[dict[str, Any]] = []
            if fixed_and_switched > 0:
                if config["defense_enabled"]:
                    defense_rows, defense_bil = _evaluate_candidates(
                        prices, config["defense_tickers"], signal_date,
                        config["defense_relative"], config["defense_absolute"],
                        config["defense_use_bil"], config["bil_ticker"], "방어",
                    ) if config["defense_tickers"] else ({}, None)
                    defense_weights, defense_audit = _defense_allocation(
                        config["defense_mode"], fixed_and_switched, defense_rows,
                        config["defense_select_count"], config["include_cash"],
                        previous_defense, config["defense_tickers"], defense_bil,
                    )
                else:
                    defense_weights = {CASH: fixed_and_switched}
                    defense_audit = [{"역할": "방어", "티커": CASH, "선택 여부": True, "최종 목표 비중": fixed_and_switched, "결과 사유": "방어자산 기능 비활성화"}]

            new_weights: dict[str, float] = {}
            for source in (attack_weights, defense_weights):
                for ticker, weight in source.items():
                    new_weights[ticker] = new_weights.get(ticker, 0.0) + weight
            attack_cash = sum(row.get("최종 목표 비중", 0.0) for row in attack_audit if row.get("티커") == CASH)
            if attack_cash:
                new_weights[CASH] = new_weights.get(CASH, 0.0) + attack_cash
            remainder = 1.0 - sum(new_weights.values())
            if remainder > 1e-10:
                new_weights[CASH] = new_weights.get(CASH, 0.0) + remainder
            if cash_replacement:
                if cash_replacement not in prices.columns:
                    raise ValueError(f"현금 대체 자산 {cash_replacement}의 가격 데이터가 없습니다.")
                replacement_amount = new_weights.pop(CASH, 0.0)
                if replacement_amount:
                    new_weights[cash_replacement] = new_weights.get(cash_replacement, 0.0) + replacement_amount
            if not math.isclose(sum(new_weights.values()), 1.0, abs_tol=1e-8):
                raise ValueError("최종 목표 비중 합계가 100%가 아닙니다.")

            execution_row = execution_prices.loc[date, tickers]
            if execution_row.isna().any() or (execution_row <= 0).any():
                raise ValueError(f"{date.date()}의 수정 시가 데이터가 부족합니다.")
            current_values = shares * execution_row
            # Rebalance at the adjusted open.  Existing holdings must also be
            # marked at that same open before target trades are calculated.
            # Using the day's adjusted close here and then buying at the open
            # would count the rebalance day's open-to-close move twice.
            before_value = float(current_values.sum() + cash)
            current_weights = {ticker: float(value / before_value) for ticker, value in current_values.items()}
            current_weights[CASH] = cash / before_value if before_value else 0.0
            all_assets = set(current_weights) | set(new_weights)
            turnover = 0.5 * sum(abs(new_weights.get(asset, 0.0) - current_weights.get(asset, 0.0)) for asset in all_assets)
            if date != start_actual:
                turnover_sum += turnover

            gross_trade = sum(abs(before_value * new_weights.get(ticker, 0.0) - current_values.get(ticker, 0.0)) for ticker in tickers)
            fee = gross_trade * fee_rate
            investable = max(0.0, before_value - fee)
            for ticker in tickers:
                shares[ticker] = investable * new_weights.get(ticker, 0.0) / float(execution_prices.at[date, ticker])
            cash = investable * new_weights.get(CASH, 0.0)
            target_weights = new_weights
            previous_attack = {ticker for ticker in attack_weights if ticker != CASH}
            previous_defense = {ticker for ticker in defense_weights if ticker != CASH}
            after_value = float((shares * prices.loc[date, tickers]).sum() + cash)
            rows = attack_audit + defense_audit
            snapshot_key = date.strftime("%Y-%m-%d")
            candidate_snapshots[snapshot_key] = pd.DataFrame(rows)
            canary_audits[snapshot_key] = canary_summary["audit"]
            market_audits[snapshot_key] = market_summary["audit"]
            history.append(
                {
                    "예정일": schedule_map[date].date(),
                    "적용일": date.date(),
                    "판단 기준일": signal_date.date(),
                    "선택된 종목": ", ".join(f"{ticker} {weight:.1%}" for ticker, weight in new_weights.items() if weight > 0),
                    "공격자산 비중": sum(attack_weights.values()),
                    "방어자산 비중": sum(value for ticker, value in defense_weights.items() if ticker != CASH),
                    "현금 비중": new_weights.get(CASH, 0.0),
                    "포트폴리오 비중 방식": weight_mode,
                    "비중 판정": weight_summary,
                    "카나리아 위험 회피": "미사용" if not canary_summary["enabled"] else canary_summary["apply_mode"],
                    "카나리아 위험 수": canary_summary.get("risk_count", 0),
                    "카나리아 수": canary_summary.get("valid_count", 0),
                    "카나리아 전환율": canary_summary.get("conversion", 0.0),
                    "카나리아 전환 비중": canary_summary.get("moved", 0.0),
                    "카나리아 적용 전 공격자산 비중": canary_summary.get("before_attack", 0.0),
                    "카나리아 적용 후 공격자산 비중": canary_summary.get("after_attack", 0.0),
                    "시장 위험 회피": "미사용" if not market_summary["enabled"] else ("발동" if market_summary.get("triggered") else "안전"),
                    "시장 위험 신호 수": market_summary.get("active_count", 0),
                    "시장 신호 수": market_summary.get("signal_count", 0),
                    "시장 전환 비중": market_summary.get("moved", 0.0),
                    "시장 신호 결합": market_summary.get("combine", "-"),
                    "현금 대체 자산": cash_replacement or CASH,
                    "시장 위험 회피 사유": (
                        "B 우선 발동으로 A 미평가" if market_summary.get("triggered")
                        else ("B 미발동 후 A 평가" if market_summary.get("enabled") else "-")
                    ),
                    "턴오버": 0.0 if date == start_actual else turnover,
                    "거래비용": fee,
                    "수정 시가": ", ".join(f"{ticker} {float(execution_prices.at[date, ticker]):.6g}" for ticker in tickers if new_weights.get(ticker, 0.0) > 0),
                    "수정 종가": ", ".join(f"{ticker} {float(prices.at[date, ticker]):.6g}" for ticker in tickers if new_weights.get(ticker, 0.0) > 0),
                    "평가금액": after_value,
                }
            )
            before_value = after_value
        elif contribution:
            # Use new cash to move toward the current target without selling.
            for ticker, weight in target_weights.items():
                if ticker == CASH or weight <= 0:
                    continue
                amount = contribution * weight
                fee = amount * fee_rate
                shares[ticker] += max(0.0, amount - fee) / float(execution_prices.at[date, ticker])
                cash -= amount
            before_value = float((shares * prices.loc[date, tickers]).sum() + cash)

        period_return = (
            before_value / float(config["initial_investment"]) - 1.0
            if not values
            else (before_value - contribution) / values[-1] - 1.0
        )
        values.append(before_value)
        invested.append(total_invested)
        flows.append(contribution)
        nav_returns.append(period_return)

    value_series = pd.Series(values, index=evaluation_index, name="포트폴리오 평가액")
    invested_series = pd.Series(invested, index=evaluation_index, name="누적 투입금")
    flow_series = pd.Series(flows, index=evaluation_index, name="외부 현금흐름")
    nav = (1.0 + pd.Series(nav_returns, index=evaluation_index).fillna(0.0)).cumprod()
    # The first observation is the initial capital after execution at the
    # applied day's open, marked at that day's close. Do not overwrite it with
    # 1.0 or the first open-to-close return disappears from NAV/month one.
    drawdown = nav / nav.cummax() - 1.0
    years = max((evaluation_index[-1] - evaluation_index[0]).days / 365.25, 1 / 365.25)
    return {
        "values": value_series,
        "invested": invested_series,
        "flows": flow_series,
        "nav": nav.rename("포트폴리오"),
        "drawdown": drawdown.rename("포트폴리오"),
        "history": pd.DataFrame(history),
        "candidate_snapshots": candidate_snapshots,
        "canary_audits": canary_audits,
        "market_audits": market_audits,
        "final_weights": target_weights,
        "annual_turnover": turnover_sum / years,
        "common_start": start_actual,
        "common_end": end_actual,
    }


