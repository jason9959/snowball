"""Streamlit screens for the dynamic portfolio backtest."""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import os
from io import BytesIO, StringIO
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from .catalog import display_name, normalize_ticker, suggestions
from .engine import MARKET_SIGNAL_REQUIRED_BY_PRESETS, MARKET_SIGNAL_SUPPORT, MomentumConfig, normalize_market_signals, resolve_recent_period, run_backtest, validate_preset_market_signals
from .market_data import load_sp500_dividend_yield, load_treasury_spreads
from .results import download_summary_csv, render as render_results


EARLIEST_DATE = dt.date(1900, 1, 1)
BENCHMARK_LABELS = ["S&P500", "NASDAQ100", "KOSPI"]
ATTACK_MODES = [
    "상대 모멘텀", "절대 모멘텀", "절대 모멘텀 + 방어 스위치",
    "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치",
]
DEFENSE_MODES = [
    "전체 동일 비중", "상대 모멘텀", "절대 모멘텀",
    "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 현금 스위치",
]
SCORE_TYPES = [
    "1-3-6개월 단순", "3-6-12개월 단순", "1-3-6-12개월 단순",
    "1-3-6-9-12개월 단순", "1-3-6-12개월 가중",
]
CANARY_DEFAULTS = ["VWO", "BND", "EEM", "AGG", "SPY", "BIL", "TIP"]
CANARY_APPLY_MODES = ["중립형", "방어형", "공격형"]
MARKET_SIGNAL_LABELS = [
    "장단기 금리차 역전 (10년-2년)",
    "S&P500 배당수익률",
    "장단기 금리차 역전 (10년-3개월)",
    "실업률",
    "실업률 & SPY (GTT-UE)",
    "삼의 법칙 (Sahm's Rule)",
    "삼의 법칙과 SPY 모멘텀",
]
PERIOD_MODES = ["직접 날짜 입력", "최근 N년"]
RECENT_YEAR_OPTIONS = (1, 3, 5, 10)


# Preset metadata is centralized so the UI never scatters strategy values.
RESEARCH_PRESETS = {
    "1. 분산투자 방어형": {
        "attack_tickers": ["QQQ", "SOXX", "XLE", "XLK", "XLV", "EDV", "GLD", "PDBC"],
        "defense_tickers": ["BIL", "IEF", "HYG"],
        "attack_mode": "상대 모멘텀",
        "attack_select_count": 4,
        "attack_relative": {"kind": "모멘텀 스코어", "score_type": "1-3-6-12개월 단순"},
        "attack_absolute": {"kind": "모멘텀 스코어", "score_type": "1-3-6-12개월 단순"},
        "defense_enabled": True,
        "defense_mode": "상대 모멘텀",
        "defense_select_count": 1,
        "defense_relative": {"kind": "모멘텀 스코어", "score_type": "1-3-6-12개월 단순"},
        "include_cash": False,
        "weight_mode": "마이너스 절대 모멘텀 수",
        "negative_threshold": 5,
        "negative_same_momentum": True,
        "positive_same_momentum": True,
        "cash_replacement": False,
        "attack_use_bil": False,
        "defense_use_bil": False,
        "risk_enabled": True,
        "canary_tickers": ["SPY", "TIP"],
        "risk_apply_mode": "방어형",
        "risk_direction": "마이너스(-)",
        "risk_momentum_type": "절대 모멘텀",
        "risk_same_momentum": False,
        "risk_custom_momentum": {"kind": "이동평균", "months": 11},
        "market_risk_enabled": True,
        "market_risk_combine": "OR",
        "market_risk_signals": ["S&P500 배당수익률"],
        "dividend_threshold": 1.32,
    },
    "2. 집중투자 공격형": {
        "attack_tickers": ["QQQ", "SCHD", "VWO"],
        "defense_tickers": ["GLD", "HYG", "PDBC"],
        "attack_mode": "상대 모멘텀",
        "attack_select_count": 1,
        "attack_relative": {"kind": "모멘텀 스코어", "score_type": "3-6-12개월 단순"},
        "defense_enabled": True,
        "defense_mode": "상대 & 절대 모멘텀 + 현금 스위치",
        "defense_select_count": 1,
        "include_cash": False,
        "weight_mode": "고정 범위",
        "fixed_defense_weight": 0.0,
        "negative_threshold": 1,
        "negative_same_momentum": True,
        "positive_same_momentum": True,
        "defense_relative": {"kind": "이동평균", "months": 6},
        "defense_absolute": {"kind": "이동평균", "months": 6},
        "defense_same_momentum": False,
        "cash_replacement": False,
        "attack_use_bil": False,
        "defense_use_bil": False,
        "risk_enabled": True,
        "canary_tickers": ["TIP"],
        "risk_apply_mode": "방어형",
        "risk_direction": "마이너스(-)",
        "risk_momentum_type": "절대 모멘텀",
        "risk_same_momentum": False,
        "risk_custom_momentum": {"kind": "이동평균", "months": 12},
        "market_risk_enabled": True,
        "market_risk_combine": "OR",
        "market_risk_signals": ["장단기 금리차 역전 (10년-3개월)", "S&P500 배당수익률"],
        "spread_lag": 7,
        "spread_active": 8,
        "spread_threshold": -0.5,
        "dividend_threshold": 1.6,
    },
}
validate_preset_market_signals(RESEARCH_PRESETS, required_by_presets=MARKET_SIGNAL_REQUIRED_BY_PRESETS)


def _default_form() -> dict[str, Any]:
    today = dt.date.today()
    return {
        "dynamic_start": today - dt.timedelta(days=365 * 10),
        "dynamic_end": today,
        "dynamic_period_mode": "직접 날짜 입력",
        "dynamic_recent_years": 10,
        "dynamic_initial": 10_000.0,
        "dynamic_investment_type": "거치식",
        "dynamic_rebalance": "월별",
        "dynamic_benchmark": "S&P500",
        "dynamic_contribution": 1_000.0,
        "dynamic_contribution_frequency": "매월",
        "dynamic_market": "해외 ETF",
        "dynamic_attack_mode": "상대 모멘텀",
        "dynamic_attack_select_count": 1,
        "dynamic_defense_enabled": False,
        "dynamic_defense_mode": "전체 동일 비중",
        "dynamic_defense_select_count": 1,
        "dynamic_include_cash": True,
        "dynamic_cash_replacement": False,
        "dynamic_same_momentum": True,
        "dynamic_weight_mode": "고정 범위",
        "dynamic_fixed_defense": False,
        "dynamic_fixed_defense_weight": 20.0,
        "dynamic_negative_threshold": 1,
        "dynamic_negative_same_momentum": True,
        "dynamic_positive_same_momentum": True,
        "dynamic_risk_enabled": False,
        "dynamic_risk_apply_mode": "중립형",
        "dynamic_risk_direction": "마이너스(-)",
        "dynamic_risk_momentum_type": "절대 모멘텀",
        "dynamic_risk_same_momentum": True,
        "dynamic_market_risk_enabled": False,
        "dynamic_market_risk_combine": "OR",
        "dynamic_market_risk_signals": [],
        "dynamic_market_spread_lag": 0,
        "dynamic_market_spread_active": 3,
        "dynamic_market_spread_threshold": 0.0,
        "dynamic_market_spread_lag_10년-2년": 0,
        "dynamic_market_spread_active_10년-2년": 3,
        "dynamic_market_spread_threshold_10년-2년": 0.0,
        "dynamic_market_spread_lag_10년-3개월": 7,
        "dynamic_market_spread_active_10년-3개월": 8,
        "dynamic_market_spread_threshold_10년-3개월": -0.5,
        "dynamic_market_dividend_threshold": 1.32,
        "dynamic_market_unemployment_months": 12,
        "dynamic_use_transaction_cost": False,
        "dynamic_transaction_cost": 0.1,
    }


def _initialize_state() -> None:
    for key, value in _default_form().items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("dynamic_attack_tickers", ["SPY", "QQQ"])
    st.session_state.setdefault("dynamic_defense_tickers", ["IEF", "IAU"])
    st.session_state.setdefault("dynamic_canary_tickers", [])
    st.session_state.setdefault("dynamic_show_results", False)
    st.session_state.setdefault("dynamic_result", None)
    st.session_state.setdefault("dynamic_result_config", None)
    st.session_state.setdefault("dynamic_error", None)


def _reset_form() -> None:
    for key in list(st.session_state):
        if key.startswith("dynamic_") and key not in {"dynamic_show_results", "dynamic_result", "dynamic_result_config"}:
            del st.session_state[key]
    for key, value in _default_form().items():
        st.session_state[key] = value
    st.session_state["dynamic_attack_tickers"] = ["SPY", "QQQ"]
    st.session_state["dynamic_defense_tickers"] = ["IEF", "IAU"]
    st.session_state["dynamic_canary_tickers"] = []
    st.session_state["dynamic_result"] = None
    st.session_state["dynamic_result_config"] = None
    st.session_state.pop("_dynamic_market_seen", None)
    st.session_state.pop("dynamic_saved_widget_values", None)


def _apply_research_preset(name: str) -> None:
    """Atomically apply a preset without touching the period controls."""
    preset = RESEARCH_PRESETS[name]
    period_keys = {"dynamic_period_mode", "dynamic_recent_years", "dynamic_start", "dynamic_end"}
    preserved = {key: st.session_state[key] for key in period_keys if key in st.session_state}
    for key in list(st.session_state):
        if key.startswith("dynamic_") and key not in period_keys:
            st.session_state.pop(key, None)
    values = _default_form()
    # Keep the preset schema explicit: weight options are UI state, while
    # research definitions use the engine's canonical names/values.
    weight_mode = preset.get("weight_mode", "고정 범위")
    fixed_defense_weight = float(preset.get("fixed_defense_weight", 0.0))
    use_fixed_defense = bool(preset.get("use_fixed_defense", weight_mode == "고정 범위" and fixed_defense_weight > 0))
    values.update({
        "dynamic_attack_tickers": list(preset["attack_tickers"]),
        "dynamic_defense_tickers": list(preset["defense_tickers"]),
        "dynamic_canary_tickers": list(preset.get("canary_tickers", [])),
        "dynamic_attack_mode": preset["attack_mode"],
        "dynamic_attack_select_count": preset["attack_select_count"],
        "dynamic_defense_enabled": preset["defense_enabled"],
        "dynamic_defense_mode": preset["defense_mode"],
        "dynamic_defense_select_count": preset.get("defense_select_count", 1),
        "dynamic_include_cash": preset.get("include_cash", False),
        "dynamic_weight_mode": weight_mode,
        "dynamic_fixed_defense": use_fixed_defense,
        "dynamic_fixed_defense_weight": fixed_defense_weight * 100.0,
        "dynamic_negative_threshold": preset.get("negative_threshold", 1),
        "dynamic_negative_same_momentum": preset.get("negative_same_momentum", True),
        "dynamic_positive_same_momentum": preset.get("positive_same_momentum", True),
        "dynamic_same_momentum": preset.get("defense_same_momentum", preset.get("same_momentum", True)),
        "dynamic_attack_use_bil": preset.get("attack_use_bil", False),
        "dynamic_defense_use_bil": preset.get("defense_use_bil", False),
        "dynamic_cash_replacement": preset.get("cash_replacement", False),
        "dynamic_risk_enabled": preset.get("risk_enabled", False),
        "dynamic_risk_apply_mode": preset.get("risk_apply_mode", "중립형"),
        "dynamic_risk_direction": preset.get("risk_direction", "마이너스(-)"),
        "dynamic_risk_momentum_type": preset.get("risk_momentum_type", "절대 모멘텀"),
        "dynamic_risk_same_momentum": preset.get("risk_same_momentum", True),
        "dynamic_market_risk_enabled": preset.get("market_risk_enabled", False),
        "dynamic_market_risk_combine": preset.get("market_risk_combine", "OR"),
        "dynamic_market_risk_signals": list(preset.get("market_risk_signals", [])),
        "dynamic_market_spread_lag": preset.get("spread_lag", 0),
        "dynamic_market_spread_active": preset.get("spread_active", 3),
        "dynamic_market_spread_threshold": preset.get("spread_threshold", 0.0),
        "dynamic_market_spread_lag_10년-2년": preset.get("spread_lag_2y", 0),
        "dynamic_market_spread_active_10년-2년": preset.get("spread_active_2y", 3),
        "dynamic_market_spread_threshold_10년-2년": preset.get("spread_threshold_2y", 0.0),
        "dynamic_market_spread_lag_10년-3개월": preset.get("spread_lag", 7),
        "dynamic_market_spread_active_10년-3개월": preset.get("spread_active", 8),
        "dynamic_market_spread_threshold_10년-3개월": preset.get("spread_threshold", -0.5),
        "dynamic_market_dividend_threshold": preset.get("dividend_threshold", 1.32),
        "dynamic_cash_replacement": False,
        "dynamic_show_results": False,
        "dynamic_result": None,
        "dynamic_result_config": None,
    })
    momentum_specs = {
        "attack_relative": preset.get("attack_relative"),
        "attack_absolute": preset.get("attack_absolute"),
        "defense_relative": preset.get("defense_relative"),
        "defense_absolute": preset.get("defense_absolute"),
        "risk_custom": preset.get("risk_custom_momentum"),
    }
    for prefix, spec in momentum_specs.items():
        if not spec:
            continue
        values[f"dynamic_{prefix}_kind"] = spec["kind"]
        if spec["kind"] == "모멘텀 스코어":
            values[f"dynamic_{prefix}_score_type"] = spec["score_type"]
        else:
            values[f"dynamic_{prefix}_months"] = spec["months"]
    # Apply only supported market signals and reset every checkbox key first.
    for index, label in enumerate(MARKET_SIGNAL_LABELS):
        values[f"dynamic_market_signal_{index}"] = label in preset.get("market_risk_signals", []) and MARKET_SIGNAL_SUPPORT.get(label, False)
    values.update(preserved)
    st.session_state.update(values)
    st.session_state["dynamic_preset_notice"] = name


def _render_error_dialog() -> None:
    message = st.session_state.get("dynamic_error")
    if not message:
        return

    @st.dialog("요청을 완료하지 못했어요", dismissible=False)
    def dialog() -> None:
        st.error(message)
        st.caption("입력 조건이나 데이터 상태를 확인한 뒤 다시 실행해주세요.")
        if st.button("닫기", type="primary", key="close_dynamic_error", use_container_width=True):
            st.session_state["dynamic_error"] = None
            st.rerun()

    dialog()


def _add_ticker(state_key: str, ticker: str) -> None:
    selected = list(st.session_state.get(state_key, []))
    if ticker and ticker not in selected:
        st.session_state[state_key] = selected + [ticker]


def _remove_ticker(state_key: str, ticker: str) -> None:
    st.session_state[state_key] = [
        value for value in st.session_state.get(state_key, []) if value != ticker
    ]


def _asset_selector(role: str, market: str) -> None:
    state_key = f"dynamic_{role}_tickers"
    label = "공격자산" if role == "attack" else "방어자산"
    st.markdown(f"#### {label}")
    st.caption("티커 또는 ETF 이름을 입력해 후보를 추가하세요. 등록된 항목은 제거할 수 있습니다.")
    query = st.text_input(
        f"{label} 검색", key=f"dynamic_{role}_query", placeholder="예: QQQ, KODEX, ACE",
        help="검색어와 일치하는 ETF를 제안합니다. 목록에 없는 Yahoo Finance 티커도 직접 추가할 수 있습니다.",
    )
    matches = suggestions(query, market)
    selected_ticker = None
    if matches:
        labels = [f"{ticker} · {name}" for ticker, name in matches]
        selected_label = st.selectbox(f"{label} 검색 결과", labels, key=f"dynamic_{role}_suggestion")
        selected_ticker = matches[labels.index(selected_label)][0]
    elif query.strip():
        selected_ticker = normalize_ticker(query, market)

    if st.button(f"{label}에 추가", key=f"dynamic_add_{role}", use_container_width=True, disabled=not selected_ticker):
        _add_ticker(state_key, selected_ticker)

    selected = list(st.session_state[state_key])
    if not selected:
        st.info(f"등록된 {label}이 없습니다.")
        return
    for index, ticker in enumerate(selected):
        text_col, button_col = st.columns([5, 1])
        text_col.markdown(f"**{display_name(ticker, market)}**")
        button_col.button(
            "제거", key=f"dynamic_remove_{role}_{index}", use_container_width=True,
            on_click=_remove_ticker, args=(state_key, ticker),
        )


def _canary_selector(market: str) -> None:
    """Select canary ETFs used only for the risk-avoidance signal."""
    state_key = "dynamic_canary_tickers"
    st.markdown("#### 카나리아 ETF")
    st.caption("위험 신호 계산에만 사용하며 실제 투자자산으로 자동 편입하지 않습니다.")
    selected = list(dict.fromkeys(st.session_state.get(state_key, [])))
    available_defaults = [ticker for ticker in CANARY_DEFAULTS if ticker not in selected]
    if available_defaults:
        # display_name already includes the ticker when a catalog name exists.
        default_labels = [display_name(ticker, market) for ticker in available_defaults]
        default_choice = st.selectbox(
            "기본 후보 선택",
            ["선택하세요"] + default_labels,
            key="dynamic_canary_default_choice",
        )
        if default_choice != "선택하세요":
            default_ticker = available_defaults[default_labels.index(default_choice)]
            st.button(
                "카나리아에 추가", key="dynamic_add_default_canary", use_container_width=True,
                on_click=_add_ticker, args=(state_key, default_ticker),
            )
    else:
        st.caption("기본 후보를 모두 선택했습니다.")

    query = st.text_input(
        "사용자 정의 카나리아 추가", key="dynamic_canary_query",
        placeholder="예: VTI, KODEX 200",
    )
    matches = suggestions(query, market)
    if matches:
        match_labels = [f"{ticker} · {name}" for ticker, name in matches]
        selected_label = st.selectbox("검색 결과", match_labels, key="dynamic_canary_suggestion")
        custom = matches[match_labels.index(selected_label)][0]
        st.button(
            "카나리아에 추가", key="dynamic_add_search_canary", use_container_width=True,
            on_click=_add_ticker, args=(state_key, custom),
        )
    elif query.strip():
        custom = normalize_ticker(query, market)
        st.button(
            "입력한 티커 추가", key="dynamic_add_custom_canary", use_container_width=True,
            on_click=_add_ticker, args=(state_key, custom),
        )
    if selected:
        st.markdown("**선택된 카나리아**")
        pills = " ".join(
            f"<span style='display:inline-block;background:#ff4b4b;color:white;"
            f"border-radius:0.45rem;padding:0.35rem 0.65rem;margin:0.15rem 0.25rem 0.15rem 0;"
            f"font-weight:600;'>{html.escape(display_name(ticker, market))}</span>"
            for ticker in selected
        )
        st.markdown(pills, unsafe_allow_html=True)
        remove_choice = st.selectbox(
            "선택된 카나리아 제거",
            ["선택하세요"] + [display_name(ticker, market) for ticker in selected],
            key="dynamic_canary_remove_choice",
        )
        if remove_choice != "선택하세요":
            remove_ticker = selected[[display_name(ticker, market) for ticker in selected].index(remove_choice)]
            st.button(
                "선택 항목 제거", key="dynamic_remove_canary", use_container_width=True,
                on_click=_remove_ticker, args=(state_key, remove_ticker),
            )
    else:
        st.info("카나리아 ETF를 선택해주세요.")


def _momentum_input(prefix: str, title: str) -> MomentumConfig:
    st.markdown(f"##### {title}")
    kind = st.radio(
        f"{title} 산출 기준", ["수익률", "이동평균", "모멘텀 스코어"], horizontal=True,
        key=f"{prefix}_kind",
        help="수익률은 과거 가격 변화, 이동평균은 현재가와 일별 종가 평균의 비율, 모멘텀 스코어는 여러 기간 수익률의 조합입니다.",
    )
    if kind == "모멘텀 스코어":
        score_type = st.selectbox("스코어 구성", SCORE_TYPES, key=f"{prefix}_score_type")
        return MomentumConfig(kind=kind, score_type=score_type)
    months = st.selectbox("기준 기간", list(range(1, 13)), format_func=lambda value: f"{value}개월", key=f"{prefix}_months")
    return MomentumConfig(kind=kind, months=int(months))


def _uses_relative(mode: str) -> bool:
    return mode in {"상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치"}


def _uses_absolute(mode: str) -> bool:
    return mode not in {"상대 모멘텀", "전체 동일 비중"}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _download_prices(symbols: tuple[str, ...], start_iso: str, end_iso: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = yf.download(
        list(symbols), start=pd.Timestamp(start_iso), end=pd.Timestamp(end_iso) + pd.Timedelta(days=1),
        auto_adjust=False, progress=False, threads=False, group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance에서 가격 데이터를 가져오지 못했습니다.")
    if isinstance(raw.columns, pd.MultiIndex):
        required_fields = {"Adj Close", "Close", "Open"}
        if not required_fields.issubset(set(raw.columns.get_level_values(0))):
            raise ValueError(
                "Yahoo Finance 응답에 시가·종가·수정종가가 모두 필요합니다. "
                "분배금과 분할을 반영한 수정 시가를 만들 수 없어 백테스트를 중단했습니다."
            )
        prices = raw["Adj Close"].copy()
        opens = raw["Open"].copy()
        closes = raw["Close"].copy()
    else:
        if not {"Adj Close", "Close", "Open"}.issubset(raw.columns):
            raise ValueError(
                "Yahoo Finance 응답에 시가·종가·수정종가가 모두 필요합니다. "
                "분배금과 분할을 반영한 수정 시가를 만들 수 없어 백테스트를 중단했습니다."
            )
        prices = raw[["Adj Close"]].copy()
        prices.columns = [symbols[0]]
        opens = raw[["Open"]].copy()
        opens.columns = [symbols[0]]
        closes = raw[["Close"]].copy()
        closes.columns = [symbols[0]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])
    if isinstance(opens, pd.Series):
        opens = opens.to_frame(name=symbols[0])
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=symbols[0])
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    opens.index = pd.to_datetime(opens.index)
    if getattr(opens.index, "tz", None) is not None:
        opens.index = opens.index.tz_localize(None)
    closes.index = pd.to_datetime(closes.index)
    if getattr(closes.index, "tz", None) is not None:
        closes.index = closes.index.tz_localize(None)
    prices = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    opens = opens.apply(pd.to_numeric, errors="coerce").sort_index()
    closes = closes.apply(pd.to_numeric, errors="coerce").sort_index()
    # Yahoo's Adj Close is a synthetic total-return price, while Open and
    # Close are on the unadjusted price basis.  Apply the same daily factor to
    # Open so trade execution and portfolio valuation share one basis.
    adjustment_factor = prices.divide(closes.where(closes > 0))
    adjusted_opens = opens.multiply(adjustment_factor)
    return (
        prices,
        adjusted_opens,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _download_fred_series(series_id: str, start_iso: str, end_iso: str) -> pd.Series:
    """Download a public FRED series without requiring an API key."""
    # Yield-curve spreads are deliberately CSV-first.  Keep this guard so a
    # future caller cannot accidentally restore the fragile FRED runtime path.
    if series_id in {"T10Y2Y", "T10Y3M"}:
        return load_treasury_spreads({series_id}, start_iso, end_iso)[series_id]
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    # The dividend-yield series has had multiple public aliases. Try the
    # current FRED identifier first, then compatible legacy aliases.
    candidates = {
        "SP500_DIV_YIELD": ["SP500_DIV_YIELD", "SP500_DIVIDEND_YIELD", "SP500DIVYIELD"],
    }.get(series_id, [series_id])
    last_error: Exception | None = None
    response = None
    for candidate in candidates:
        for attempt in range(2):
            try:
                response = requests.get(
                    url,
                    params={"id": candidate, "cosd": start_iso, "coed": end_iso},
                    headers={"User-Agent": "stock-simulator/1.0"},
                    timeout=(5, 15),
                )
                response.raise_for_status()
                frame = pd.read_csv(StringIO(response.text), na_values=["."])
                if frame.shape[1] >= 2:
                    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
                    index = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
                    series = pd.Series(values.to_numpy(), index=index, name=series_id).dropna().sort_index()
                    if not series.empty:
                        return series
                last_error = ValueError(f"FRED {candidate} 데이터가 비어 있습니다.")
            except (requests.RequestException, ValueError, OSError) as error:
                last_error = error
                if attempt == 0:
                    time.sleep(0.8)
    raise ValueError(f"FRED {series_id} 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.") from last_error


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _download_shiller_dividend_yield(start_iso: str, end_iso: str) -> pd.Series:
    """Load Shiller's monthly S&P data and derive dividend yield.

    The source is the Yale-hosted original workbook, not a FRED alias. A
    local workbook cache or ``SHILLER_DIVIDEND_CSV`` override can be used when
    the remote workbook is temporarily unavailable.
    """
    # Yale occasionally blocks cloud-hosted egress.  These mirrors contain the
    # same Robert Shiller workbook and keep the preset usable on Streamlit Cloud.
    source_urls = (
        "https://raw.githubusercontent.com/jamesplease/stock-market-data/master/ie_data.xls",
        "https://raw.githubusercontent.com/jamesplease/stock-market-data/main/ie_data.xls",
        "https://posix4e.github.io/shiller_wrapper_data/ie_data.xls",
        "https://www.econ.yale.edu/~shiller/data/ie_data.xls",
    )
    source_url = source_urls[0]
    cache_dir = Path(".cache")
    cache_path = cache_dir / "shiller_ie_data.xls"
    csv_override = Path(os.environ["SHILLER_DIVIDEND_CSV"]) if os.environ.get("SHILLER_DIVIDEND_CSV") else None
    payload: bytes | None = None
    source = source_url
    if csv_override and csv_override.exists():
        frame = pd.read_csv(csv_override)
        source = str(csv_override)
    else:
        last_error: Exception | None = None
        for candidate_url in source_urls:
            try:
                response = requests.get(candidate_url, timeout=(8, 30), headers={"User-Agent": "stock-simulator/1.0"})
                response.raise_for_status()
                if not response.content:
                    raise ValueError("빈 응답입니다.")
                payload = response.content
                source_url = candidate_url
                source = candidate_url
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(payload)
                break
            except (requests.RequestException, OSError, ValueError) as error:
                last_error = error
        if payload is None:
            if not cache_path.exists():
                raise ValueError(
                    "Yale Shiller 배당 데이터와 공개 미러 다운로드에 실패했고 검증된 로컬 캐시도 없습니다. "
                    "SHILLER_DIVIDEND_CSV 환경변수로 Date,Price,Dividend CSV를 지정할 수 있습니다."
                ) from last_error
            payload = cache_path.read_bytes()
            source = str(cache_path)
        try:
            # The first sheet is a one-cell disclaimer.  Price and dividend
            # observations live in the explicitly named Data sheet.
            frame = pd.read_excel(BytesIO(payload), sheet_name="Data", header=None, engine="xlrd")
        except Exception as error:
            raise ValueError(
                "Yale Shiller 원본 파일을 읽지 못했습니다. CSV override(Date,Price,Dividend)를 사용해주세요."
            ) from error
    def select_columns(source_frame: pd.DataFrame) -> pd.DataFrame | None:
        """Select one date, nominal-price and nominal-dividend column.

        The Shiller workbook contains additional real-price and real-dividend
        columns whose displayed headers are also ``Price`` and ``Dividend``.
        Renaming ``P``/``D`` first therefore creates duplicate column names and
        makes ``frame["Price"]`` return a DataFrame instead of a Series.  Prefer
        the workbook's original nominal ``P`` and ``D`` fields, and only use
        ``Price``/``Dividend`` for the documented CSV override format.
        """

        labels = [str(value).strip() for value in source_frame.columns]
        folded = [label.casefold() for label in labels]

        def first_position(*candidates: str) -> int | None:
            for candidate in candidates:
                needle = candidate.casefold()
                if needle in folded:
                    return folded.index(needle)
            return None

        date_position = first_position("Date")
        price_position = first_position("P", "Price")
        dividend_position = first_position("D", "Dividend")
        positions = (date_position, price_position, dividend_position)
        if any(position is None for position in positions) or len(set(positions)) != 3:
            return None
        selected = source_frame.iloc[:, list(positions)].copy()
        selected.columns = ["Date", "Price", "Dividend"]
        return selected

    selected = select_columns(frame)
    if selected is None:
        for header in range(min(30, len(frame))):
            candidate = frame.iloc[header + 1:].copy()
            candidate.columns = [str(value).strip() for value in frame.iloc[header].tolist()]
            selected = select_columns(candidate)
            if selected is not None:
                break
    if selected is None:
        raise ValueError("Yale/CSV 배당 데이터에는 Date, Price, Dividend 열이 필요합니다.")
    frame = selected
    dates = pd.to_numeric(frame["Date"], errors="coerce")
    # Shiller dates are decimal years (e.g. 1950.01).  Parsing the numeric
    # value directly as a datetime would incorrectly interpret it as Unix
    # nanoseconds, so recognize the decimal-year range first.
    parsed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    decimal_mask = dates.between(1800, 3000, inclusive="both")
    parsed.loc[decimal_mask] = dates.loc[decimal_mask].map(
        lambda value: pd.Timestamp(year=int(value), month=max(1, min(12, int(round((value - int(value)) * 100)))), day=1)
    )
    parsed.loc[~decimal_mask] = pd.to_datetime(frame.loc[~decimal_mask, "Date"], errors="coerce")
    price = pd.to_numeric(frame["Price"], errors="coerce")
    dividend = pd.to_numeric(frame["Dividend"], errors="coerce")
    result = (dividend / price * 100).where(price > 0).dropna()
    result.index = pd.DatetimeIndex(parsed.loc[result.index]).to_period("M").to_timestamp("M")
    result = result.groupby(level=0).last().sort_index().loc[pd.Timestamp(start_iso):pd.Timestamp(end_iso)]
    result.name = "SP500_DIV_YIELD"
    result.attrs.update({"source": source, "source_url": source_url, "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(), "sha256": hashlib.sha256(payload or csv_override.read_bytes()).hexdigest()})
    if result.empty:
        raise ValueError("Yale Shiller 배당 데이터에 요청 기간의 유효한 값이 없습니다.")
    return result


def _prepare_prices(config: dict[str, Any]) -> tuple[pd.DataFrame, str, str, dict[str, pd.Series], pd.DataFrame]:
    market = config["market"]
    requested = list(dict.fromkeys(config["attack_tickers"] + config["defense_tickers"]))
    risk = config.get("risk_avoidance") or {}
    canaries = list(risk.get("tickers", [])) if risk.get("enabled") else []
    market_risk = config.get("market_risk") or {}
    if market == "해외 ETF":
        benchmark_source = {"S&P500": "SPY", "NASDAQ100": "QQQ", "KOSPI": "069500.KS"}[config["benchmark"]]
        extra = [benchmark_source, "BIL"] + (["KRW=X"] if config["benchmark"] == "KOSPI" else [])
    else:
        benchmark_source = {"S&P500": "360750.KS", "NASDAQ100": "133690.KS", "KOSPI": "069500.KS"}[config["benchmark"]]
        extra = [benchmark_source, "BIL", "KRW=X"]
    market_signals = set(market_risk.get("signals", [])) if market_risk.get("enabled") else set()
    market_symbols = ["SPY"] if any("SPY" in label for label in market_signals) else []
    symbols = tuple(dict.fromkeys(requested + extra + canaries + market_symbols))
    # Momentum needs twelve months of history.  Delayed yield-curve signals
    # can require more: for example, lag 7 + an 8-month event window needs at
    # least 15 completed months before the first decision date.
    lookback_months = 13
    spread_settings = market_risk.get("spread_settings") or {}
    for setting_key, signal_text, default_lag, default_active in (
        ("10년-2년", "10년-2년", 0, 3),
        ("10년-3개월", "10년-3개월", 7, 8),
    ):
        if any(signal_text in label for label in market_signals):
            setting = spread_settings.get(setting_key, {})
            lag = max(0, int(setting.get("lag", default_lag)))
            active = max(1, int(setting.get("active", default_active)))
            lookback_months = max(lookback_months, lag + active + 1)
    lookback = pd.Timestamp(config["start_date"]) - pd.DateOffset(months=lookback_months, days=10)
    raw, execution_prices = _download_prices(symbols, lookback.date().isoformat(), config["end_date"].isoformat())
    missing = [ticker for ticker in symbols if ticker not in raw.columns or raw[ticker].dropna().empty]
    if missing:
        missing_canaries = [ticker for ticker in missing if ticker in canaries]
        if missing_canaries:
            raise ValueError(
                "선택한 카나리아 ETF의 가격 데이터를 확보하지 못했습니다: "
                + ", ".join(missing_canaries)
                + f" (필요 기간: {lookback:%Y-%m-%d} ~ {config['end_date']:%Y-%m-%d})"
            )
        raise ValueError("가격 데이터가 없는 종목: " + ", ".join(missing))
    prices = raw.ffill()
    execution_prices = execution_prices.ffill()
    if market == "국내 ETF":
        prices["BIL_KRW"] = prices["BIL"] * prices["KRW=X"]
        execution_prices["BIL_KRW"] = execution_prices["BIL"] * execution_prices["KRW=X"]
        market_data = _load_market_data(market_signals, lookback.date().isoformat(), config["end_date"].isoformat())
        return prices, benchmark_source, "BIL_KRW", market_data, execution_prices
    if config["benchmark"] == "KOSPI":
        prices["KOSPI_USD"] = prices[benchmark_source] / prices["KRW=X"]
        execution_prices["KOSPI_USD"] = execution_prices[benchmark_source] / execution_prices["KRW=X"]
        market_data = _load_market_data(market_signals, lookback.date().isoformat(), config["end_date"].isoformat())
        return prices, "KOSPI_USD", "BIL", market_data, execution_prices
    market_data = _load_market_data(market_signals, lookback.date().isoformat(), config["end_date"].isoformat())
    return prices, benchmark_source, "BIL", market_data, execution_prices


def _load_market_data(signals: set[str], start_iso: str, end_iso: str) -> dict[str, pd.Series]:
    if not signals:
        return {}
    needed = set()
    if any("10년-2년" in label for label in signals):
        # Prefer FRED's published spread series.  Do not also request the
        # component yields: they are only a fallback for old cached datasets,
        # and an unnecessary DGS10 request can make an otherwise valid run fail.
        needed.add("T10Y2Y")
    if any("10년-3개월" in label for label in signals):
        needed.add("T10Y3M")
    if any(label in signals for label in {"실업률", "실업률 & SPY (GTT-UE)", "삼의 법칙 (Sahm's Rule)", "삼의 법칙과 SPY 모멘텀"}):
        needed.add("UNRATE")
    spread_ids = needed & {"T10Y2Y", "T10Y3M"}
    data = load_treasury_spreads(spread_ids, start_iso, end_iso)
    # UNRATE remains a separate public series.  Yield spreads never enter the
    # FRED runtime path: they are read from the bundled CSV and only missing
    # months are refreshed into the ephemeral runtime cache.
    for series_id in sorted(needed - spread_ids):
        data[series_id] = _download_fred_series(series_id, start_iso, end_iso)
    if "S&P500 배당수익률" in signals:
        data["SP500_DIV_YIELD"] = load_sp500_dividend_yield(start_iso, end_iso)
    return data


def _normalized(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = series.reindex(index).ffill().bfill()
    if aligned.dropna().empty:
        raise ValueError("비교 대상의 공통 가격 데이터가 없습니다.")
    return (aligned / aligned.iloc[0]).rename(series.name)


def _run(config: dict[str, Any]) -> dict[str, Any]:
    prices, benchmark_ticker, bil_ticker, market_data, execution_prices = _prepare_prices(config)
    config["bil_ticker"] = bil_ticker
    engine_config = dict(config)
    engine_config["_market_data"] = market_data
    engine_config["execution_prices"] = execution_prices
    result = run_backtest(prices, engine_config)
    index = result["nav"].index
    benchmark_nav = _normalized(prices[benchmark_ticker].rename("벤치마크"), index)
    attack_navs = pd.concat([_normalized(prices[ticker].rename(ticker), index) for ticker in config["attack_tickers"]], axis=1)
    equal_weight_nav = attack_navs.mean(axis=1).rename("동일 비중")
    all_assets = list(dict.fromkeys(config["attack_tickers"] + config["defense_tickers"]))
    asset_nav = pd.concat([_normalized(prices[ticker].rename(ticker), index) for ticker in all_assets], axis=1)
    # Keep the calculation snapshot useful to a human reviewer: the engine
    # stores tickers and raw scores, while the screen adds the catalog name.
    snapshots = {}
    for snapshot_key, snapshot in result["candidate_snapshots"].items():
        snapshot = snapshot.copy()
        if "티커" in snapshot.columns:
            snapshot.insert(
                2 if "역할" in snapshot.columns else 1,
                "ETF 명",
                snapshot["티커"].map(
                    lambda ticker: "현금" if ticker == "현금" else display_name(str(ticker), config["market"])
                ),
            )
        snapshots[snapshot_key] = snapshot
    canary_audits = {}
    for snapshot_key, audit in result.get("canary_audits", {}).items():
        audit = audit.copy()
        if "티커" in audit.columns:
            audit.insert(
                1,
                "ETF 명",
                audit["티커"].map(
                    lambda ticker: "현금" if ticker == "현금" else display_name(str(ticker), config["market"])
                ),
            )
        canary_audits[snapshot_key] = audit
    market_audits = {
        snapshot_key: audit.copy()
        for snapshot_key, audit in result.get("market_audits", {}).items()
    }
    result.update({
        "benchmark_nav": benchmark_nav,
        "equal_weight_nav": equal_weight_nav,
        "asset_nav": asset_nav,
        "candidate_snapshots": snapshots,
        "canary_audits": canary_audits,
        "market_audits": market_audits,
    })
    return result


def _render_conditions() -> None:
    st.title("🔄 동적 포트폴리오 백테스트")
    st.markdown("<p class='step-caption'>시장 조건에 따라 투자자산과 방어자산의 비중을 바꾸는 전략을 검증합니다.</p>", unsafe_allow_html=True)
    if st.button("↺ 조건 초기화", key="reset_dynamic_conditions"):
        _reset_form()
        st.rerun()
    st.caption("연구 포트폴리오 프리셋")
    preset_cols = st.columns(2)
    for column, name in zip(preset_cols, RESEARCH_PRESETS):
        with column:
            if st.button(name, key=f"apply_{name}", use_container_width=True):
                _apply_research_preset(name)
                st.rerun()
    if st.session_state.pop("dynamic_preset_notice", None):
        st.toast("프리셋을 적용했습니다. 조건을 확인하고 필요하면 수정하세요.")

    tabs = st.tabs([
        "기본설정", "종목선택", "투자자산 선택", "방어자산 선택",
        "포트폴리오 비중", "위험 회피 옵션",
    ])
    with tabs[0]:
        period_mode = st.radio(
            "백테스트 기간 방식", PERIOD_MODES, horizontal=True,
            key="dynamic_period_mode",
            help="두 방식의 날짜 값은 서로 동기화하지 않고 독립적으로 보존합니다.",
        )
        date_col, end_col, initial_col = st.columns(3)
        start_date = date_col.date_input(
            "시작 날짜", min_value=EARLIEST_DATE, max_value=dt.date.today(),
            key="dynamic_start", disabled=period_mode != "직접 날짜 입력",
        )
        end_date = end_col.date_input(
            "종료 날짜", min_value=EARLIEST_DATE, max_value=dt.date.today(),
            key="dynamic_end", disabled=period_mode != "직접 날짜 입력",
        )
        recent_years = st.selectbox(
            "최근 기간", RECENT_YEAR_OPTIONS, format_func=lambda value: f"최근 {value}년",
            key="dynamic_recent_years", disabled=period_mode != "최근 N년",
        )
        initial = initial_col.number_input("초기 투자금", min_value=1.0, step=1_000.0, key="dynamic_initial")
        type_col, rebalance_col, benchmark_col = st.columns(3)
        investment_type = type_col.selectbox("투자 방식", ["거치식", "적립식"], key="dynamic_investment_type")
        rebalance = rebalance_col.selectbox("리밸런싱 주기", ["월별", "분기", "반기", "연별"], key="dynamic_rebalance")
        benchmark = benchmark_col.selectbox("벤치마크", BENCHMARK_LABELS, key="dynamic_benchmark")
        contribution, contribution_frequency = 0.0, "매월"
        if investment_type == "적립식":
            amount_col, frequency_col = st.columns(2)
            contribution = amount_col.number_input("정기 적립금", min_value=1.0, step=100.0, key="dynamic_contribution")
            contribution_frequency = frequency_col.selectbox("적립 주기", ["매월", "매분기", "매년"], key="dynamic_contribution_frequency")
        use_cost = st.checkbox("거래비용 반영", key="dynamic_use_transaction_cost")
        cost_rate = st.number_input(
            "편도 거래비용률 (%)", min_value=0.0, max_value=10.0, step=0.01,
            key="dynamic_transaction_cost", disabled=not use_cost,
            help="매수와 매도 각각의 실제 거래금액에 적용합니다.",
        )

    with tabs[1]:
        market = st.selectbox("ETF 종류", ["해외 ETF", "국내 ETF"], key="dynamic_market")
        previous_market = st.session_state.get("_dynamic_market_seen")
        if previous_market is None:
            st.session_state["_dynamic_market_seen"] = market
        elif previous_market != market:
            st.session_state["dynamic_attack_tickers"] = []
            st.session_state["dynamic_defense_tickers"] = []
            st.session_state["_dynamic_market_seen"] = market
            st.info("ETF 종류가 변경되어 기존 후보 목록을 초기화했습니다.")
        attack_col, defense_col = st.columns(2)
        with attack_col:
            _asset_selector("attack", market)
        with defense_col:
            _asset_selector("defense", market)

    attack_tickers = list(st.session_state["dynamic_attack_tickers"])
    defense_tickers = list(st.session_state["dynamic_defense_tickers"])
    with tabs[2]:
        attack_mode = st.radio("투자자산 선택 방식", ATTACK_MODES, key="dynamic_attack_mode", help="후보군에서 투자자산을 고르고 공격자산 몫을 배분하는 규칙입니다.")
        attack_select_count = 1
        if _uses_relative(attack_mode):
            attack_count_options = list(range(1, max(1, len(attack_tickers)) + 1))
            st.session_state["dynamic_attack_select_count"] = min(
                int(st.session_state.get("dynamic_attack_select_count", 1)),
                len(attack_count_options),
            )
            attack_select_count = st.selectbox(
                "선택자산 수", attack_count_options,
                key="dynamic_attack_select_count", disabled=not attack_tickers,
            )
        attack_relative = _momentum_input("dynamic_attack_relative", "상대모멘텀") if _uses_relative(attack_mode) else MomentumConfig()
        attack_absolute = _momentum_input("dynamic_attack_absolute", "절대모멘텀") if _uses_absolute(attack_mode) else MomentumConfig()
        attack_use_bil = st.checkbox("절대모멘텀 기준을 BIL로 변경", key="dynamic_attack_use_bil", disabled=not _uses_absolute(attack_mode))

    with tabs[3]:
        defense_enabled = st.checkbox("방어자산 선택 사용", key="dynamic_defense_enabled")
        defense_mode = st.radio("방어자산 선택 방식", DEFENSE_MODES, key="dynamic_defense_mode", disabled=not defense_enabled)
        include_cash = st.checkbox("현금을 방어 후보에 포함", key="dynamic_include_cash", disabled=not defense_enabled)
        cash_replacement = st.checkbox(
            "현금 대신 BIL로 보유",
            key="dynamic_cash_replacement",
            help="절대모멘텀 탈락분·방어 전환분을 현금 잔고가 아닌 BIL 가격 흐름으로 반영합니다. 절대모멘텀 비교 기준(BIL)과는 별개의 기능입니다.",
        )
        defense_select_count = 1
        if defense_mode in {"상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 현금 스위치"}:
            count = len(defense_tickers) + (1 if include_cash else 0)
            defense_count_options = list(range(1, max(1, count) + 1))
            st.session_state["dynamic_defense_select_count"] = min(
                int(st.session_state.get("dynamic_defense_select_count", 1)),
                len(defense_count_options),
            )
            defense_select_count = st.selectbox(
                "선택자산 수", defense_count_options,
                key="dynamic_defense_select_count", disabled=not defense_enabled,
            )
        same_momentum = st.checkbox("투자자산 선택기준과 동일 모멘텀 적용", key="dynamic_same_momentum", disabled=not defense_enabled or defense_mode == "전체 동일 비중")
        if same_momentum:
            defense_relative, defense_absolute, defense_use_bil = attack_relative, attack_absolute, attack_use_bil
            st.caption("투자자산 탭의 산출 기준·기간·BIL 설정을 사용합니다.")
        else:
            defense_relative = _momentum_input("dynamic_defense_relative", "방어 상대모멘텀") if _uses_relative(defense_mode) else MomentumConfig()
            defense_absolute = _momentum_input("dynamic_defense_absolute", "방어 절대모멘텀") if _uses_absolute(defense_mode) else MomentumConfig()
            defense_use_bil = st.checkbox("방어 절대모멘텀 기준을 BIL로 변경", key="dynamic_defense_use_bil", disabled=not defense_enabled or not _uses_absolute(defense_mode))

    with tabs[4]:
        st.caption("공격자산과 방어자산의 기본 비중을 결정하는 방식입니다. 한 번에 하나만 적용됩니다.")
        weight_mode = st.radio(
            "포트폴리오 비중 결정 방식",
            ["고정 범위", "마이너스 절대 모멘텀 수", "플러스 절대 모멘텀 자산 개수 비중 한도"],
            key="dynamic_weight_mode",
            help="고정 범위는 입력한 방어 비중을 유지합니다. 나머지 두 방식은 전체 공격자산 후보의 절대 모멘텀 결과로 기본 비중을 계산합니다.",
        )
        use_fixed_defense = False
        fixed_defense = 0.0
        negative_threshold = 1
        negative_same_momentum = True
        positive_same_momentum = True
        negative_momentum = MomentumConfig()
        positive_momentum = MomentumConfig()
        if weight_mode == "고정 범위":
            use_fixed_defense = st.checkbox("방어자산 고정비중 사용", key="dynamic_fixed_defense")
            fixed_defense = st.number_input(
                "방어자산 고정비중 (%)", min_value=0.0, max_value=100.0, step=1.0,
                key="dynamic_fixed_defense_weight", disabled=not use_fixed_defense,
            ) if use_fixed_defense else 0.0
            effective_defense = fixed_defense if use_fixed_defense else 0.0
            split = pd.DataFrame({"구분": ["공격자산 기본 몫", "방어자산 고정 몫"], "비중": [100.0 - effective_defense, effective_defense]})
            st.dataframe(split.style.format({"비중": "{:.1f}%"}), hide_index=True, use_container_width=True)
            if use_fixed_defense and not defense_enabled:
                st.info("방어자산 선택을 사용하지 않아 방어자산 고정 몫은 현금으로 보유합니다.")
        elif weight_mode == "마이너스 절대 모멘텀 수":
            threshold_options = list(range(1, max(1, len(attack_tickers)) + 1))
            st.session_state["dynamic_negative_threshold"] = min(
                int(st.session_state.get("dynamic_negative_threshold", 1)), len(threshold_options)
            )
            negative_threshold = st.selectbox(
                "기준 자산 수 N", threshold_options,
                format_func=lambda value: f"{value}개 이상이면 전체 방어자산 전환",
                key="dynamic_negative_threshold", disabled=not attack_tickers,
            )
            negative_same_momentum = st.checkbox(
                "투자자산 선택기준과 동일 모멘텀 적용",
                key="dynamic_negative_same_momentum",
                help="체크하면 투자자산 선택 탭의 절대 모멘텀 기준을 재사용합니다. 절대 기준이 없는 방식은 상대 기준을 사용합니다.",
            )
            if negative_same_momentum:
                st.caption("투자자산 탭의 모멘텀 기준을 사용해 전체 공격자산 후보의 0% 미만 개수를 계산합니다.")
            else:
                negative_momentum = _momentum_input("dynamic_weight_negative", "비중 산출용 절대모멘텀")
        else:
            positive_same_momentum = st.checkbox(
                "투자자산 선택기준과 동일 모멘텀 적용",
                key="dynamic_positive_same_momentum",
                help="체크하면 투자자산 선택 탭의 절대 모멘텀 기준을 재사용합니다. 절대 기준이 없는 방식은 상대 기준을 사용합니다.",
            )
            if positive_same_momentum:
                st.caption("투자자산 탭의 모멘텀 기준을 사용해 전체 공격자산 후보 중 0% 이상 자산의 비중을 계산합니다.")
            else:
                positive_momentum = _momentum_input("dynamic_weight_positive", "비중 산출용 절대모멘텀")

    with tabs[5]:
        st.caption("공격자산에 남은 비중만 카나리아 신호에 따라 방어자산 또는 현금으로 전환합니다.")
        risk_enabled = st.checkbox(
            "위험 회피 옵션 A — 카나리아 ETF 사용",
            key="dynamic_risk_enabled",
            help="카나리아 ETF는 위험 신호 계산에만 사용되며 실제 투자자산으로 편입되지 않습니다.",
        )
        risk_tickers = list(st.session_state.get("dynamic_canary_tickers", []))
        if risk_enabled:
            _canary_selector(market)
            risk_tickers = list(st.session_state.get("dynamic_canary_tickers", []))
            risk_apply_mode = st.radio(
                "위험 신호 적용 방식", CANARY_APPLY_MODES,
                horizontal=True, key="dynamic_risk_apply_mode",
                help="중립형은 위험 카나리아 비율만큼, 방어형은 하나라도 위험하면, 공격형은 모두 위험할 때 전환합니다.",
            )
            risk_direction = st.radio(
                "위험 신호 방향", ["마이너스(-)", "플러스(+)"] ,
                horizontal=True, key="dynamic_risk_direction",
                help="선택한 모멘텀 값의 부호가 방향과 일치하면 위험 신호입니다. 정확히 0은 안전으로 처리합니다.",
            )
            risk_momentum_type = st.radio(
                "카나리아 모멘텀 기준", ["절대 모멘텀", "상대 모멘텀"],
                horizontal=True, key="dynamic_risk_momentum_type",
            )
            risk_same_momentum = st.checkbox(
                "투자자산 선택기준과 동일 모멘텀 적용",
                key="dynamic_risk_same_momentum",
                help="체크하면 투자자산 선택 탭의 산출 기준·기간·계산식을 사용합니다.",
            )
            risk_custom_momentum = (
                _momentum_input("dynamic_risk_custom", "카나리아 모멘텀")
                if not risk_same_momentum else MomentumConfig()
            )
        else:
            risk_apply_mode = st.session_state.get("dynamic_risk_apply_mode", "중립형")
            risk_direction = st.session_state.get("dynamic_risk_direction", "마이너스(-)")
            risk_momentum_type = st.session_state.get("dynamic_risk_momentum_type", "절대 모멘텀")
            risk_same_momentum = True
            risk_custom_momentum = MomentumConfig()

        st.divider()
        market_risk_enabled = st.checkbox(
            "위험 회피 옵션 B — 시장 데이터 사용",
            key="dynamic_market_risk_enabled",
            help="금리차·실업률 등 공개 시장 지표를 위험 신호로 사용합니다.",
        )
        market_risk_signals = list(st.session_state.get("dynamic_market_risk_signals", []))
        if market_risk_enabled:
            market_risk_combine = st.radio(
                "시장 위험 신호 적용 방식", ["OR", "AND"], horizontal=True,
                key="dynamic_market_risk_combine",
                help="OR은 하나라도 활성화되면, AND는 선택한 조건이 모두 활성화될 때 발동합니다.",
            )
            selected_signals = []
            signal_cols = st.columns(2)
            for index, label in enumerate(MARKET_SIGNAL_LABELS):
                with signal_cols[index % 2]:
                    supported = MARKET_SIGNAL_SUPPORT.get(label, False)
                    checked = st.checkbox(
                        label,
                        key=f"dynamic_market_signal_{index}",
                        disabled=not supported,
                        help=("검증 후 제공" if not supported else "Multpl 월별 데이터(추정치 포함)를 한 달 지연해 사용합니다."),
                    )
                    if not supported:
                        st.caption("검증 후 제공")
                    elif checked:
                        selected_signals.append(label)
            market_risk_signals = selected_signals
            spread_settings = {}
            for setting_key, setting_label, default_lag, default_active, default_threshold in (
                ("10년-2년", "10년-2년", 0, 3, 0.0),
                ("10년-3개월", "10년-3개월", 7, 8, -0.5),
            ):
                enabled = any(setting_label in label for label in market_risk_signals)
                st.markdown(f"**{setting_label} 금리차 설정**")
                lag_value = st.number_input(
                    f"{setting_label} 반영 지연 (개월)", min_value=0, max_value=24, step=1,
                    key=f"dynamic_market_spread_lag_{setting_key}", disabled=not enabled,
                )
                active_value = st.number_input(
                    f"{setting_label} 신호 유지 기간 (개월)", min_value=1, max_value=24, step=1,
                    key=f"dynamic_market_spread_active_{setting_key}", disabled=not enabled,
                )
                threshold_value = st.number_input(
                    f"{setting_label} 역전 임계값 (%p)", min_value=-10.0, max_value=10.0, step=0.1,
                    key=f"dynamic_market_spread_threshold_{setting_key}", disabled=not enabled,
                    help="값이 임계값보다 작을 때만 위험 신호입니다. 정확히 같은 값은 안전으로 처리합니다.",
                )
                spread_settings[setting_key] = {
                    "lag": int(lag_value), "active": int(active_value), "threshold": float(threshold_value),
                }
            dividend_threshold = st.number_input(
                "S&P500 배당수익률 기준 (%)", min_value=0.0, max_value=20.0, step=0.01,
                key="dynamic_market_dividend_threshold", disabled="S&P500 배당수익률" not in market_risk_signals,
            )
            unemployment_months = st.number_input(
                "실업률 비교 평균 기간 (개월)", min_value=3, max_value=36, step=1,
                key="dynamic_market_unemployment_months",
                disabled=not any("실업률" in label or "삼의 법칙" in label for label in market_risk_signals),
            )
            if not market_risk_signals:
                st.info("시장 위험 신호를 하나 이상 선택해주세요.")
        else:
            market_risk_combine = st.session_state.get("dynamic_market_risk_combine", "OR")
            spread_settings = {
                "10년-2년": {
                    "lag": st.session_state.get("dynamic_market_spread_lag_10년-2년", 0),
                    "active": st.session_state.get("dynamic_market_spread_active_10년-2년", 3),
                    "threshold": st.session_state.get("dynamic_market_spread_threshold_10년-2년", 0.0),
                },
                "10년-3개월": {
                    "lag": st.session_state.get("dynamic_market_spread_lag_10년-3개월", 7),
                    "active": st.session_state.get("dynamic_market_spread_active_10년-3개월", 8),
                    "threshold": st.session_state.get("dynamic_market_spread_threshold_10년-3개월", -0.5),
                },
            }
            dividend_threshold = st.session_state.get("dynamic_market_dividend_threshold", 1.32)
            unemployment_months = st.session_state.get("dynamic_market_unemployment_months", 12)

    back_col, result_col = st.columns(2)
    with back_col:
        go_back = st.button("뒤로", key="back_dynamic_portfolio", use_container_width=True)
    with result_col:
        run = st.button("결과 보기", type="primary", key="run_dynamic_portfolio", use_container_width=True)
    if go_back:
        st.session_state["current_page"] = "feature"
        st.rerun()
    if not run:
        return
    if period_mode == "최근 N년":
        start_date, end_date = resolve_recent_period(dt.date.today(), int(recent_years))
    if start_date >= end_date:
        st.session_state["dynamic_error"] = "시작 날짜는 종료 날짜보다 앞서야 합니다."
        st.rerun()
    if not attack_tickers:
        st.session_state["dynamic_error"] = "공격자산을 한 개 이상 등록해주세요."
        st.rerun()
    if defense_enabled and not defense_tickers and not include_cash:
        st.session_state["dynamic_error"] = "방어자산을 등록하거나 현금을 방어 후보에 포함해주세요."
        st.rerun()

    config = {
        "start_date": start_date, "end_date": end_date, "initial_investment": float(initial),
        "investment_type": investment_type, "rebalance_frequency": rebalance, "benchmark": benchmark,
        "contribution_amount": float(contribution), "contribution_frequency": contribution_frequency,
        "market": market, "currency": "KRW" if market == "국내 ETF" else "USD",
        "attack_tickers": attack_tickers, "defense_tickers": defense_tickers,
        "attack_mode": attack_mode, "attack_select_count": int(attack_select_count),
        "attack_relative": attack_relative, "attack_absolute": attack_absolute, "attack_use_bil": bool(attack_use_bil),
        "defense_enabled": bool(defense_enabled), "defense_mode": defense_mode,
        "defense_select_count": int(defense_select_count), "defense_relative": defense_relative,
        "defense_absolute": defense_absolute, "defense_use_bil": bool(defense_use_bil),
        "include_cash": bool(include_cash), "weight_mode": weight_mode,
        "cash_replacement_ticker": (
            ("BIL_KRW" if market == "국내 ETF" else "BIL") if cash_replacement else None
        ),
        "fixed_defense_weight": float((fixed_defense if use_fixed_defense else 0.0) / 100.0),
        "negative_threshold": int(negative_threshold),
        "negative_same_momentum": bool(negative_same_momentum),
        "negative_momentum": negative_momentum,
        "positive_same_momentum": bool(positive_same_momentum),
        "positive_momentum": positive_momentum,
        "risk_avoidance": {
            "enabled": bool(risk_enabled),
            "tickers": risk_tickers,
            "apply_mode": risk_apply_mode,
            "direction": risk_direction,
            "momentum_type": risk_momentum_type,
            "same_momentum": bool(risk_same_momentum),
            "custom_momentum": risk_custom_momentum,
        },
        "market_risk": {
            "enabled": bool(market_risk_enabled),
            "combine": market_risk_combine,
            "signals": market_risk_signals,
            "spread_settings": spread_settings,
            "dividend_threshold": float(dividend_threshold),
            "unemployment_months": int(unemployment_months),
        },
        "use_transaction_cost": bool(use_cost), "transaction_cost_rate": float(cost_rate / 100.0),
    }
    try:
        st.session_state["dynamic_saved_widget_values"] = {
            key: value
            for key, value in st.session_state.items()
            if key.startswith("dynamic_")
            and key not in {"dynamic_result", "dynamic_result_config", "dynamic_show_results", "dynamic_error"}
        }
        with st.spinner("가격 데이터와 리밸런싱 경로를 계산하고 있어요..."):
            result = _run(config)
        st.session_state["dynamic_result"] = result
        st.session_state["dynamic_result_config"] = config
        st.session_state["dynamic_show_results"] = True
        st.rerun()
    except Exception as error:
        st.session_state["dynamic_error"] = f"동적 백테스트를 계산하지 못했습니다: {error}"
        st.rerun()


def _render_result_page() -> None:
    result, config = st.session_state.get("dynamic_result"), st.session_state.get("dynamic_result_config")
    if not result or not config:
        st.session_state["dynamic_show_results"] = False
        st.rerun()
    render_results(result, config)
    back_col, save_col = st.columns(2)
    with back_col:
        if st.button("뒤로", key="back_dynamic_conditions", use_container_width=True):
            st.session_state["dynamic_show_results"] = False
            st.session_state["dynamic_result"] = None
            st.session_state["dynamic_result_config"] = None
            st.session_state["current_page"] = "feature"
            st.rerun()
    with save_col:
        st.download_button(
            "결과 저장", data=download_summary_csv(result),
            file_name="dynamic-portfolio-rebalance-history.csv", mime="text/csv",
            key="download_dynamic_result", use_container_width=True,
        )


def render(page: str) -> None:
    _initialize_state()
    if not st.session_state.get("dynamic_show_results"):
        for key, value in st.session_state.get("dynamic_saved_widget_values", {}).items():
            st.session_state.setdefault(key, value)
    _render_error_dialog()
    if page != "dynamic_portfolio_conditions":
        st.session_state["current_page"] = "dynamic_portfolio_conditions"
        st.rerun()
        return
    # Keep the condition form mounted and place results below it. This avoids
    # stale widgets from a separate result route and preserves all inputs.
    _render_conditions()
    if st.session_state.get("dynamic_show_results"):
        st.divider()
        _render_result_page()
