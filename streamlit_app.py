"""Streamlit entry point for Snowball Portfolio Lab."""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from snowball.core.catalog import US_ETFS, display_name
from snowball.core.engine import resolve_recent_period
from snowball.app_state import SavedRun
from snowball.data.market_data import (
    compare_dividend_snapshots,
    download_latest_sp500_dividend_snapshot,
    load_sp500_dividend_yield,
    load_treasury_spreads,
)
from snowball.data.prices import PriceBundle, download_yahoo_prices
from snowball.domestic import domestic_label, domestic_symbols
from snowball.exports import diagnostic_frames, diagnostic_zip
from snowball.presets import MomentumSpec, PRESETS, PortfolioPreset, SpreadRule
from snowball.service import (
    run_blended_presets,
    run_domestic_blended_presets,
    run_domestic_preset,
    run_preset,
)


st.set_page_config(page_title="Snowball Portfolio Lab", page_icon="❄️", layout="wide")

PRESET_NAMES = tuple(PRESETS)
PRESET_7_NAME = "7. 집중투자 공격형 1-3-6-9-12"
COMBINED_PRESET_NAME = "3. 프리셋 결합"
DOMESTIC_PRESET_1 = "4. 국내 분산투자 방어형"
DOMESTIC_PRESET_2 = "5. 국내 집중투자 공격형"
DOMESTIC_COMBINED_PRESET_NAME = "6. 국내 프리셋 결합"
ALTERNATIVE_COMBINED_PRESET_NAME = "8. 프리셋 1 + 7 결합"
COMBINED_PRESET_NAMES = {
    COMBINED_PRESET_NAME, DOMESTIC_COMBINED_PRESET_NAME,
    ALTERNATIVE_COMBINED_PRESET_NAME,
}
DOMESTIC_PRESET_NAMES = {
    DOMESTIC_PRESET_1, DOMESTIC_PRESET_2, DOMESTIC_COMBINED_PRESET_NAME,
}
MOMENTUM_KINDS = ("수익률", "이동평균", "모멘텀 스코어")
SCORE_TYPES = (
    "1-3-6개월 단순", "3-6-12개월 단순", "1-3-6-12개월 단순",
    "1-3-6-9-12개월 단순", "1-3-6-12개월 가중",
)
ATTACK_MODES = (
    "상대 모멘텀", "절대 모멘텀", "절대 모멘텀 + 방어 스위치",
    "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치",
)
DEFENSE_MODES = (
    "전체 동일 비중", "상대 모멘텀", "절대 모멘텀",
    "상대 & 절대 모멘텀 + 현금 스위치",
)
TESTED_MARKET_SIGNALS = (
    "장단기 금리차 역전 (10년-2년)",
    "장단기 금리차 역전 (10년-3개월)",
    "S&P500 배당수익률",
)
MARKET_DATA_MODES = {
    "고정 스냅샷": "fixed",
    "최신 데이터": "latest",
}


def _css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.25rem; padding-bottom: 4rem; max-width: 1180px;}
        div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
        .preset-note {color:#5f6677; font-size:.92rem; margin-top:-.4rem;}
        .metric-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; margin:.6rem 0 1rem;}
        .metric-card {background:#f7f8fc; border:1px solid #e8eaf2; padding:.8rem .9rem; border-radius:12px; min-width:0;}
        .metric-label {color:#555d6e; font-size:.82rem; line-height:1.25; margin-bottom:.32rem;}
        .metric-value {color:#222735; font-size:1.35rem; line-height:1.2; font-weight:650; white-space:nowrap;}
        @media (max-width: 640px) {
          .block-container {padding: .75rem .75rem 4.5rem;}
          h1 {font-size:1.75rem !important; line-height:1.2 !important;}
          h2 {font-size:1.35rem !important;}
          h3 {font-size:1.12rem !important;}
          p, label, .stMarkdown {font-size:.92rem;}
          .metric-grid {grid-template-columns:repeat(2,minmax(0,1fr)); gap:.5rem;}
          .metric-card {padding:.65rem .7rem; border-radius:10px;}
          .metric-label {font-size:.74rem;}
          .metric-value {font-size:1.08rem;}
          div[data-testid="stDataFrame"] {font-size:.78rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _momentum_editor(label: str, spec: MomentumSpec, key: str) -> MomentumSpec:
    st.markdown(f"**{label}**")
    left, right = st.columns(2)
    kind = left.selectbox("산출 기준", MOMENTUM_KINDS, index=MOMENTUM_KINDS.index(spec.kind), key=f"{key}_kind")
    if kind == "모멘텀 스코어":
        score = right.selectbox(
            "스코어 구성", SCORE_TYPES,
            index=SCORE_TYPES.index(spec.score_type) if spec.score_type in SCORE_TYPES else 2,
            key=f"{key}_score",
        )
        months = spec.months
    else:
        months = int(right.selectbox("기준 기간", list(range(1, 25)), index=max(0, min(23, spec.months - 1)), key=f"{key}_months"))
        score = spec.score_type
    return MomentumSpec(kind=kind, months=months, score_type=score)


def _asset_options(extra: tuple[str, ...] = ()) -> list[str]:
    return sorted(set(US_ETFS) | set(extra))


def _preset_buttons() -> str:
    st.subheader("연구 포트폴리오 불러오기")
    left, center, right = st.columns(3)
    if left.button("1. 분산투자 방어형", use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == PRESET_NAMES[0] else "secondary"):
        st.session_state.preset_name = PRESET_NAMES[0]
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    if center.button("2. 집중투자 공격형", use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == PRESET_NAMES[1] else "secondary"):
        st.session_state.preset_name = PRESET_NAMES[1]
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    if right.button("3. 프리셋 결합", use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == COMBINED_PRESET_NAME else "secondary"):
        st.session_state.preset_name = COMBINED_PRESET_NAME
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    left, center, right = st.columns(3)
    if left.button(DOMESTIC_PRESET_1, use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == DOMESTIC_PRESET_1 else "secondary"):
        st.session_state.preset_name = DOMESTIC_PRESET_1
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    if center.button(DOMESTIC_PRESET_2, use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == DOMESTIC_PRESET_2 else "secondary"):
        st.session_state.preset_name = DOMESTIC_PRESET_2
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    if right.button(DOMESTIC_COMBINED_PRESET_NAME, use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == DOMESTIC_COMBINED_PRESET_NAME else "secondary"):
        st.session_state.preset_name = DOMESTIC_COMBINED_PRESET_NAME
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    left, right = st.columns(2)
    if left.button(PRESET_7_NAME, use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == PRESET_7_NAME else "secondary"):
        st.session_state.preset_name = PRESET_7_NAME
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    if right.button(ALTERNATIVE_COMBINED_PRESET_NAME, use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == ALTERNATIVE_COMBINED_PRESET_NAME else "secondary"):
        st.session_state.preset_name = ALTERNATIVE_COMBINED_PRESET_NAME
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    selected = st.session_state.get("preset_name", PRESET_NAMES[0])
    st.caption("버튼을 누르면 아래 모든 전략 조건이 해당 연구 프리셋으로 초기화됩니다. 이후 필요한 항목만 바꿀 수 있습니다.")
    return selected


def _period_controls() -> tuple[dt.date, dt.date]:
    st.subheader("백테스트 기간")
    mode = st.radio("기간 입력 방식", ("시작일·종료일 직접 입력", "오늘부터 최근 N년"), horizontal=True, key="period_mode")
    if mode == "오늘부터 최근 N년":
        years = st.selectbox("최근 기간", (1, 3, 5, 10, 15, 20, 25), index=2, format_func=lambda x: f"{x}년", key="recent_years")
        return resolve_recent_period(dt.date.today(), int(years))
    left, right = st.columns(2)
    start = left.date_input("시작일", value=dt.date(2010, 1, 1), min_value=dt.date(1982, 1, 1), key="direct_start")
    end = right.date_input("종료일", value=dt.date.today(), min_value=dt.date(1982, 1, 2), key="direct_end")
    return start, end


def _strategy_controls(base: PortfolioPreset) -> PortfolioPreset:
    version = st.session_state.get("form_version", 0)
    prefix = f"strategy_{base.name}_{version}"
    options = _asset_options(base.symbols)

    with st.container(border=True):
        st.subheader("🎯 투자자산 선택")
        attack_tickers = tuple(st.multiselect(
            "투자자산 후보군", options, default=list(base.attack_tickers),
            format_func=lambda x: display_name(x, "미국 ETF"), key=f"{prefix}_attack_assets",
        ))
        attack_mode = st.radio("모멘텀 방식", ATTACK_MODES, index=ATTACK_MODES.index(base.attack_mode), key=f"{prefix}_attack_mode")
        attack_count = int(st.number_input(
            "선택자산 수", 1, max(1, len(attack_tickers)), min(base.attack_select_count, max(1, len(attack_tickers))),
            key=f"{prefix}_attack_count",
        ))
        attack_momentum = _momentum_editor("모멘텀 산출 기준", base.attack_relative, f"{prefix}_attack_momentum")

    with st.container(border=True):
        st.subheader("⚖️ 포트폴리오 비중")
        weight_modes = ("고정 범위", "마이너스 절대 모멘텀 수", "플러스 절대 모멘텀 자산 개수 비중 한도")
        weight_mode = st.radio("공격자산 비중 제한", weight_modes, index=weight_modes.index(base.weight_mode), key=f"{prefix}_weight_mode")
        fixed_defense = base.fixed_defense_weight
        negative_threshold = base.negative_threshold
        if weight_mode == "고정 범위":
            fixed_defense = st.slider("고정 방어자산 비중", 0, 100, int(base.fixed_defense_weight * 100), 5, key=f"{prefix}_fixed_defense") / 100
        elif weight_mode == "마이너스 절대 모멘텀 수":
            negative_threshold = int(st.number_input(
                "전체 방어 전환 기준 자산 수", 1, max(1, len(attack_tickers)),
                min(base.negative_threshold, max(1, len(attack_tickers))), key=f"{prefix}_negative_threshold",
            ))
            st.caption("원시 모멘텀 값이 정확히 0이면 음수로 세지 않습니다. 판단값은 반올림하지 않습니다.")

    with st.container(border=True):
        st.subheader("🕊️ 위험 회피 옵션 A · 카나리아 ETF")
        canary_enabled = st.toggle("카나리아 위험 회피 사용", value=bool(base.canary_tickers), key=f"{prefix}_canary_enabled")
        canary_tickers: tuple[str, ...] = ()
        canary_momentum = base.canary_momentum
        if canary_enabled:
            canary_tickers = tuple(st.multiselect(
                "카나리아 유니버스", options, default=list(base.canary_tickers),
                format_func=lambda x: display_name(x, "미국 ETF"), key=f"{prefix}_canary_assets",
            ))
            canary_momentum = _momentum_editor("카나리아 모멘텀", base.canary_momentum, f"{prefix}_canary_momentum")
            st.caption("방어형: 하나라도 위험이면 공격자산을 100% 방어자산으로 전환합니다.")
            # Streamlit Cloud can retain an older preset instance in a live
            # session while hot-reloading this file. Treat the newly added
            # setting as raw-value mode when that legacy object lacks it.
            decision_digits = getattr(base, "canary_decision_percent_digits", None)
            if decision_digits is None:
                st.caption("카나리아 부호는 반올림하지 않은 원시값으로 판정합니다.")
            else:
                st.caption(
                    f"카나리아 원시값을 보존하고, 퍼센트 소수점 "
                    f"{decision_digits + 1}자리에서 반올림한 "
                    f"{decision_digits}자리 값을 부호 판정에 사용합니다."
                )

    with st.container(border=True):
        st.subheader("🚩 위험 회피 옵션 B · 시장 데이터")
        st.caption("검증을 마친 10년-2년, 10년-3개월 금리차와 S&P500 배당수익률만 활성화했습니다.")
        selected_signals = tuple(st.multiselect(
            "시장 위험 신호", TESTED_MARKET_SIGNALS, default=list(base.market_signals), key=f"{prefix}_market_signals",
        ))
        combine = st.radio("신호 결합", ("OR", "AND"), index=0 if base.market_combine == "OR" else 1, horizontal=True, key=f"{prefix}_combine")
        spread_10y2y = base.spread_10y2y
        spread_10y3m = base.spread_10y3m
        dividend_threshold = base.dividend_threshold
        if "장단기 금리차 역전 (10년-2년)" in selected_signals:
            source = base.spread_10y2y or SpreadRule(0, 3, 0.0)
            a, b, c = st.columns(3)
            spread_10y2y = SpreadRule(int(a.number_input("10Y-2Y 반영 지연(개월)", 0, 24, source.lag_months, key=f"{prefix}_2y_lag")), int(b.number_input("10Y-2Y 반영 기간(개월)", 1, 24, source.active_months, key=f"{prefix}_2y_active")), float(c.number_input("10Y-2Y 역전값(%p)", value=source.threshold, step=0.1, key=f"{prefix}_2y_threshold")))
        if "장단기 금리차 역전 (10년-3개월)" in selected_signals:
            source = base.spread_10y3m or SpreadRule(7, 8, -0.5)
            a, b, c = st.columns(3)
            spread_10y3m = SpreadRule(int(a.number_input("10Y-3M 반영 지연(개월)", 0, 24, source.lag_months, key=f"{prefix}_3m_lag")), int(b.number_input("10Y-3M 반영 기간(개월)", 1, 24, source.active_months, key=f"{prefix}_3m_active")), float(c.number_input("10Y-3M 역전값(%p)", value=source.threshold, step=0.1, key=f"{prefix}_3m_threshold")))
        if "S&P500 배당수익률" in selected_signals:
            dividend_threshold = float(st.number_input("S&P500 기준 배당수익률(%)", 0.0, 10.0, float(base.dividend_threshold), 0.01, key=f"{prefix}_dividend"))

    with st.container(border=True):
        st.subheader("🛡️ 방어자산 선택기준")
        defense_tickers = tuple(st.multiselect(
            "방어자산 후보군", options, default=list(base.defense_tickers),
            format_func=lambda x: display_name(x, "미국 ETF"), key=f"{prefix}_defense_assets",
        ))
        defense_mode = st.radio("선택 기준", DEFENSE_MODES, index=DEFENSE_MODES.index(base.defense_mode), key=f"{prefix}_defense_mode")
        defense_count = int(st.number_input(
            "방어 선택자산 수", 1, max(1, len(defense_tickers)), min(base.defense_select_count, max(1, len(defense_tickers))),
            key=f"{prefix}_defense_count",
        ))
        same_momentum = st.checkbox("투자자산 선택기준과 동일 모멘텀 적용", value=base.defense_same_momentum, key=f"{prefix}_same_momentum")
        defense_momentum = attack_momentum if same_momentum else _momentum_editor("방어자산 모멘텀", base.defense_relative, f"{prefix}_defense_momentum")

    if not attack_tickers:
        st.error("투자자산 후보군을 하나 이상 선택해야 합니다.")
    if not defense_tickers:
        st.error("방어자산 후보군을 하나 이상 선택해야 합니다.")

    return replace(
        base,
        attack_tickers=attack_tickers,
        defense_tickers=defense_tickers,
        attack_mode=attack_mode,
        attack_select_count=min(attack_count, max(1, len(attack_tickers))),
        attack_relative=attack_momentum,
        attack_absolute=attack_momentum,
        defense_mode=defense_mode,
        defense_select_count=min(defense_count, max(1, len(defense_tickers))),
        defense_relative=defense_momentum,
        defense_absolute=defense_momentum,
        defense_same_momentum=same_momentum,
        weight_mode=weight_mode,
        fixed_defense_weight=fixed_defense,
        negative_threshold=min(negative_threshold, max(1, len(attack_tickers))),
        canary_tickers=canary_tickers,
        canary_momentum=canary_momentum,
        market_combine=combine,
        market_signals=selected_signals,
        spread_10y2y=spread_10y2y,
        spread_10y3m=spread_10y3m,
        dividend_threshold=dividend_threshold,
    )


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _prices(symbols: tuple[str, ...], start_iso: str, end_iso: str) -> PriceBundle:
    return download_yahoo_prices(symbols, pd.Timestamp(start_iso), pd.Timestamp(end_iso))


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _market_data(signals: tuple[str, ...], start_iso: str, end_iso: str, mode: str) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    spread_ids = set()
    if "장단기 금리차 역전 (10년-2년)" in signals:
        spread_ids.add("T10Y2Y")
    if "장단기 금리차 역전 (10년-3개월)" in signals:
        spread_ids.add("T10Y3M")
    result.update(load_treasury_spreads(spread_ids, start_iso, end_iso, mode=mode))
    if "S&P500 배당수익률" in signals:
        result["SP500_DIV_YIELD"] = load_sp500_dividend_yield(start_iso, end_iso, mode=mode)
    return result


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _latest_dividend_snapshot() -> pd.DataFrame:
    return download_latest_sp500_dividend_snapshot()


@st.cache_resource(show_spinner=False)
def _saved_runs() -> dict[str, tuple[float, SavedRun, dict, PriceBundle]]:
    """Process-local result store keyed only by an opaque, unguessable ID."""
    return {}


def _remember_run(saved: SavedRun, result: dict, bundle: PriceBundle) -> str:
    store = _saved_runs()
    now = time.time()
    # Bound memory use and discard records older than seven days.
    expired = [key for key, (created, *_rest) in store.items() if now - created > 7 * 24 * 60 * 60]
    for key in expired:
        store.pop(key, None)
    while len(store) >= 20:
        oldest = min(store, key=lambda key: store[key][0])
        store.pop(oldest, None)
    run_id = secrets.token_urlsafe(24)
    store[run_id] = (now, saved, result, bundle)
    return run_id


def _restore_run(run_id: str | None) -> tuple[SavedRun, dict, PriceBundle] | None:
    if not run_id or len(run_id) > 128:
        return None
    record = _saved_runs().get(run_id)
    if record is None:
        return None
    created, saved, result, bundle = record
    if time.time() - created > 7 * 24 * 60 * 60:
        _saved_runs().pop(run_id, None)
        return None
    return saved, result, bundle


def _percent(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.2%}"


def _monthly_returns(nav: pd.Series) -> pd.DataFrame:
    monthly = nav.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()
    frame = monthly.to_frame("return")
    frame["연도"] = frame.index.year
    frame["월"] = frame.index.month
    pivot = frame.pivot(index="연도", columns="월", values="return")
    pivot.columns = [f"{month}월" for month in pivot.columns]
    return pivot


def _selection_table(selection: object, sleeve_weight: float = 1.0) -> pd.DataFrame:
    rows = []
    if not isinstance(selection, str):
        return pd.DataFrame(columns=["종목", "전략 내부 비중", "전체 포트폴리오 비중"])
    for item in selection.split(", "):
        try:
            ticker, percent = item.rsplit(" ", 1)
            internal = float(percent.rstrip("%")) / 100.0
        except (ValueError, AttributeError):
            continue
        rows.append({
            "종목": domestic_label(ticker) if ticker.endswith(".KS") else ticker,
            "전략 내부 비중": f"{internal:.1%}",
            "전체 포트폴리오 비중": f"{internal * sleeve_weight:.1%}",
        })
    return pd.DataFrame(rows)


def _return_cell_style(value: object) -> str:
    """Color monthly returns without matplotlib, which is absent on Cloud."""
    if pd.isna(value):
        return ""
    number = float(value)
    if number > 0:
        strength = min(abs(number) / 0.10, 1.0)
        return f"background-color: rgba(46, 160, 90, {0.10 + strength * 0.30:.3f}); color: #174c2b"
    if number < 0:
        strength = min(abs(number) / 0.10, 1.0)
        return f"background-color: rgba(220, 70, 70, {0.10 + strength * 0.30:.3f}); color: #6f2020"
    return "background-color: #f4f5f8; color: #4d5565"


def _market_snapshot_table(result: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    market = result.get("engine_config", {}).get("_market_data", {})
    for indicator, series in market.items():
        observations = series.attrs.get("observation_dates", {})
        for date, value in series.items():
            month = str(pd.Timestamp(date).to_period("M"))
            rows.append({
                "관측 월": month,
                "지표": indicator,
                "원시값": float(value),
                "실제 관측일": observations.get(month),
                "출처": series.attrs.get("source", "-"),
                "스냅샷 ID": series.attrs.get("snapshot_id", "-"),
                "모드": series.attrs.get("snapshot_mode", "-"),
            })
    return pd.DataFrame(rows)


def _show_results(result: dict, bundle: PriceBundle) -> None:
    metrics = result["metrics"]
    st.header("백테스트 결과")
    labels = (
        ("누적 수익률", "total_return"), ("연환산 수익률", "cagr"),
        ("연 변동성", "annual_volatility"), ("최대 낙폭", "max_drawdown"),
        ("연 수익률 표준편차", "annual_return_std"), ("최악의 해", "worst_year"),
    )
    cards = "".join(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{_percent(metrics[key])}</div></div>'
        for label, key in labels
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)
    st.caption(f"실제 공통 평가기간: {result['common_start'].date()} ~ {result['common_end'].date()} · 연간 턴오버 {result['annual_turnover']:.2%}")
    st.download_button(
        "비교 진단자료 한 번에 다운로드 (ZIP)",
        diagnostic_zip(result),
        file_name=f"snowball_diagnostics_{result['common_start'].date()}_{result['common_end'].date()}.zip",
        mime="application/zip",
        use_container_width=True,
        help="월별 수익률, 리밸런싱, 자산·카나리아·시장 신호, 일별 NAV와 낙폭이 포함됩니다.",
    )
    exports = diagnostic_frames(result)

    tabs = st.tabs(("수익률 차트", "손실폭", "월별 수익률", "리밸런싱", "판단 원시값", "데이터 스냅샷", "적용 설정"))
    with tabs[0]:
        nav = result["nav"]
        chart = ((nav / nav.iloc[0]) - 1.0).rename("포트폴리오").to_frame()
        benchmark = result.get("benchmark_ticker", "QQQ")
        if benchmark in bundle.adjusted_close:
            qqq = bundle.adjusted_close[benchmark].reindex(nav.index).ffill().dropna()
            if not qqq.empty:
                label = "국내 나스닥100" if benchmark.endswith(".KS") else "QQQ"
                chart.loc[qqq.index, label] = qqq / qqq.iloc[0] - 1.0
        st.line_chart(chart, y_label="누적 수익률")
    with tabs[1]:
        st.line_chart(result["drawdown"].to_frame(), y_label="낙폭")
    with tabs[2]:
        monthly = _monthly_returns(result["nav"])
        if monthly.empty:
            st.info("월별 수익률을 계산할 기간이 부족합니다.")
        else:
            styled = monthly.style.format("{:.2%}").map(_return_cell_style)
            st.dataframe(styled, use_container_width=True)
            st.download_button(
                "월별 수익률 CSV 다운로드",
                exports["01_monthly_returns.csv"].to_csv(index=False).encode("utf-8-sig"),
                file_name="monthly_returns.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with tabs[3]:
        if result.get("preset_name") in COMBINED_PRESET_NAMES and not result["history"].empty:
            history = result["history"]
            first_label, second_label = result.get("sleeve_labels", ("프리셋 1", "프리셋 2"))
            applied_dates = history["적용일"].astype(str).tolist()
            selected_date = st.selectbox(
                "확인할 리밸런싱 적용일", applied_dates,
                index=len(applied_dates) - 1, key="combined_rebalance_date",
            )
            row = history.loc[history["적용일"].astype(str) == selected_date].iloc[-1]
            st.markdown("**최종 결합 종목**")
            final_table = _selection_table(row["선택된 종목"])
            if not final_table.empty:
                st.dataframe(
                    final_table[["종목", "전략 내부 비중"]].rename(
                        columns={"전략 내부 비중": "최종 비중"},
                    ),
                    use_container_width=True, hide_index=True,
                )
            st.markdown(f"**{first_label} 원본 선택 · 전체의 {float(row[f'{first_label} 비중']):.0%} 배정**")
            st.dataframe(
                _selection_table(row[f"{first_label} 선택"], float(row[f"{first_label} 비중"])),
                use_container_width=True, hide_index=True,
            )
            st.markdown(f"**{second_label} 원본 선택 · 전체의 {float(row[f'{second_label} 비중']):.0%} 배정**")
            st.dataframe(
                _selection_table(row[f"{second_label} 선택"], float(row[f"{second_label} 비중"])),
                use_container_width=True, hide_index=True,
            )
            if row[f"{first_label} 선택"] == row[f"{second_label} 선택"]:
                st.info("이 달은 두 프리셋이 실제로 같은 종목을 선택했습니다. 각 전략은 독립 계산되며 비중만 합산됩니다.")
            st.markdown("**전체 리밸런싱 기록**")
        st.dataframe(result["history"], use_container_width=True, hide_index=True)
        st.download_button(
            "리밸런싱 내역 CSV 다운로드",
            exports["02_rebalancing.csv"].to_csv(index=False).encode("utf-8-sig"),
            file_name="rebalancing.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with tabs[4]:
        keys = list(result["candidate_snapshots"])
        if keys:
            selected = st.selectbox("리밸런싱 적용일", keys, index=len(keys) - 1)
            st.markdown("**자산 선택 계산값**")
            st.dataframe(result["candidate_snapshots"][selected], use_container_width=True, hide_index=True)
            if not result["canary_audits"][selected].empty:
                st.markdown("**카나리아 원시 신호**")
                st.dataframe(result["canary_audits"][selected], use_container_width=True, hide_index=True)
            if not result["market_audits"][selected].empty:
                st.markdown("**시장 데이터 원시 신호**")
                st.dataframe(result["market_audits"][selected], use_container_width=True, hide_index=True)
            st.caption("판단은 반올림하지 않은 원시값으로 수행합니다. 화면 표시는 가독성을 위해 단순화될 수 있습니다.")
            st.download_button(
                "전체 판단 원시값 CSV 다운로드",
                pd.concat([
                    exports["03_asset_signals.csv"],
                    exports["04_canary_signals.csv"],
                    exports["05_market_signals.csv"],
                ], ignore_index=True, sort=False).to_csv(index=False).encode("utf-8-sig"),
                file_name="decision_raw_values.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with tabs[5]:
        metadata = result.get("data_snapshot", {})
        st.markdown("**이번 실행에 사용된 데이터 버전**")
        st.json(metadata, expanded=False)
        snapshot = _market_snapshot_table(result)
        if snapshot.empty:
            st.info("선택한 전략에는 별도의 시장 데이터 지표가 없습니다.")
        else:
            st.dataframe(snapshot, use_container_width=True, hide_index=True)
            st.download_button(
                "시장 데이터 스냅샷 CSV 다운로드",
                snapshot.to_csv(index=False).encode("utf-8-sig"),
                file_name="market_data_snapshot.csv",
                mime="text/csv",
                use_container_width=True,
            )
        left, right = st.columns(2)
        left.download_button(
            "수정종가 CSV 다운로드",
            bundle.adjusted_close.rename_axis("Date").to_csv().encode("utf-8-sig"),
            file_name=f"adjusted_close_{metadata.get('price_snapshot_id', 'snapshot')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        right.download_button(
            "수정시가 CSV 다운로드",
            bundle.adjusted_open.rename_axis("Date").to_csv().encode("utf-8-sig"),
            file_name=f"adjusted_open_{metadata.get('price_snapshot_id', 'snapshot')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        market = result.get("engine_config", {}).get("_market_data", {})
        if "SP500_DIV_YIELD" in market and st.toggle("최신 Multpl 값과 비교", key="compare_latest_dividend"):
            try:
                with st.spinner("최신 배당수익률 자료와 비교하고 있습니다…"):
                    latest = _latest_dividend_snapshot()
                comparison = compare_dividend_snapshots(
                    market["SP500_DIV_YIELD"], latest,
                    result["common_start"].date().isoformat(), result["common_end"].date().isoformat(),
                )
                changed = comparison[comparison["상태"] != "동일"]
                st.metric("변경·신규 관측치", len(changed))
                st.dataframe(changed if not changed.empty else comparison.tail(12), use_container_width=True, hide_index=True)
                st.download_button(
                    "최신 후보 스냅샷 CSV 다운로드",
                    latest.rename_axis("Date").reset_index().to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"sp500_dividend_{str(latest.attrs.get('sha256', 'latest'))[:12]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption("고정 스냅샷은 자동으로 덮어쓰지 않습니다. 최신 후보를 검토한 뒤 별도 버전으로 승격합니다.")
            except Exception as error:
                st.error(f"최신 스냅샷을 비교하지 못했습니다: {error}")
    with tabs[6]:
        config = dict(result["engine_config"])
        config.pop("execution_prices", None)
        config.pop("_market_data", None)
        st.json(config, expanded=False)


def _execute(saved: SavedRun) -> tuple[dict, PriceBundle]:
    saved.validate()
    is_domestic = getattr(saved, "domestic_execution", False)
    warmup = pd.Timestamp(saved.start_date) - pd.DateOffset(months=18)
    secondary_symbols = saved.secondary_preset.symbols if saved.secondary_preset is not None else ()
    symbols = tuple(dict.fromkeys(saved.preset.symbols + secondary_symbols + ("QQQ",)))
    signal_bundle = _prices(symbols, warmup.date().isoformat(), saved.end_date.isoformat())
    market_mode = getattr(saved, "market_data_mode", "fixed")
    secondary_signals = saved.secondary_preset.market_signals if saved.secondary_preset is not None else ()
    signals = tuple(dict.fromkeys(saved.preset.market_signals + secondary_signals))
    market = _market_data(signals, warmup.date().isoformat(), saved.end_date.isoformat(), market_mode)
    if is_domestic:
        execution_us_symbols = tuple(dict.fromkeys(
            saved.preset.attack_tickers + saved.preset.defense_tickers
            + (() if saved.secondary_preset is None else (
                saved.secondary_preset.attack_tickers + saved.secondary_preset.defense_tickers
            ))
            + ("QQQ",)
        ))
        execution_bundle = _prices(
            domestic_symbols(execution_us_symbols),
            warmup.date().isoformat(), saved.end_date.isoformat(),
        )
        if saved.secondary_preset is None:
            domestic_name = (
                DOMESTIC_PRESET_1 if saved.preset.name == PRESET_NAMES[0]
                else DOMESTIC_PRESET_2
            )
            result = run_domestic_preset(
                saved.preset, signal_bundle, execution_bundle, market,
                saved.start_date, saved.end_date,
                initial_investment=saved.initial_investment,
                transaction_cost_rate=saved.transaction_cost_rate,
                preset_name=domestic_name,
            )
        else:
            result = run_domestic_blended_presets(
                saved.preset, saved.secondary_preset, saved.preset1_weight,
                signal_bundle, execution_bundle, market,
                saved.start_date, saved.end_date,
                initial_investment=saved.initial_investment,
                transaction_cost_rate=saved.transaction_cost_rate,
            )
        bundle = execution_bundle
    elif saved.secondary_preset is None:
        bundle = signal_bundle
        result = run_preset(
            saved.preset, bundle, market, saved.start_date, saved.end_date,
            initial_investment=saved.initial_investment,
            transaction_cost_rate=saved.transaction_cost_rate,
        )
    else:
        bundle = signal_bundle
        is_alternative_blend = saved.secondary_preset.name == PRESET_7_NAME
        result = run_blended_presets(
            saved.preset, saved.secondary_preset, saved.preset1_weight,
            bundle, market, saved.start_date, saved.end_date,
            initial_investment=saved.initial_investment,
            transaction_cost_rate=saved.transaction_cost_rate,
            preset_name=(ALTERNATIVE_COMBINED_PRESET_NAME if is_alternative_blend else COMBINED_PRESET_NAME),
            second_label=("프리셋 7" if is_alternative_blend else "프리셋 2"),
        )
    price_hash = hashlib.sha256(
        pd.util.hash_pandas_object(bundle.adjusted_close, index=True).values.tobytes()
        + pd.util.hash_pandas_object(bundle.adjusted_open, index=True).values.tobytes()
    ).hexdigest()[:12]
    result["data_snapshot"] = {
        "price_source": bundle.source,
        "price_snapshot_id": f"yahoo-{price_hash}",
        "market_data_mode": market_mode,
        "market_snapshots": {
            name: series.attrs.get("snapshot_id", "-") for name, series in market.items()
        },
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return result, bundle


def main() -> None:
    _css()
    st.title("❄️ Snowball Portfolio Lab")
    st.write("연구 프리셋을 불러오거나 조건을 조정해 동적 포트폴리오를 검증합니다.")
    restore_id = st.query_params.get("run_id")
    restored = _restore_run(restore_id)
    if restored is not None and "last_result" not in st.session_state:
        saved, result, bundle = restored
        st.session_state.last_result = result
        st.session_state.last_bundle = bundle
        st.session_state.last_saved_run = saved
        if saved.secondary_preset is not None:
            if getattr(saved, "domestic_execution", False):
                st.session_state.preset_name = DOMESTIC_COMBINED_PRESET_NAME
            elif saved.secondary_preset.name == PRESET_7_NAME:
                st.session_state.preset_name = ALTERNATIVE_COMBINED_PRESET_NAME
            else:
                st.session_state.preset_name = COMBINED_PRESET_NAME
            st.session_state.combined_preset1_weight = int(round(saved.preset1_weight * 100))
        elif getattr(saved, "domestic_execution", False):
            st.session_state.preset_name = (
                DOMESTIC_PRESET_1 if saved.preset.name == PRESET_NAMES[0]
                else DOMESTIC_PRESET_2
            )
        elif saved.preset.name in PRESETS:
            st.session_state.preset_name = saved.preset.name

    preset_name = _preset_buttons()
    start_date, end_date = _period_controls()
    combined_weight = 50
    if preset_name in COMBINED_PRESET_NAMES:
        with st.container(border=True):
            st.subheader("⚖️ 결합 비율")
            combined_weight = int(st.number_input(
                "프리셋 1 비중(%)", min_value=0, max_value=100,
                value=50, step=5,
                key="combined_preset1_weight",
            ))
            second_number = 7 if preset_name == ALTERNATIVE_COMBINED_PRESET_NAME else 2
            st.caption(
                f"프리셋 1 {combined_weight}% + 프리셋 {second_number} {100 - combined_weight}%로 "
                "각 전략의 리밸런싱 종목과 일별 성과를 결합합니다."
            )
        edited = PRESETS[PRESET_NAMES[0]]
    else:
        if preset_name == DOMESTIC_PRESET_1:
            base = PRESETS[PRESET_NAMES[0]]
        elif preset_name == DOMESTIC_PRESET_2:
            base = PRESETS[PRESET_NAMES[1]]
        else:
            base = PRESETS[preset_name]
        with st.expander("전략 조건 편집", expanded=True):
            edited = _strategy_controls(base)

    with st.container(border=True):
        st.subheader("⚙️ 기본 설정")
        left, right = st.columns(2)
        domestic_execution = preset_name in DOMESTIC_PRESET_NAMES
        if domestic_execution:
            initial = float(left.number_input(
                "초기 투자금(KRW)", min_value=100_000.0, value=10_000_000.0,
                step=1_000_000.0, key="domestic_initial",
            ))
        else:
            initial = float(left.number_input(
                "초기 투자금(USD)", min_value=100.0, value=10_000.0,
                step=1_000.0, key="us_initial",
            ))
        cost_bps = float(right.number_input("편도 거래비용(bp)", min_value=0.0, max_value=500.0, value=0.0, step=1.0))
        mode_label = st.radio(
            "시장 데이터 버전",
            tuple(MARKET_DATA_MODES),
            horizontal=True,
            help="고정 스냅샷은 같은 과거 결과를 재현하고, 최신 데이터는 새 공급원 값을 별도 버전으로 사용합니다.",
        )

    if st.button("백테스트 실행", type="primary", use_container_width=True):
        if start_date >= end_date:
            st.error("시작일은 종료일보다 앞서야 합니다.")
            return
        try:
            edited.validate()
        except ValueError as error:
            st.error(str(error))
            return
        saved = SavedRun(
            preset=edited,
            start_date=start_date,
            end_date=end_date,
            initial_investment=initial,
            transaction_cost_rate=cost_bps / 10_000.0,
            market_data_mode=MARKET_DATA_MODES[mode_label],
            secondary_preset=(
                PRESETS[PRESET_7_NAME]
                if preset_name == ALTERNATIVE_COMBINED_PRESET_NAME
                else PRESETS[PRESET_NAMES[1]] if preset_name in COMBINED_PRESET_NAMES else None
            ),
            preset1_weight=combined_weight / 100.0 if preset_name in COMBINED_PRESET_NAMES else 1.0,
            domestic_execution=domestic_execution,
        )
        try:
            with st.spinner("가격과 시장 데이터를 불러와 계산하고 있습니다…"):
                result, bundle = _execute(saved)
            st.session_state.last_result = result
            st.session_state.last_bundle = bundle
            st.session_state.last_saved_run = saved
            st.query_params["run_id"] = _remember_run(saved, result, bundle)
        except Exception as error:  # Streamlit must turn provider failures into actionable UI feedback.
            st.error(f"백테스트를 완료하지 못했습니다: {error}")

    if "last_result" in st.session_state:
        st.caption("마지막 결과는 새로고침하거나 이 탭을 다시 열어도 자동으로 복원됩니다.")
        _show_results(st.session_state.last_result, st.session_state.last_bundle)


if __name__ == "__main__":
    main()
