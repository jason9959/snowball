"""Streamlit entry point for Snowball Portfolio Lab."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import secrets
import time
from dataclasses import replace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from snowball import account_store as _account_store
from snowball.accounting import (
    account_period_metrics,
    calculate_account_valuation,
    download_unadjusted_closes,
    required_price_symbols,
    transaction_frame,
)
from snowball.core.catalog import KR_ETFS, US_ETFS, display_name
from snowball.core.engine import resolve_recent_period
from snowball.app_state import SavedRun
from snowball.data.market_data import (
    compare_dividend_snapshots,
    download_latest_sp500_dividend_snapshot,
    load_strategy_market_data,
)
from snowball.data.prices import PriceBundle, download_yahoo_prices
from snowball.domestic import DOMESTIC_EXECUTION_MAP, DOMESTIC_NAMES, domestic_label, domestic_symbols
from snowball.exports import diagnostic_frames, diagnostic_zip, market_data_export
from snowball.presets import MomentumSpec, PRESETS, PortfolioPreset, SpreadRule
from snowball.service import (
    run_blended_presets,
    run_domestic_blended_presets,
    run_domestic_preset,
    run_preset,
)
from snowball.user_presets import UserPreset, export_user_presets, import_user_presets


# Streamlit Cloud can keep an already-imported module during a hot reload.
# Reload this small schema module so a deployment and its UI always use the same version.
_account_store = importlib.reload(_account_store)
DEFAULT_ACCOUNT_TAB = _account_store.DEFAULT_ACCOUNT_TAB
TRADE_COLUMNS = _account_store.TRADE_COLUMNS
GoogleSheetAccountStore = _account_store.GoogleSheetAccountStore
SheetSchemaError = _account_store.SheetSchemaError
account_display_name = _account_store.account_display_name


st.set_page_config(page_title="Snowball Portfolio Lab", page_icon="❄️", layout="wide")

PRESET_NAMES = tuple(PRESETS)
PRESET_7_NAME = "7. 집중투자 공격형 1-3-6-9-12"
PRESET_9_NAME = "9. MDD 개선 공격형"
COMBINED_PRESET_NAME = "3. 프리셋 결합"
DOMESTIC_PRESET_1 = "4. 국내 분산투자 방어형"
DOMESTIC_PRESET_2 = "5. 국내 집중투자 공격형"
DOMESTIC_COMBINED_PRESET_NAME = "6. 국내 프리셋 결합"
ALTERNATIVE_COMBINED_PRESET_NAME = "8. 프리셋 1 + 7 결합"
OPTIMIZED_COMBINED_PRESET_NAME = "10. 프리셋 1 + 9 결합"
COMBINED_PRESET_NAMES = {
    COMBINED_PRESET_NAME, DOMESTIC_COMBINED_PRESET_NAME,
    ALTERNATIVE_COMBINED_PRESET_NAME, OPTIMIZED_COMBINED_PRESET_NAME,
}
DOMESTIC_PRESET_NAMES = {
    DOMESTIC_PRESET_1, DOMESTIC_PRESET_2, DOMESTIC_COMBINED_PRESET_NAME,
}
USER_PRESET_PREFIX = "사용자::"
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
        .home-hero {padding:1.5rem 1.2rem; border-radius:22px; color:white; margin:.4rem 0 1.2rem;
          background:linear-gradient(135deg,#3157d5,#6b4ce6); box-shadow:0 12px 34px rgba(49,87,213,.18);}
        .home-kicker {font-size:.78rem; letter-spacing:.08em; opacity:.78; margin-bottom:.35rem;}
        .home-title {font-size:2rem; line-height:1.2; font-weight:800; margin:0 0 .45rem;}
        .home-copy {font-size:.98rem; line-height:1.55; opacity:.9; margin:0;}
        .feature-copy {min-height:3.2rem; color:#626b7c; font-size:.9rem; line-height:1.5;}
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
          .home-hero {padding:1.25rem 1rem; border-radius:18px;}
          .home-title {font-size:1.6rem;}
          .feature-copy {min-height:auto;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _go_to(page: str) -> None:
    st.session_state.app_page = page
    if page == "home":
        st.query_params.clear()
    st.rerun()


@st.cache_resource(show_spinner=False)
def _cached_google_sheet_store(
    spreadsheet_id: str,
    service_account_json: str,
    schema_version: int,
) -> GoogleSheetAccountStore:
    del schema_version  # Used only to invalidate Streamlit's resource cache after schema changes.
    store = GoogleSheetAccountStore.connect(spreadsheet_id, service_account_json)
    store.ensure_schema()
    return store


def _google_sheet_store() -> tuple[GoogleSheetAccountStore | None, str | None]:
    try:
        spreadsheet_id = str(st.secrets.get("GOOGLE_SHEET_ID", "")).strip()
        service_account_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    except Exception:
        return None, None
    if not spreadsheet_id or not service_account_json:
        return None, None
    if not isinstance(service_account_json, str):
        import json
        service_account_json = json.dumps(dict(service_account_json))
    try:
        return _cached_google_sheet_store(spreadsheet_id, service_account_json, 3), None
    except (ValueError, RuntimeError, SheetSchemaError) as error:
        return None, str(error)
    except Exception as error:
        return None, f"Google Sheet에 연결하지 못했습니다: {error}"


def _account_sheet_cache_key(store: GoogleSheetAccountStore) -> str:
    """Return a non-secret key that separates cached data by spreadsheet."""
    spreadsheet_id = str(getattr(store.spreadsheet, "id", store.title))
    return hashlib.sha256(spreadsheet_id.encode("utf-8")).hexdigest()[:16]


@st.cache_data(ttl=5 * 60, max_entries=4, show_spinner=False)
def _cached_account_tabs(
    sheet_cache_key: str,
    _store: GoogleSheetAccountStore,
) -> tuple[str, ...]:
    del sheet_cache_key
    return _store.list_account_tabs()


@st.cache_data(ttl=60, max_entries=32, show_spinner=False)
def _cached_account_transactions(
    sheet_cache_key: str,
    account_tab: str,
    _store: GoogleSheetAccountStore,
) -> list[dict[str, object]]:
    del sheet_cache_key
    return _store.load_transactions(account_tab)


def _clear_account_sheet_cache(*, include_tabs: bool = False) -> None:
    """Invalidate account data after a write or an explicit user refresh."""
    _cached_account_transactions.clear()
    if include_tabs:
        _cached_account_tabs.clear()


@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def _cached_account_prices(
    symbols: tuple[str, ...],
    start_text: str,
    end_text: str,
) -> tuple[pd.DataFrame, pd.Series]:
    return download_unadjusted_closes(
        symbols, pd.Timestamp(start_text), pd.Timestamp(end_text),
    )


def _home_page() -> None:
    st.markdown(
        """
        <div class="home-hero">
          <div class="home-kicker">SNOWBALL PORTFOLIO LAB</div>
          <div class="home-title">투자 전략과 실제 운용을 한곳에서</div>
          <p class="home-copy">전략을 검증하거나 실제 계좌의 운용 현황을 확인하세요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("📊 백테스팅")
            st.markdown(
                '<div class="feature-copy">프리셋과 사용자 조건을 이용해 수익률, 변동성, '
                "MDD와 리밸런싱 내역을 검증합니다.</div>",
                unsafe_allow_html=True,
            )
            if st.button("백테스팅 시작", type="primary", use_container_width=True):
                _go_to("backtest")
    with right:
        with st.container(border=True):
            st.subheader("📈 투자 계좌 현황")
            st.markdown(
                '<div class="feature-copy">Google Sheet의 거래기록을 바탕으로 평가액, 수익률, '
                "MDD와 보유종목을 확인합니다.</div>",
                unsafe_allow_html=True,
            )
            if st.button("계좌 현황 보기", use_container_width=True):
                _go_to("trades")
    st.caption("거래 입력·수정은 비공개 Google Sheet의 ‘계좌_’ 탭에서 진행합니다.")


def _legacy_trade_entry_page() -> None:
    if st.button("← 기능 선택으로", key="trade_home"):
        _go_to("home")
    st.title("🧾 매수·매도 입력")
    st.write("실제 체결과 현금흐름을 기록하고 월간 운용 결과를 확인합니다.")
    sheet_store, sheet_error = _google_sheet_store()
    st.session_state.setdefault("trade_drafts", [])
    if sheet_store is not None:
        try:
            drafts: list[dict[str, object]] = sheet_store.load_transactions()
            st.success(f"Google Sheet 연결됨: {sheet_store.title}")
            st.caption("이 화면에서 추가하거나 취소한 내역은 Google Sheet의 ‘거래내역’ 탭에 바로 반영됩니다.")
        except Exception as error:
            sheet_store = None
            drafts = st.session_state.trade_drafts
            sheet_error = f"거래내역을 읽지 못했습니다: {error}"
    else:
        drafts = st.session_state.trade_drafts
    if sheet_error:
        st.error(sheet_error)
        st.warning("연결 오류가 해결될 때까지 입력 내용은 이 브라우저 세션에만 임시 저장됩니다.")
    elif sheet_store is None:
        st.info("Google Sheet 연결 전입니다. 현재 내역은 이 브라우저 세션에만 임시 저장됩니다.")
    input_tab, ledger_tab, account_tab, rebalance_tab = st.tabs(
        ("✍️ 거래 입력", "📋 거래내역", "💼 계좌 현황", "🔔 리밸런싱 기록"),
    )

    with input_tab:
        transaction_type = st.radio(
            "거래 유형", ("매수", "매도", "입금", "출금", "배당", "잔액조정"),
            index=0, horizontal=True, key="transaction_type",
        )
        with st.form("trade_entry_form", clear_on_submit=True):
            first, second = st.columns(2)
            trade_date = first.date_input(
                "거래일", value=dt.datetime.now(ZoneInfo("Asia/Seoul")).date(),
            )
            market = second.selectbox("시장", ("한국", "미국"), disabled=transaction_type in {"입금", "출금", "잔액조정"})
            currency = "KRW" if market == "한국" else "USD"
            ticker = ""
            asset_name = ""
            quantity = 0.0
            price = 0.0
            fee = 0.0
            amount = 0.0
            fx_rate = 1.0

            if transaction_type in {"매수", "매도"}:
                first, second = st.columns(2)
                ticker = first.text_input("종목코드", placeholder="예: 329750 또는 QQQ")
                asset_name = second.text_input("종목명(선택)", placeholder="코드가 등록돼 있으면 자동 입력")
                third, fourth = st.columns(2)
                quantity = float(third.number_input("수량", min_value=0.0, step=1.0, format="%.4f"))
                price = float(fourth.number_input(f"체결 단가({currency})", min_value=0.0, step=10.0 if currency == "KRW" else 0.01))
                fifth, sixth = st.columns(2)
                fee = float(fifth.number_input(f"수수료·세금({currency})", min_value=0.0, step=10.0 if currency == "KRW" else 0.01))
                if currency == "USD":
                    fx_rate = float(sixth.number_input("적용 환율(원/달러)", min_value=0.0, value=1_400.0, step=1.0))
                else:
                    sixth.text_input("통화", value="KRW", disabled=True)
                amount = quantity * price
                st.caption(f"예상 거래금액: {amount * fx_rate:,.0f}원 · 수수료 제외")
            elif transaction_type == "배당":
                first, second = st.columns(2)
                ticker = first.text_input("배당 종목코드", placeholder="예: 329750")
                asset_name = second.text_input("종목명(선택)")
                amount = float(st.number_input(f"실제 입금된 배당금({currency})", min_value=0.0, step=100.0 if currency == "KRW" else 0.01))
                if currency == "USD":
                    fx_rate = float(st.number_input("적용 환율(원/달러)", min_value=0.0, value=1_400.0, step=1.0))
            else:
                market = "현금"
                currency = "KRW"
                ticker = "현금"
                if transaction_type == "잔액조정":
                    amount = float(st.number_input(
                        "현금 증감액(KRW)", value=0.0, step=10_000.0,
                        help="현금을 늘리려면 양수, 줄이려면 음수로 입력합니다.",
                    ))
                else:
                    amount = float(st.number_input("금액(KRW)", min_value=0.0, step=10_000.0))

            memo = st.text_input("메모", placeholder="예: 11월 리밸런싱")
            submitted = st.form_submit_button("내역 추가", type="primary", use_container_width=True)

        normalized_ticker = ticker.strip().upper()
        if submitted:
            if transaction_type in {"매수", "매도"} and (
                not normalized_ticker or quantity <= 0 or price <= 0 or fx_rate <= 0
            ):
                st.error("종목코드, 수량, 체결 단가와 환율을 확인하세요.")
            elif transaction_type == "배당" and (not normalized_ticker or amount <= 0 or fx_rate <= 0):
                st.error("배당 종목코드와 실제 입금액을 입력하세요.")
            elif transaction_type == "잔액조정" and abs(amount) <= 1e-10:
                st.error("0이 아닌 조정 금액을 입력하세요.")
            elif transaction_type not in {"매수", "매도", "배당", "잔액조정"} and amount <= 0:
                st.error("금액을 입력하세요.")
            else:
                lookup_code = normalized_ticker
                if lookup_code.isdigit() and len(lookup_code) == 6:
                    lookup_symbol = f"{lookup_code}.KS"
                    inferred_name = DOMESTIC_NAMES.get(lookup_symbol, "")
                else:
                    inferred_name = US_ETFS.get(lookup_code, "")
                record = {
                    "거래ID": secrets.token_hex(6),
                    "입력시각": dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
                    "거래일": trade_date.isoformat(), "유형": transaction_type, "시장": market,
                    "종목코드": normalized_ticker, "종목명": asset_name.strip() or inferred_name,
                    "수량": quantity, "체결단가": price, "수수료·세금": fee,
                    "금액": amount, "통화": currency, "환율": fx_rate,
                    "메모": memo.strip(), "상태": "정상",
                }
                if sheet_store is not None:
                    try:
                        sheet_store.append_transaction(record)
                        st.success("Google Sheet에 내역을 저장했습니다.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"저장하지 못했습니다: {error}")
                else:
                    drafts.append(record)
                    st.success("임시 내역을 추가했습니다. Google Sheet 연결 후에는 영구 저장됩니다.")

    ledger = pd.DataFrame(drafts)
    with ledger_tab:
        if ledger.empty:
            st.info("입력된 거래내역이 없습니다.")
        else:
            left, right = st.columns(2)
            type_filter = left.multiselect("유형 필터", sorted(ledger["유형"].unique()))
            include_cancelled = right.toggle("취소 내역 포함", value=True)
            visible = ledger.copy()
            if type_filter:
                visible = visible[visible["유형"].isin(type_filter)]
            if not include_cancelled:
                visible = visible[visible["상태"] == "정상"]
            st.dataframe(visible, use_container_width=True, hide_index=True)
            st.download_button(
                "CSV 백업 다운로드", visible.to_csv(index=False).encode("utf-8-sig"),
                file_name="snowball_transactions.csv", mime="text/csv", use_container_width=True,
            )
            active = ledger[ledger["상태"] == "정상"]
            if not active.empty:
                label_by_id = {
                    row["거래ID"]: f'{row["거래일"]} · {row["유형"]} · {row["종목코드"] or "현금"}'
                    for _, row in active.iterrows()
                }
                cancel_id = st.selectbox(
                    "취소할 내역", tuple(label_by_id), format_func=label_by_id.get,
                )
                if st.button("선택 내역 취소", use_container_width=True):
                    if sheet_store is not None:
                        try:
                            if not sheet_store.cancel_transaction(str(cancel_id)):
                                st.error("취소할 거래를 Google Sheet에서 찾지 못했습니다.")
                            else:
                                st.success("Google Sheet에서 선택한 내역을 취소 처리했습니다.")
                                st.rerun()
                        except Exception as error:
                            st.error(f"취소 처리하지 못했습니다: {error}")
                    else:
                        for record in drafts:
                            if record["거래ID"] == cancel_id:
                                record["상태"] = "취소"
                                break
                        st.success("선택한 내역을 취소 처리했습니다. 원본 행은 유지됩니다.")
                        st.rerun()

    with account_tab:
        active = ledger[ledger["상태"] == "정상"].copy() if not ledger.empty else ledger
        if active.empty:
            st.info("입금과 거래를 입력하면 예상 현금과 보유수량이 표시됩니다.")
        else:
            active["원화금액"] = active["금액"].astype(float) * active["환율"].astype(float)
            active["원화비용"] = active["수수료·세금"].astype(float) * active["환율"].astype(float)
            cash_change = pd.Series(0.0, index=active.index)
            cash_change[active["유형"] == "입금"] = active.loc[active["유형"] == "입금", "원화금액"]
            cash_change[active["유형"] == "출금"] = -active.loc[active["유형"] == "출금", "원화금액"]
            cash_change[active["유형"] == "배당"] = active.loc[active["유형"] == "배당", "원화금액"]
            cash_change[active["유형"] == "잔액조정"] = active.loc[active["유형"] == "잔액조정", "원화금액"]
            cash_change[active["유형"] == "매수"] = -(
                active.loc[active["유형"] == "매수", "원화금액"]
                + active.loc[active["유형"] == "매수", "원화비용"]
            )
            cash_change[active["유형"] == "매도"] = (
                active.loc[active["유형"] == "매도", "원화금액"]
                - active.loc[active["유형"] == "매도", "원화비용"]
            )
            deposits = active.loc[active["유형"] == "입금", "원화금액"].sum()
            withdrawals = active.loc[active["유형"] == "출금", "원화금액"].sum()
            first, second, third = st.columns(3)
            first.metric("순입금", f"{deposits - withdrawals:,.0f}원")
            second.metric("예상 현금", f"{cash_change.sum():,.0f}원")
            third.metric("누적 비용", f"{active['원화비용'].sum():,.0f}원")

            trades = active[active["유형"].isin(("매수", "매도"))].copy()
            if not trades.empty:
                trades["증감수량"] = np.where(trades["유형"] == "매수", trades["수량"], -trades["수량"])
                holdings = trades.groupby(["시장", "종목코드", "종목명"], dropna=False)["증감수량"].sum().reset_index()
                holdings = holdings[holdings["증감수량"].abs() > 1e-10].rename(columns={"증감수량": "보유수량"})
                st.subheader("보유수량")
                st.dataframe(holdings, use_container_width=True, hide_index=True)
                st.caption("평가금액과 손익은 Google Sheet 연결 후 일반 종가를 받아 계산합니다.")

    with rebalance_tab:
        st.subheader("프리셋 3 월간 운용")
        st.markdown(
            "- **프리셋 1:** 분산투자 방어형 50%\n"
            "- **프리셋 2:** 집중투자 공격형 50%\n"
            "- **운용 시작:** 2026-11-02 시가\n"
            "- **다음 기록:** 월말 미국 종가 확정 후 GitHub Actions가 자동 저장"
        )
        st.info("Google Sheet 연결 후 목표 비중, 실제 비중, 매수·매도 필요 수량과 월간 성과가 여기에 표시됩니다.")


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _transaction_asset_catalog(market: str) -> dict[str, str]:
    if market == "한국":
        combined = {symbol.removesuffix(".KS"): name for symbol, name in KR_ETFS.items()}
        combined.update({symbol.removesuffix(".KS"): name for symbol, name in DOMESTIC_NAMES.items()})
        return dict(sorted(combined.items()))
    return dict(sorted(US_ETFS.items()))


def _transaction_form(
    *,
    prefix: str,
    submit_label: str,
    initial: dict[str, object] | None = None,
) -> dict[str, object] | None:
    initial = initial or {}
    transaction_types = ("매수", "매도", "입금", "출금", "배당", "잔액조정")
    initial_type = str(initial.get("유형", "매수"))
    transaction_type = st.radio(
        "거래 유형",
        transaction_types,
        index=transaction_types.index(initial_type) if initial_type in transaction_types else 0,
        horizontal=True,
        key=f"{prefix}_type",
    )
    market = "현금"
    selected_code = "현금"
    selected_name = ""
    if transaction_type in {"매수", "매도", "배당"}:
        markets = ("한국", "미국")
        initial_market = str(initial.get("시장", "한국"))
        market = st.radio(
            "시장", markets,
            index=markets.index(initial_market) if initial_market in markets else 0,
            horizontal=True, key=f"{prefix}_market",
        )
        catalog = _transaction_asset_catalog(market)
        initial_market = str(initial.get("시장", "한국"))
        current_code = (
            str(initial.get("종목코드", "")).strip().upper().removesuffix(".KS")
            if initial_market == market else ""
        )
        options = list(catalog)
        if current_code and current_code not in options:
            catalog[current_code] = str(initial.get("종목명", "")) or current_code
            options.insert(0, current_code)
        options.append("직접 입력")
        default_index = options.index(current_code) if current_code in options else 0
        selected = st.selectbox(
            "종목코드",
            options,
            index=default_index,
            format_func=lambda code: (
                "직접 입력" if code == "직접 입력" else f"{code} · {catalog[code]}"
            ),
            key=f"{prefix}_asset_{market}",
        )
        if selected == "직접 입력":
            first, second = st.columns(2)
            selected_code = first.text_input(
                "직접 종목코드", value=current_code, key=f"{prefix}_manual_code_{market}",
            ).strip().upper().removesuffix(".KS")
            selected_name = second.text_input(
                "직접 종목명", value=str(initial.get("종목명", "")), key=f"{prefix}_manual_name_{market}",
            ).strip()
        else:
            selected_code = selected
            selected_name = catalog[selected]
            st.caption(f"선택 종목: {selected_code} · {selected_name}")

    with st.form(f"{prefix}_form", clear_on_submit=not bool(initial)):
        first, second = st.columns(2)
        initial_date = pd.to_datetime(initial.get("거래일"), errors="coerce")
        trade_date = first.date_input(
            "거래일",
            value=(
                initial_date.date() if not pd.isna(initial_date)
                else dt.datetime.now(ZoneInfo("Asia/Seoul")).date()
            ),
        )
        currency = "USD" if market == "미국" else "KRW"
        second.text_input("통화", value=currency, disabled=True)
        quantity = price = fee = amount = 0.0
        fx_rate = _as_float(initial.get("환율"), 1_400.0 if currency == "USD" else 1.0)
        if transaction_type in {"매수", "매도"}:
            third, fourth = st.columns(2)
            quantity = float(third.number_input(
                "수량", min_value=0.0, value=_as_float(initial.get("수량")),
                step=1.0, format="%.4f",
            ))
            price = float(fourth.number_input(
                f"체결 단가({currency})", min_value=0.0,
                value=_as_float(initial.get("체결단가")),
                step=10.0 if currency == "KRW" else 0.01,
            ))
            fifth, sixth = st.columns(2)
            fee = float(fifth.number_input(
                f"수수료·세금({currency})", min_value=0.0,
                value=_as_float(initial.get("수수료·세금")),
                step=10.0 if currency == "KRW" else 0.01,
            ))
            if currency == "USD":
                fx_rate = float(sixth.number_input(
                    "적용 환율(원/달러)", min_value=0.0, value=fx_rate, step=1.0,
                ))
            else:
                sixth.text_input("적용 환율", value="1", disabled=True)
            amount = quantity * price
            st.caption(f"예상 거래금액: {amount * fx_rate:,.0f}원 · 수수료 제외")
        elif transaction_type == "배당":
            amount = float(st.number_input(
                f"실제 입금된 배당금({currency})", min_value=0.0,
                value=_as_float(initial.get("금액")),
                step=100.0 if currency == "KRW" else 0.01,
            ))
            if currency == "USD":
                fx_rate = float(st.number_input(
                    "적용 환율(원/달러)", min_value=0.0, value=fx_rate, step=1.0,
                ))
        elif transaction_type == "잔액조정":
            amount = float(st.number_input(
                "현금 증감액(KRW)", value=_as_float(initial.get("금액")), step=10_000.0,
                help="현금을 늘리려면 양수, 줄이려면 음수로 입력합니다.",
            ))
        else:
            amount = float(st.number_input(
                "금액(KRW)", min_value=0.0, value=_as_float(initial.get("금액")),
                step=10_000.0,
            ))
        memo = st.text_input("메모", value=str(initial.get("메모", "")))
        submitted = st.form_submit_button(submit_label, type="primary", use_container_width=True)

    if not submitted:
        return None
    if transaction_type in {"매수", "매도"} and (
        not selected_code or quantity <= 0 or price <= 0 or fx_rate <= 0
    ):
        st.error("종목, 수량, 체결 단가와 환율을 확인하세요.")
        return None
    if transaction_type == "배당" and (not selected_code or amount <= 0 or fx_rate <= 0):
        st.error("배당 종목과 실제 입금액을 확인하세요.")
        return None
    if transaction_type == "잔액조정" and abs(amount) <= 1e-10:
        st.error("0이 아닌 조정 금액을 입력하세요.")
        return None
    if transaction_type in {"입금", "출금"} and amount <= 0:
        st.error("금액을 입력하세요.")
        return None
    return {
        "거래일": trade_date.isoformat(), "유형": transaction_type, "시장": market,
        "종목코드": selected_code, "종목명": selected_name, "수량": quantity,
        "체결단가": price, "수수료·세금": fee, "금액": amount,
        "통화": currency, "환율": fx_rate, "메모": memo.strip(),
    }


def _account_value_chart(history: pd.DataFrame) -> None:
    if history.empty:
        return
    chart = history.reset_index().rename(columns={"index": "날짜"})
    st.vega_lite_chart(
        chart,
        {
            "layer": [
                {
                    "mark": {"type": "line", "color": "#3157d5", "strokeWidth": 2.5},
                    "encoding": {
                        "x": {"field": "날짜", "type": "temporal", "title": None},
                        "y": {"field": "총평가액", "type": "quantitative", "title": "총평가액(원)"},
                        "tooltip": [
                            {"field": "날짜", "type": "temporal"},
                            {"field": "총평가액", "type": "quantitative", "format": ",.0f"},
                        ],
                    },
                },
                {
                    "mark": {"type": "bar", "color": "#f59e0b", "opacity": 0.45},
                    "encoding": {
                        "x": {"field": "날짜", "type": "temporal", "title": None},
                        "y": {"field": "입출금", "type": "quantitative", "title": "입출금(원)"},
                        "tooltip": [
                            {"field": "날짜", "type": "temporal"},
                            {"field": "입출금", "type": "quantitative", "format": ",.0f"},
                        ],
                    },
                },
            ],
            "resolve": {"scale": {"y": "independent"}},
            "height": 310,
        },
        use_container_width=True,
    )


def _account_performance_chart(history: pd.DataFrame) -> None:
    if history.empty or not {"누적수익률", "낙폭"}.issubset(history.columns):
        return
    chart = history.reset_index().rename(columns={"index": "날짜"})
    st.vega_lite_chart(
        chart,
        {
            "layer": [
                {
                    "mark": {"type": "line", "color": "#3157d5", "strokeWidth": 2.5},
                    "encoding": {
                        "x": {"field": "날짜", "type": "temporal", "title": None},
                        "y": {
                            "field": "누적수익률", "type": "quantitative", "title": "수익률",
                            "axis": {"format": ".1%"},
                        },
                        "tooltip": [
                            {"field": "날짜", "type": "temporal"},
                            {"field": "누적수익률", "type": "quantitative", "format": ".2%"},
                        ],
                    },
                },
                {
                    "mark": {"type": "area", "color": "#ef4444", "opacity": 0.18},
                    "encoding": {
                        "x": {"field": "날짜", "type": "temporal", "title": None},
                        "y": {
                            "field": "낙폭", "type": "quantitative", "title": "수익률",
                            "axis": {"format": ".1%"},
                        },
                        "tooltip": [
                            {"field": "날짜", "type": "temporal"},
                            {"field": "낙폭", "type": "quantitative", "format": ".2%"},
                        ],
                    },
                },
            ],
            "height": 280,
        },
        use_container_width=True,
    )


def _account_metric_cards(items: tuple[tuple[str, str], ...]) -> None:
    cards = "".join(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def _legacy_account_management_page() -> None:
    if st.button("← 기능 선택으로", key="trade_home"):
        _go_to("home")
    st.title("🧾 계좌 관리")
    st.write("계좌별 실제 체결과 현금흐름을 기록하고 잔고와 운용 결과를 확인합니다.")
    sheet_store, sheet_error = _google_sheet_store()
    st.session_state.setdefault("trade_drafts_by_account", {DEFAULT_ACCOUNT_TAB: []})
    if sheet_store is not None:
        try:
            sheet_cache_key = _account_sheet_cache_key(sheet_store)
            account_tabs = _cached_account_tabs(sheet_cache_key, sheet_store)
            st.success(f"Google Sheet 연결됨: {sheet_store.title}")
        except Exception as error:
            sheet_store = None
            sheet_cache_key = ""
            account_tabs = (DEFAULT_ACCOUNT_TAB,)
            sheet_error = f"계좌 탭을 읽지 못했습니다: {error}"
    else:
        sheet_cache_key = ""
        account_tabs = tuple(st.session_state.trade_drafts_by_account) or (DEFAULT_ACCOUNT_TAB,)
    if sheet_error:
        st.error(sheet_error)
    elif sheet_store is None:
        st.info("Google Sheet 연결 전입니다. 현재 내역은 이 브라우저 세션에만 임시 저장됩니다.")

    pending_account = st.session_state.pop("pending_account_tab", None)
    if pending_account in account_tabs:
        st.session_state.selected_account_tab = pending_account
    elif st.session_state.get("selected_account_tab") not in account_tabs:
        st.session_state.selected_account_tab = account_tabs[0]
    account_notice = st.session_state.pop("account_notice", None)
    if account_notice:
        st.success(account_notice)
    selected_account = st.selectbox(
        "계좌 선택",
        account_tabs,
        format_func=account_display_name,
        key="selected_account_tab",
    )
    if sheet_store is not None:
        if st.button(
            "Google Sheet 새로고침",
            help="Google Sheet를 직접 수정했거나 최신 내역이 바로 보이지 않을 때 사용하세요.",
            use_container_width=True,
        ):
            _clear_account_sheet_cache(include_tabs=True)
            st.rerun()
        st.caption("계좌 목록은 5분, 거래내역은 1분간 임시 보관됩니다. 저장·수정·삭제 후에는 즉시 갱신됩니다.")
    with st.expander("계좌 추가"):
        if sheet_store is None:
            st.caption("Google Sheet 연결 후 계좌 탭을 추가할 수 있습니다.")
        else:
            with st.form("new_account_form", clear_on_submit=True):
                new_account_name = st.text_input("계좌명", placeholder="예: 일반계좌, IRP, 연금저축")
                create_account = st.form_submit_button("계좌 만들기", use_container_width=True)
            if create_account:
                try:
                    created_tab = sheet_store.create_account_tab(new_account_name)
                    _clear_account_sheet_cache(include_tabs=True)
                    st.session_state.pending_account_tab = created_tab
                    st.session_state.account_notice = f"{created_tab} 탭을 만들었습니다."
                    st.rerun()
                except Exception as error:
                    st.error(str(error))

    if sheet_store is not None:
        try:
            drafts: list[dict[str, object]] = _cached_account_transactions(
                sheet_cache_key, selected_account, sheet_store,
            )
        except Exception as error:
            drafts = []
            st.error(f"선택한 계좌 내역을 읽지 못했습니다: {error}")
    else:
        drafts = st.session_state.trade_drafts_by_account.setdefault(selected_account, [])

    input_tab, account_tab, ledger_tab, rebalance_tab = st.tabs(
        ("✍️ 거래 입력", "💼 계좌 현황", "📋 거래내역", "🔔 리밸런싱 기록"),
    )
    with input_tab:
        st.caption(f"저장 위치: {selected_account}")
        record = _transaction_form(prefix=f"add_{selected_account}", submit_label="내역 저장")
        if record is not None:
            record = {
                "거래ID": secrets.token_hex(6),
                "입력시각": dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
                **record,
                "상태": "정상",
                "수정시각": "",
            }
            if sheet_store is not None:
                try:
                    sheet_store.append_transaction(selected_account, record)
                    _clear_account_sheet_cache()
                    st.success("선택한 계좌 탭에 저장했습니다.")
                    st.rerun()
                except Exception as error:
                    st.error(f"저장하지 못했습니다: {error}")
            else:
                drafts.append(record)
                st.success("임시 내역을 저장했습니다.")

    ledger = pd.DataFrame(drafts)
    if not ledger.empty:
        ledger["_정렬시각"] = pd.to_datetime(ledger["입력시각"], errors="coerce")
        ledger = ledger.sort_values(["_정렬시각", "거래일"], ascending=False, kind="stable")

    with account_tab:
        st.subheader(account_display_name(selected_account))
        active_frame = transaction_frame(drafts)
        if active_frame.empty:
            st.info("입금과 거래를 입력하면 계좌 잔고가 표시됩니다.")
        else:
            symbols = required_price_symbols(active_frame)
            price_key = f"account_prices::{selected_account}::{','.join(symbols)}"
            closes = pd.DataFrame()
            usdkrw = pd.Series(dtype=float)
            prices_ready = not symbols
            if symbols:
                if st.button("현재가 새로고침", use_container_width=True, key=f"prices_{selected_account}"):
                    try:
                        with st.spinner("현재가와 과거 종가를 불러오는 중입니다..."):
                            closes, usdkrw = _cached_account_prices(
                                symbols,
                                active_frame["거래일"].min().strftime("%Y-%m-%d"),
                                dt.datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
                            )
                        st.session_state[price_key] = (closes, usdkrw)
                        prices_ready = True
                    except Exception as error:
                        st.error(str(error))
                if price_key in st.session_state:
                    closes, usdkrw = st.session_state[price_key]
                    prices_ready = True
                else:
                    st.info("보유 종목 평가를 위해 ‘현재가 새로고침’을 눌러주세요.")
            if prices_ready:
                valuation = calculate_account_valuation(active_frame, closes, usdkrw)
                first, second, third, fourth = st.columns(4)
                first.metric("총 평가액", f"{valuation.total_value:,.0f}원")
                second.metric("순입금", f"{valuation.net_deposits:,.0f}원")
                third.metric("평가손익", f"{valuation.profit:,.0f}원")
                fourth.metric(
                    "투자금 대비 수익률",
                    "-" if valuation.return_rate is None else f"{valuation.return_rate:.2%}",
                )
                _account_value_chart(valuation.history)
                st.caption("실선은 총 평가액, 막대는 입출금입니다. 주식 평가는 Yahoo Finance 일반 종가를 사용합니다.")
                st.subheader("보유 종목")
                if valuation.holdings.empty:
                    st.info("평가 가능한 보유 종목이 없습니다.")
                else:
                    holdings = valuation.holdings.copy()
                    holdings["현재가"] = holdings["현재가"].map(lambda value: f"{value:,.2f}")
                    holdings["평가금액"] = holdings["평가금액"].map(lambda value: f"{value:,.0f}원")
                    st.dataframe(holdings, use_container_width=True, hide_index=True)
            st.subheader("최근 거래")
            recent = ledger.drop(columns=["_정렬시각"], errors="ignore").head(10)
            st.dataframe(recent, use_container_width=True, hide_index=True)

    with ledger_tab:
        if ledger.empty:
            st.info("입력된 거래내역이 없습니다.")
        else:
            edit_state_key = f"ledger_edit_id::{selected_account}"
            delete_state_key = f"ledger_delete_id::{selected_account}"
            active_edit_id = str(st.session_state.get(edit_state_key, ""))
            pending_delete_id = str(st.session_state.get(delete_state_key, ""))

            for _, row in ledger.iterrows():
                transaction_id = str(row["거래ID"])
                is_active = str(row.get("상태", "정상")) == "정상"
                with st.container(border=True):
                    delete_column, edit_column, summary_column = st.columns([1, 1, 5])
                    if delete_column.button(
                        "삭제",
                        key=f"delete_{selected_account}_{transaction_id}",
                        disabled=not is_active,
                        use_container_width=True,
                    ):
                        st.session_state[delete_state_key] = transaction_id
                        st.session_state.pop(edit_state_key, None)
                        st.rerun()
                    if edit_column.button(
                        "수정",
                        key=f"edit_{selected_account}_{transaction_id}",
                        disabled=not is_active,
                        use_container_width=True,
                    ):
                        st.session_state[edit_state_key] = transaction_id
                        st.session_state.pop(delete_state_key, None)
                        st.rerun()

                    symbol = str(row.get("종목코드", "")).strip() or "현금"
                    name = str(row.get("종목명", "")).strip()
                    summary_column.markdown(
                        f'**{row["입력시각"]} · {row["유형"]} · {symbol}**'
                    )
                    details = [f'거래일 {row["거래일"]}']
                    if name:
                        details.append(name)
                    if row["유형"] in ("매수", "매도"):
                        details.extend((f'수량 {row["수량"]}', f'체결단가 {row["체결단가"]}'))
                    raw_amount = str(row.get("금액", "") or "0").replace(",", "")
                    try:
                        amount = float(raw_amount)
                    except ValueError:
                        amount = 0.0
                    details.append(f"금액 {amount:,.0f}원")
                    if not is_active:
                        details.append("삭제됨")
                    summary_column.caption(" · ".join(details))

                    if pending_delete_id == transaction_id and is_active:
                        st.warning("이 내역을 삭제하면 계좌 계산에서 제외됩니다. 기록은 취소 상태로 보존됩니다.")
                        confirm_column, cancel_column = st.columns(2)
                        if confirm_column.button(
                            "삭제 확인",
                            type="primary",
                            key=f"confirm_delete_{selected_account}_{transaction_id}",
                            use_container_width=True,
                        ):
                            if sheet_store is not None:
                                try:
                                    deleted = sheet_store.cancel_transaction(selected_account, transaction_id)
                                except Exception as error:
                                    st.error(f"삭제하지 못했습니다: {error}")
                                else:
                                    if deleted:
                                        _clear_account_sheet_cache()
                                        st.session_state.pop(delete_state_key, None)
                                        st.rerun()
                                    st.error("삭제할 내역을 찾지 못했습니다.")
                            else:
                                source = next(
                                    item for item in drafts if str(item["거래ID"]) == transaction_id
                                )
                                source["상태"] = "취소"
                                source["수정시각"] = dt.datetime.now(
                                    ZoneInfo("Asia/Seoul")
                                ).strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.pop(delete_state_key, None)
                                st.rerun()
                        if cancel_column.button(
                            "돌아가기",
                            key=f"cancel_delete_{selected_account}_{transaction_id}",
                            use_container_width=True,
                        ):
                            st.session_state.pop(delete_state_key, None)
                            st.rerun()

                    if active_edit_id == transaction_id and is_active:
                        source = next(
                            item for item in drafts if str(item["거래ID"]) == transaction_id
                        )
                        st.subheader("선택 내역 수정")
                        changes = _transaction_form(
                            prefix=f"edit_{selected_account}_{transaction_id}",
                            submit_label="수정 내용 저장",
                            initial=source,
                        )
                        if changes is not None:
                            if sheet_store is not None:
                                try:
                                    updated = sheet_store.update_transaction(
                                        selected_account, transaction_id, changes,
                                    )
                                except Exception as error:
                                    st.error(f"수정하지 못했습니다: {error}")
                                else:
                                    if updated:
                                        _clear_account_sheet_cache()
                                        st.session_state.pop(edit_state_key, None)
                                        st.rerun()
                                    st.error("수정할 내역을 찾지 못했습니다.")
                            else:
                                source.update(changes)
                                source["수정시각"] = dt.datetime.now(
                                    ZoneInfo("Asia/Seoul")
                                ).strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.pop(edit_state_key, None)
                                st.rerun()
            st.download_button(
                "CSV 백업 다운로드",
                ledger.drop(columns=["_정렬시각"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{selected_account}_transactions.csv",
                mime="text/csv", use_container_width=True,
            )

    with rebalance_tab:
        st.subheader("프리셋 3 월간 운용")
        st.markdown(
            "- **프리셋 1:** 분산투자 방어형 50%\n"
            "- **프리셋 2:** 집중투자 공격형 50%\n"
            "- **운용 시작:** 2026-11-02 시가\n"
            "- **선택 계좌:** " + account_display_name(selected_account)
        )
        st.info("다음 단계에서 월말 목표 비중과 실제 계좌 비중을 비교해 표시합니다.")


def _trade_entry_page() -> None:
    """Read-only account dashboard backed by account-prefixed Sheet tabs."""
    if st.button("← 기능 선택으로", key="account_home"):
        _go_to("home")
    st.title("📊 투자 계좌 현황")
    st.write("Google Sheet에 기록한 투자 이력과 현재 운용 성과를 조회합니다.")

    sheet_store, sheet_error = _google_sheet_store()
    if sheet_store is None:
        if sheet_error:
            st.error(sheet_error)
        else:
            st.info("Google Sheet 연결 정보가 필요합니다.")
        return

    try:
        sheet_cache_key = _account_sheet_cache_key(sheet_store)
        account_tabs = _cached_account_tabs(sheet_cache_key, sheet_store)
    except Exception as error:
        st.error(f"계좌 목록을 불러오지 못했습니다: {error}")
        return
    if not account_tabs:
        st.info("이름이 ‘계좌_’로 시작하는 Google Sheet 탭을 먼저 만들어주세요.")
        return

    if st.session_state.get("dashboard_account_tab") not in account_tabs:
        st.session_state.dashboard_account_tab = account_tabs[0]
    selected_account = st.selectbox(
        "계좌 선택",
        account_tabs,
        format_func=account_display_name,
        key="dashboard_account_tab",
    )
    refresh_sheet, refresh_prices = st.columns(2)
    if refresh_sheet.button(
        "Google Sheet 새로고침",
        help="Sheet에서 거래를 입력하거나 수정한 뒤 누르면 즉시 다시 읽습니다.",
        use_container_width=True,
    ):
        _clear_account_sheet_cache(include_tabs=True)
        st.rerun()
    if refresh_prices.button(
        "시세 새로고침",
        help="Yahoo Finance 종가와 환율 캐시를 비우고 다시 불러옵니다.",
        use_container_width=True,
    ):
        _cached_account_prices.clear()
        st.rerun()
    st.caption(
        f"원본 입력·수정 위치: Google Sheet의 `{selected_account}` 탭 · "
        "계좌 목록 5분, 거래내역 1분, 시세 15분 캐시"
    )

    try:
        records: list[dict[str, object]] = _cached_account_transactions(
            sheet_cache_key, selected_account, sheet_store,
        )
    except Exception as error:
        st.error(f"선택한 계좌 내역을 불러오지 못했습니다: {error}")
        return

    active_frame = transaction_frame(records)
    if active_frame.empty:
        st.info("정상 상태의 거래가 없습니다. Google Sheet에 입금과 거래내역을 입력해 주세요.")
        return

    first_date = active_frame["거래일"].min().date()
    today = dt.datetime.now(ZoneInfo("Asia/Seoul")).date()
    from_key = f"account_from::{selected_account}"
    to_key = f"account_to::{selected_account}"
    if from_key not in st.session_state or not first_date <= st.session_state[from_key] <= today:
        st.session_state[from_key] = first_date
    if to_key not in st.session_state or not first_date <= st.session_state[to_key] <= today:
        st.session_state[to_key] = today
    date_left, date_right = st.columns(2)
    from_date = date_left.date_input(
        "조회 시작일",
        min_value=first_date,
        max_value=today,
        key=from_key,
    )
    to_date = date_right.date_input(
        "조회 종료일",
        min_value=first_date,
        max_value=today,
        key=to_key,
    )
    if from_date > to_date:
        st.error("조회 시작일은 종료일보다 앞서야 합니다.")
        return

    symbols = required_price_symbols(active_frame)
    closes = pd.DataFrame()
    usdkrw = pd.Series(dtype=float)
    if symbols:
        try:
            with st.spinner("계좌 평가에 필요한 종가와 환율을 불러오는 중입니다… (최대 약 10초)"):
                closes, usdkrw = _cached_account_prices(
                    symbols, first_date.isoformat(), today.isoformat(),
                )
        except Exception as error:
            st.error(f"시세를 불러오지 못했습니다: {error}")
            return

    start_ts = pd.Timestamp(from_date)
    end_ts = pd.Timestamp(to_date)
    frame_to_end = active_frame[active_frame["거래일"] <= end_ts].copy()
    if frame_to_end.empty:
        st.info("선택한 종료일까지의 거래가 없습니다.")
        return
    valuation = calculate_account_valuation(
        frame_to_end, closes, usdkrw, end_date=end_ts,
    )
    performance = account_period_metrics(valuation.history, start_ts, end_ts)
    period_frame = frame_to_end[
        (frame_to_end["거래일"] >= start_ts) & (frame_to_end["거래일"] <= end_ts)
    ]
    period_deposits = float(
        period_frame.loc[period_frame["유형"] == "입금", "원화금액"].sum()
    )
    period_withdrawals = float(
        period_frame.loc[period_frame["유형"] == "출금", "원화금액"].sum()
    )
    cumulative_deposits = float(
        frame_to_end.loc[frame_to_end["유형"] == "입금", "원화금액"].sum()
    )
    cumulative_withdrawals = float(
        frame_to_end.loc[frame_to_end["유형"] == "출금", "원화금액"].sum()
    )
    performance_valid = valuation.cash >= -1e-6

    def display_percent(value: float | None) -> str:
        return "-" if value is None or not np.isfinite(value) else f"{value:.2%}"

    dashboard_tab, holdings_tab, history_tab, rebalance_tab = st.tabs(
        ("📊 투자 대시보드", "💼 보유종목", "📋 투자이력", "🔔 리밸런싱 기록"),
    )
    with dashboard_tab:
        st.subheader(f"{account_display_name(selected_account)} · {to_date:%Y-%m-%d} 기준")
        if not performance_valid:
            st.warning(
                "현금 잔액이 음수입니다. Google Sheet에서 매수 전에 필요한 입금 내역이 "
                "모두 기록됐는지 확인해 주세요. 원장이 완성될 때까지 수익률과 낙폭은 표시하지 않습니다."
            )
        _account_metric_cards((
            ("총 평가액", f"{valuation.total_value:,.0f}원"),
            ("현금", f"{valuation.cash:,.0f}원"),
            ("보유자산 평가액", f"{valuation.holdings_value:,.0f}원"),
            ("누적 입금", f"{cumulative_deposits:,.0f}원"),
            ("누적 출금", f"{cumulative_withdrawals:,.0f}원"),
            ("누적 평가손익", f"{valuation.profit:,.0f}원"),
            ("누적 단순수익률", display_percent(valuation.return_rate if performance_valid else None)),
            ("기간 입금", f"{period_deposits:,.0f}원"),
            ("기간 출금", f"{period_withdrawals:,.0f}원"),
            ("기간 수익률(TWR)", display_percent(performance.period_return if performance_valid else None)),
            ("연환산 수익률", display_percent(performance.annualized_return if performance_valid else None)),
            ("최대 낙폭(MDD)", display_percent(performance.max_drawdown if performance_valid else None)),
            ("현재 낙폭", display_percent(performance.current_drawdown if performance_valid else None)),
        ))
        st.caption(
            "기간수익률과 낙폭은 입출금 영향을 제거한 시간가중수익률(TWR) 기준입니다. "
            "누적 단순수익률은 총평가액과 누적 순입금의 차이를 기준으로 합니다."
        )
        value_chart, return_chart = st.tabs(("평가액·입출금", "수익률·낙폭"))
        with value_chart:
            _account_value_chart(performance.history)
        with return_chart:
            if performance_valid:
                _account_performance_chart(performance.history)
            else:
                st.info("입금 내역을 보완하면 수익률·낙폭 차트가 표시됩니다.")

    with holdings_tab:
        st.subheader(f"{to_date:%Y-%m-%d} 기준 보유종목")
        if valuation.holdings.empty:
            st.info("선택한 종료일에 보유 중인 종목이 없습니다.")
        else:
            holdings = valuation.holdings.copy()
            holdings["현재가"] = holdings["현재가"].map(lambda value: f"{value:,.2f}")
            holdings["환율"] = holdings["환율"].map(lambda value: f"{value:,.2f}")
            holdings["평가금액"] = holdings["평가금액"].map(lambda value: f"{value:,.0f}원")
            st.dataframe(holdings, use_container_width=True, hide_index=True)

    with history_tab:
        st.subheader(f"투자이력 · {from_date:%Y-%m-%d} ~ {to_date:%Y-%m-%d}")
        ledger = pd.DataFrame(records)
        if ledger.empty:
            st.info("거래내역이 없습니다.")
        else:
            ledger["_거래일"] = pd.to_datetime(ledger["거래일"], errors="coerce")
            ledger["_입력시각"] = pd.to_datetime(ledger["입력시각"], errors="coerce")
            ledger = ledger[
                (ledger["_거래일"] >= start_ts) & (ledger["_거래일"] <= end_ts)
            ].sort_values(["_거래일", "_입력시각"], ascending=False, kind="stable")
            display_columns = [
                "입력시각", "거래일", "유형", "시장", "종목코드", "종목명", "수량",
                "체결단가", "수수료·세금", "금액", "통화", "환율", "메모", "상태",
            ]
            display_columns = [column for column in display_columns if column in ledger]
            st.dataframe(ledger[display_columns], use_container_width=True, hide_index=True)
            st.download_button(
                "조회기간 CSV 다운로드",
                ledger[display_columns].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{selected_account}_{from_date}_{to_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.info("거래내역 추가·수정·삭제는 Google Sheet에서 진행한 뒤 상단의 새로고침 버튼을 눌러주세요.")

    with rebalance_tab:
        st.subheader("프리셋 3 월간 운용")
        st.markdown(
            "- **프리셋 1:** 분산투자 방어형 50%\n"
            "- **프리셋 2:** 집중투자 공격형 50%\n"
            "- **운용 시작:** 2026-11-02 시가\n"
            "- **선택 계좌:** " + account_display_name(selected_account)
        )
        st.info("다음 단계에서 월말 목표 비중과 실제 계좌 비중을 비교해 표시합니다.")


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


def _domestic_mapping_table(*presets: PortfolioPreset) -> pd.DataFrame:
    roles: dict[str, set[str]] = {}
    for position, preset in enumerate(presets, start=1):
        for ticker in preset.attack_tickers:
            roles.setdefault(ticker, set()).add(f"프리셋 {position} 투자")
        for ticker in preset.defense_tickers:
            roles.setdefault(ticker, set()).add(f"프리셋 {position} 방어")
    rows = []
    for ticker, role in roles.items():
        domestic = DOMESTIC_EXECUTION_MAP[ticker]
        rows.append({
            "구분": " · ".join(sorted(role)),
            "미국 신호": ticker,
            "국내 실행 ETF": domestic_label(domestic),
        })
    return pd.DataFrame(rows)


def _user_presets() -> dict[str, UserPreset]:
    if "user_presets" not in st.session_state:
        st.session_state.user_presets = {}
    return st.session_state.user_presets


def _user_preset_key(title: str) -> str:
    return f"{USER_PRESET_PREFIX}{title}"


def _user_preset_manager() -> None:
    library = _user_presets()
    with st.expander("내 프리셋 관리", expanded=bool(library)):
        uploaded = st.file_uploader("프리셋 JSON 불러오기", type=("json",), key="user_preset_upload")
        if uploaded is not None:
            digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
            if digest != st.session_state.get("imported_user_preset_digest"):
                try:
                    imported = import_user_presets(uploaded.getvalue())
                    for record in imported:
                        library[record.title] = record
                    st.session_state.imported_user_preset_digest = digest
                    st.session_state.user_preset_notice = f"내 프리셋 {len(imported)}개를 불러왔습니다."
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        if library:
            titles = sorted(library)
            selected = st.selectbox("저장된 프리셋", titles, key="managed_user_preset")
            left, right = st.columns(2)
            if left.button("선택 프리셋 불러오기", use_container_width=True):
                st.session_state.preset_name = _user_preset_key(selected)
                st.session_state.form_version = st.session_state.get("form_version", 0) + 1
                st.rerun()
            if right.button("선택 프리셋 삭제", use_container_width=True):
                del library[selected]
                if st.session_state.get("preset_name") == _user_preset_key(selected):
                    st.session_state.preset_name = PRESET_NAMES[0]
                st.session_state.user_preset_notice = f"‘{selected}’ 프리셋을 삭제했습니다."
                st.rerun()
            st.download_button(
                "내 프리셋 전체 JSON 다운로드",
                export_user_presets(library.values()),
                file_name="snowball_user_presets.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("새로고침·재배포·다른 기기에서 복구하려면 이 JSON을 보관한 뒤 다시 불러오세요.")
        else:
            st.caption("아직 저장된 내 프리셋이 없습니다. 아래에서 조건을 만든 뒤 제목과 함께 저장하세요.")


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
    left, right = st.columns(2)
    if left.button(PRESET_9_NAME, use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == PRESET_9_NAME else "secondary"):
        st.session_state.preset_name = PRESET_9_NAME
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    if right.button(OPTIMIZED_COMBINED_PRESET_NAME, use_container_width=True, type="primary" if st.session_state.get("preset_name", PRESET_NAMES[0]) == OPTIMIZED_COMBINED_PRESET_NAME else "secondary"):
        st.session_state.preset_name = OPTIMIZED_COMBINED_PRESET_NAME
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        st.rerun()
    selected = st.session_state.get("preset_name", PRESET_NAMES[0])
    st.caption("버튼을 누르면 아래 모든 전략 조건이 해당 연구 프리셋으로 초기화됩니다. 이후 필요한 항목만 바꿀 수 있습니다.")
    _user_preset_manager()
    notice = st.session_state.pop("user_preset_notice", None)
    if notice:
        st.success(notice)
    return selected


def _save_user_preset_controls(edited: PortfolioPreset, domestic_execution: bool) -> None:
    with st.container(border=True):
        st.subheader("💾 내 프리셋 저장")
        selected = st.session_state.get("preset_name", "")
        current_title = selected.removeprefix(USER_PRESET_PREFIX) if selected.startswith(USER_PRESET_PREFIX) else ""
        title = st.text_input(
            "프리셋 제목", value=current_title,
            placeholder="예: 장기 투자 방어 전략", key=f"user_preset_title_{selected}",
        ).strip()
        st.caption("현재 화면의 모든 전략 조건과 국내 실행 여부를 함께 저장합니다. 같은 제목은 최신 설정으로 덮어씁니다.")
        if st.button("현재 조건을 프리셋으로 저장", use_container_width=True):
            if not title:
                st.error("프리셋 제목을 입력해 주세요.")
                return
            try:
                strategy = replace(edited, name=title)
                record = UserPreset(title, strategy, domestic_execution)
                record.validate()
            except ValueError as error:
                st.error(str(error))
                return
            _user_presets()[title] = record
            st.session_state.preset_name = _user_preset_key(title)
            st.session_state.form_version = st.session_state.get("form_version", 0) + 1
            st.session_state.user_preset_notice = f"‘{title}’을(를) 내 프리셋에 저장했습니다."
            st.rerun()


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


def _strategy_controls(base: PortfolioPreset, *, domestic_execution: bool = False) -> PortfolioPreset:
    version = st.session_state.get("form_version", 0)
    prefix = f"strategy_{base.name}_{version}"
    signal_options = _asset_options(base.symbols)
    asset_options = sorted(DOMESTIC_EXECUTION_MAP) if domestic_execution else signal_options

    def asset_label(ticker: str) -> str:
        if domestic_execution:
            mapped = DOMESTIC_EXECUTION_MAP[ticker]
            return f"{domestic_label(mapped)} · 신호 {ticker}"
        return display_name(ticker, "미국 ETF")

    with st.container(border=True):
        st.subheader("🎯 투자자산 선택")
        attack_tickers = tuple(st.multiselect(
            "투자자산 후보군" + (" · 국내 실행 ETF" if domestic_execution else ""),
            asset_options, default=list(base.attack_tickers),
            format_func=asset_label, key=f"{prefix}_attack_assets",
        ))
        if domestic_execution:
            st.caption("화면에는 실제 매수할 국내 ETF를 표시하고, 모멘텀 판단은 오른쪽의 대응 미국 ETF로 계산합니다.")
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
                "카나리아 유니버스", signal_options, default=list(base.canary_tickers),
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
            "방어자산 후보군" + (" · 국내 실행 ETF" if domestic_execution else ""),
            asset_options, default=list(base.defense_tickers),
            format_func=asset_label, key=f"{prefix}_defense_assets",
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


@st.cache_data(ttl=6 * 60 * 60, max_entries=12, show_spinner=False)
def _prices(symbols: tuple[str, ...], start_iso: str, end_iso: str) -> PriceBundle:
    return download_yahoo_prices(symbols, pd.Timestamp(start_iso), pd.Timestamp(end_iso))


@st.cache_data(ttl=24 * 60 * 60, max_entries=24, show_spinner=False)
def _market_data(signals: tuple[str, ...], start_iso: str, end_iso: str, mode: str) -> dict[str, pd.Series]:
    return load_strategy_market_data(signals, start_iso, end_iso, mode=mode)


@st.cache_data(ttl=24 * 60 * 60, max_entries=2, show_spinner=False)
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
    return market_data_export(result)


def _result_cache_key(result: dict) -> str:
    """Stable key for expensive display-only exports of one completed run."""
    nav = result["nav"]
    nav_hash = hashlib.sha256(
        pd.util.hash_pandas_object(nav, index=True).values.tobytes()
    ).hexdigest()[:16]
    snapshot = result.get("data_snapshot", {}).get("price_snapshot_id", "-")
    return "|".join((
        str(result.get("preset_name", "-")), str(result["common_start"]),
        str(result["common_end"]), str(snapshot), nav_hash,
    ))


@st.cache_data(show_spinner=False)
def _cached_diagnostic_zip(cache_key: str, _result: dict) -> bytes:
    return diagnostic_zip(_result)


@st.cache_data(show_spinner=False)
def _cached_diagnostic_frames(cache_key: str, _result: dict) -> dict[str, pd.DataFrame]:
    return diagnostic_frames(_result)


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
    cache_key = _result_cache_key(result)
    st.download_button(
        "비교 진단자료 한 번에 다운로드 (ZIP)",
        _cached_diagnostic_zip(cache_key, result),
        file_name=f"snowball_diagnostics_{result['common_start'].date()}_{result['common_end'].date()}.zip",
        mime="application/zip",
        use_container_width=True,
        help="월별 수익률, 리밸런싱, 자산·카나리아·시장 신호, 일별 NAV와 낙폭이 포함됩니다.",
    )
    sections = ("수익률 차트", "손실폭", "월별 수익률", "리밸런싱", "판단 원시값", "데이터 스냅샷", "적용 설정")
    section = st.radio(
        "결과 보기", sections, horizontal=True, label_visibility="collapsed",
        key=f"result_section_{cache_key}",
    )
    if section == "수익률 차트":
        nav = result["nav"]
        chart = ((nav / nav.iloc[0]) - 1.0).rename("포트폴리오").to_frame()
        benchmark = result.get("benchmark_ticker", "QQQ")
        if benchmark in bundle.adjusted_close:
            qqq = bundle.adjusted_close[benchmark].reindex(nav.index).ffill().dropna()
            if not qqq.empty:
                label = "국내 나스닥100" if benchmark.endswith(".KS") else "QQQ"
                chart.loc[qqq.index, label] = qqq / qqq.iloc[0] - 1.0
        st.line_chart(chart, y_label="누적 수익률")
    elif section == "손실폭":
        st.line_chart(result["drawdown"].to_frame(), y_label="낙폭")
    elif section == "월별 수익률":
        exports = _cached_diagnostic_frames(cache_key, result)
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
    elif section == "리밸런싱":
        exports = _cached_diagnostic_frames(cache_key, result)
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
    elif section == "판단 원시값":
        exports = _cached_diagnostic_frames(cache_key, result)
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
    elif section == "데이터 스냅샷":
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
    elif section == "적용 설정":
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
                else DOMESTIC_PRESET_2 if saved.preset.name == PRESET_NAMES[1]
                else saved.preset.name
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
        if saved.secondary_preset.name == PRESET_7_NAME:
            combined_name, second_label = ALTERNATIVE_COMBINED_PRESET_NAME, "프리셋 7"
        elif saved.secondary_preset.name == PRESET_9_NAME:
            combined_name, second_label = OPTIMIZED_COMBINED_PRESET_NAME, "프리셋 9"
        else:
            combined_name, second_label = COMBINED_PRESET_NAME, "프리셋 2"
        result = run_blended_presets(
            saved.preset, saved.secondary_preset, saved.preset1_weight,
            bundle, market, saved.start_date, saved.end_date,
            initial_investment=saved.initial_investment,
            transaction_cost_rate=saved.transaction_cost_rate,
            preset_name=combined_name,
            second_label=second_label,
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
    if st.query_params.get("run_id"):
        st.session_state.app_page = "backtest"
    page = st.session_state.get("app_page", "home")
    if page == "home":
        _home_page()
        return
    if page == "trades":
        _trade_entry_page()
        return
    if st.button("← 기능 선택으로", key="backtest_home"):
        _go_to("home")
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
            elif saved.secondary_preset.name == PRESET_9_NAME:
                st.session_state.preset_name = OPTIMIZED_COMBINED_PRESET_NAME
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
    user_record = None
    if preset_name.startswith(USER_PRESET_PREFIX):
        user_record = _user_presets().get(preset_name.removeprefix(USER_PRESET_PREFIX))
        if user_record is None:
            st.warning("저장된 사용자 프리셋을 찾지 못해 프리셋 1로 돌아갑니다.")
            preset_name = PRESET_NAMES[0]
            st.session_state.preset_name = preset_name
    domestic_execution = (
        user_record.domestic_execution if user_record is not None
        else preset_name in DOMESTIC_PRESET_NAMES
    )
    combined_weight = 50
    if preset_name in COMBINED_PRESET_NAMES:
        with st.container(border=True):
            st.subheader("⚖️ 결합 비율")
            combined_weight = int(st.number_input(
                "프리셋 1 비중(%)", min_value=0, max_value=100,
                value=50, step=5,
                key="combined_preset1_weight",
            ))
            second_number = (
                7 if preset_name == ALTERNATIVE_COMBINED_PRESET_NAME
                else 9 if preset_name == OPTIMIZED_COMBINED_PRESET_NAME
                else 2
            )
            st.caption(
                f"프리셋 1 {combined_weight}% + 프리셋 {second_number} {100 - combined_weight}%로 "
                "각 전략의 리밸런싱 종목과 일별 성과를 결합합니다."
            )
            if domestic_execution:
                st.markdown("**프리셋 6 국내 실행 종목**")
                st.dataframe(
                    _domestic_mapping_table(
                        PRESETS[PRESET_NAMES[0]], PRESETS[PRESET_NAMES[1]],
                    ),
                    use_container_width=True, hide_index=True,
                )
                st.caption("신호는 프리셋 1·2의 미국 ETF로 각각 계산하고, 표의 국내 ETF를 다음 한국 거래일에 매수합니다.")
        edited = PRESETS[PRESET_NAMES[0]]
    else:
        if user_record is not None:
            base = user_record.strategy
        elif preset_name == DOMESTIC_PRESET_1:
            base = PRESETS[PRESET_NAMES[0]]
        elif preset_name == DOMESTIC_PRESET_2:
            base = PRESETS[PRESET_NAMES[1]]
        else:
            base = PRESETS[preset_name]
        with st.expander("전략 조건 편집", expanded=True):
            edited = _strategy_controls(base, domestic_execution=domestic_execution)
        _save_user_preset_controls(edited, domestic_execution)

    with st.container(border=True):
        st.subheader("⚙️ 기본 설정")
        left, right = st.columns(2)
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

    secondary_preset = (
        PRESETS[PRESET_7_NAME]
        if preset_name == ALTERNATIVE_COMBINED_PRESET_NAME
        else PRESETS[PRESET_9_NAME]
        if preset_name == OPTIMIZED_COMBINED_PRESET_NAME
        else PRESETS[PRESET_NAMES[1]] if preset_name in COMBINED_PRESET_NAMES else None
    )
    current_saved = SavedRun(
        preset=edited,
        start_date=start_date,
        end_date=end_date,
        initial_investment=initial,
        transaction_cost_rate=cost_bps / 10_000.0,
        market_data_mode=MARKET_DATA_MODES[mode_label],
        secondary_preset=secondary_preset,
        preset1_weight=combined_weight / 100.0 if preset_name in COMBINED_PRESET_NAMES else 1.0,
        domestic_execution=domestic_execution,
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
        try:
            with st.spinner("가격과 시장 데이터를 불러와 계산하고 있습니다…"):
                result, bundle = _execute(current_saved)
            st.session_state.last_result = result
            st.session_state.last_bundle = bundle
            st.session_state.last_saved_run = current_saved
            st.query_params["run_id"] = _remember_run(current_saved, result, bundle)
        except Exception as error:  # Streamlit must turn provider failures into actionable UI feedback.
            st.error(f"백테스트를 완료하지 못했습니다: {error}")

    if "last_result" in st.session_state:
        result_matches = st.session_state.get("last_saved_run") == current_saved
        if result_matches:
            st.caption("마지막 결과는 새로고침하거나 이 탭을 다시 열어도 자동으로 복원됩니다.")
            _show_results(st.session_state.last_result, st.session_state.last_bundle)
        else:
            st.info("전략 조건이 변경되어 이전 백테스트 결과를 접었습니다. 새 조건으로 실행하거나 필요하면 이전 결과를 펼쳐 보세요.")
            stale_key = hashlib.sha256(repr(current_saved).encode("utf-8")).hexdigest()[:12]
            if st.toggle("이전 결과 보기", value=False, key=f"show_stale_result_{stale_key}"):
                _show_results(st.session_state.last_result, st.session_state.last_bundle)


if __name__ == "__main__":
    main()
