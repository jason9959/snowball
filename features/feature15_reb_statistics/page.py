from __future__ import annotations

import datetime as dt
import html
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common.metric_ui import metric_dataframe
from .api import fetch_statistics, resolve_api_key
from .analysis import analyze_statistics, build_region_comparison


PRESETS = {
    "전국 아파트 매매가격지수": {
        "stat_table_id": "A_2024_00045",
        "cycle": "MM",
        "region_id": "500001",
        "item_id": "100001",
        "value_kind": "index",
        "description": "전국 아파트 매매가격지수의 월별 흐름입니다.",
    },
    "전국 월별 지가변동률": {
        "stat_table_id": "A_2024_00903",
        "cycle": "MM",
        "region_id": "500001",
        "item_id": "100001",
        "value_kind": "rate",
        "description": "전국 토지가격의 월별 변동률입니다.",
    },
}

REGIONS = {
    "전국": "500001",
    "수도권": "500002",
    "지방권": "500003",
    "5대광역시": "500004",
    "4대광역시": "500005",
}


def _reset() -> None:
    for key in list(st.session_state):
        if key.startswith("reb_"):
            st.session_state.pop(key, None)


def _excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    export = frame.copy()
    export["날짜"] = export["날짜"].dt.strftime("%Y-%m-%d")
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="R-ONE 통계", index=False)
    return buffer.getvalue()


def _summary_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        "<div class='reb-summary-card'>"
        f"<div class='reb-summary-label'>{html.escape(label)}</div>"
        f"<div class='reb-summary-value'>{html.escape(value)}</div>"
        f"<div class='reb-summary-help'>{html.escape(help_text)}</div>"
        "</div>"
        for label, value, help_text in items
    )
    st.html(
        """
        <style>
        .reb-summary-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:.4rem 0 1.5rem}
        .reb-summary-card{border:1px solid #e5e8eb;border-radius:14px;padding:15px 16px;background:#fff;min-width:0}
        .reb-summary-label{font-size:.82rem;color:#6b7684;margin-bottom:7px}
        .reb-summary-value{font-size:1.18rem;line-height:1.35;font-weight:700;color:#191f28;overflow-wrap:anywhere}
        .reb-summary-help{font-size:.72rem;line-height:1.4;color:#8b95a1;margin-top:6px}
        @media(max-width:760px){.reb-summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.reb-summary-value{font-size:1.02rem}}
        </style>
        """ + f"<div class='reb-summary-grid'>{cards}</div>",
    )


def _conditions() -> None:
    st.title("🏘️ 한국부동산원 부동산 통계")
    st.markdown(
        "<p class='step-caption'>R-ONE 공식 통계를 조회해 지표 추이와 기간별 변화를 분석합니다.</p>",
        unsafe_allow_html=True,
    )
    if st.button("↺ 조건 초기화", key="reb_reset"):
        _reset(); st.rerun()

    st.subheader("통계 선택")
    preset_name = st.selectbox(
        "분석할 통계",
        [*PRESETS, "직접 코드 입력"],
        key="reb_preset",
    )
    custom = preset_name == "직접 코드 입력"
    preset = PRESETS.get(preset_name, {})

    if custom:
        st.caption("R-ONE 통계코드 검색에서 확인한 코드를 입력하세요.")
        first, second = st.columns(2)
        with first:
            stat_table_id = st.text_input("통계표 코드", value="A_2024_00045", key="reb_stat_table_id").strip()
        with second:
            item_id = st.text_input("항목 코드", value="100001", key="reb_item_id").strip()
            cycle_label = st.selectbox("수록 주기", ["월", "분기", "연", "주"], key="reb_cycle_label")
        cycle = {"월": "MM", "분기": "QY", "연": "YY", "주": "WK"}[cycle_label]
        value_kind = st.selectbox(
            "값의 종류",
            ["지수", "기간 변동률"],
            key="reb_value_kind_label",
        )
        value_kind = "index" if value_kind == "지수" else "rate"
    else:
        stat_table_id = preset["stat_table_id"]
        item_id = preset["item_id"]
        cycle = preset["cycle"]
        value_kind = preset["value_kind"]
        st.info(preset["description"])

    st.divider(); st.subheader("비교 지역")
    if custom:
        region_codes_text = st.text_input(
            "지역·분류 코드",
            value="500001, 500002, 500003",
            help="여러 지역은 쉼표로 구분하세요. 결과의 실제 지역명은 R-ONE 응답을 사용합니다.",
            key="reb_region_codes",
        )
        region_codes = list(dict.fromkeys(code.strip() for code in region_codes_text.split(",") if code.strip()))
    else:
        selected_regions = st.multiselect(
            "지역 선택",
            list(REGIONS),
            default=["전국", "수도권", "지방권"],
            key="reb_regions",
        )
        region_codes = [REGIONS[name] for name in selected_regions]

    st.divider(); st.subheader("조회 기간")
    current_year = dt.date.today().year
    start_col, end_col = st.columns(2)
    with start_col:
        start_year = st.number_input("시작 연도", min_value=1900, max_value=current_year, value=max(2006, current_year - 10), step=1, key="reb_start_year")
    with end_col:
        end_year = st.number_input("종료 연도", min_value=1900, max_value=current_year, value=current_year, step=1, key="reb_end_year")

    api_key = resolve_api_key(st.secrets)
    if api_key:
        st.success("R-ONE 인증키가 연결되어 전체 기간을 조회합니다.")
    else:
        st.warning("RONE_API_KEY가 없어 공식 샘플 모드로 조회합니다. 샘플 모드는 첫 5건만 제공합니다.")

    back_col, run_col = st.columns(2)
    if back_col.button("뒤로", key="reb_back", use_container_width=True):
        st.session_state["current_page"] = "feature"; st.rerun()
    run = run_col.button("결과 보기", type="primary", key="reb_run", use_container_width=True)
    if not run:
        return
    if int(start_year) > int(end_year):
        st.error("시작 연도는 종료 연도보다 늦을 수 없습니다."); return
    if not all((stat_table_id, item_id)) or not region_codes:
        st.error("통계표·항목 코드와 한 개 이상의 지역을 입력해주세요."); return

    try:
        with st.spinner("한국부동산원 R-ONE 자료를 불러오고 있습니다..."):
            frames = []
            metadata_items = []
            for region_code in region_codes:
                region_frame, region_metadata = fetch_statistics(
                    stat_table_id=stat_table_id,
                    cycle=cycle,
                    region_id=region_code,
                    item_id=item_id,
                    start_year=int(start_year),
                    end_year=int(end_year),
                    api_key=api_key,
                )
                region_frame["지역명"] = region_frame.get("CLS_FULLNM", region_frame.get("CLS_NM")).fillna(region_code)
                frames.append(region_frame)
                metadata_items.append(region_metadata)
            raw = pd.concat(frames, ignore_index=True)
            metadata = {
                "sample_mode": any(item["sample_mode"] for item in metadata_items),
                "returned_count": sum(item["returned_count"] for item in metadata_items),
                "total_count": sum(item["total_count"] for item in metadata_items),
            }
        st.session_state["reb_result"] = {
            "data": analyze_statistics(raw, value_kind),
            "metadata": metadata,
            "preset_name": preset_name,
            "value_kind": value_kind,
            "codes": (stat_table_id, tuple(region_codes), item_id, cycle),
        }
        st.session_state["current_page"] = "reb_statistics_results"
        st.rerun()
    except Exception as exc:
        st.error(f"R-ONE 통계를 불러오지 못했습니다: {exc}")


def _results() -> None:
    result = st.session_state.get("reb_result")
    if not result:
        st.session_state["current_page"] = "reb_statistics_conditions"; st.rerun()
    data: pd.DataFrame = result["data"]
    metadata = result["metadata"]
    unit = str(data.get("UI_NM", pd.Series([""])).iloc[0] or "")
    regions = list(dict.fromkeys(data["지역명"].astype(str)))
    region = ", ".join(regions)
    item = str(data.get("ITM_NM", pd.Series(["통계값"])).iloc[0])

    st.title("🏘️ 한국부동산원 통계 분석 결과")
    st.caption(f"{result['preset_name']} · {region} · {item}")
    if metadata["sample_mode"]:
        st.warning(f"인증키가 없는 샘플 결과입니다. 전체 {metadata['total_count']:,}건 중 {metadata['returned_count']}건만 표시합니다.")

    normalized, ranking = build_region_comparison(data, result["value_kind"])
    best = ranking.iloc[0] if not ranking.empty else None
    average_annualized = ranking["연환산 상승률 (%)"].mean() if not ranking.empty else np.nan
    change = data["기간 변동률 (%)"].dropna()
    _summary_cards([
        ("비교 지역", f"{len(regions)}개", "같은 기간으로 비교한 지역 수"),
        ("누적 상승률 1위", f"{best['지역']} {best['누적 상승률 (%)']:+.2f}%" if best is not None else "-", "공통 시작일을 100으로 환산한 결과"),
        ("평균 연환산 상승률", f"{average_annualized:+.2f}%" if np.isfinite(average_annualized) else "-", "비교 지역 연환산 상승률의 평균"),
        ("상승 기간 비율", f"{change.gt(0).mean() * 100:.1f}%" if len(change) else "-", "전체 지역 관측치 중 상승한 비율"),
    ])

    st.subheader("통계값 추이")
    line = go.Figure()
    for region_name, group in data.groupby("지역명", sort=False):
        line.add_trace(go.Scatter(x=group["날짜"], y=group["값"], mode="lines", name=str(region_name), line={"width": 2.3}))
    line.update_layout(xaxis_title="시점", yaxis_title=f"{item} ({unit})" if unit else item, hovermode="x unified", margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=420)
    st.plotly_chart(line, use_container_width=True)

    st.subheader("기준값 100 지역 비교")
    comparison = go.Figure()
    for region_name in normalized.columns:
        comparison.add_trace(go.Scatter(x=normalized.index, y=normalized[region_name], mode="lines", name=str(region_name), line={"width": 2.5}))
    comparison.add_hline(y=100, line_color="#b0b8c1", line_dash="dot")
    comparison.update_layout(xaxis_title="시점", yaxis_title="비교지수 (시작=100)", hovermode="x unified", margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=420)
    st.plotly_chart(comparison, use_container_width=True)

    st.subheader("누적 상승률")
    cumulative = go.Figure()
    for region_name in normalized.columns:
        cumulative.add_trace(go.Scatter(x=normalized.index, y=(normalized[region_name] - 100), mode="lines", name=str(region_name), line={"width": 2.3}))
    cumulative.add_hline(y=0, line_color="#b0b8c1", line_dash="dot")
    cumulative.update_layout(xaxis_title="시점", yaxis_title="누적 상승률 (%)", hovermode="x unified", margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=400)
    st.plotly_chart(cumulative, use_container_width=True)

    st.subheader("기간별 변동률")
    bar = go.Figure()
    for region_name, group in data.groupby("지역명", sort=False):
        bar.add_trace(go.Bar(x=group["날짜"], y=group["기간 변동률 (%)"], name=str(region_name)))
    bar.add_hline(y=0, line_color="#8B95A1", line_width=1)
    bar.update_layout(xaxis_title="시점", yaxis_title="변동률 (%)", barmode="group", margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=380)
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("지역별 상승률 순위")
    ranking_display = ranking.reset_index().copy()
    if not ranking_display.empty:
        ranking_display["누적 상승률 (%)"] = ranking_display["누적 상승률 (%)"].round(2)
        ranking_display["연환산 상승률 (%)"] = ranking_display["연환산 상승률 (%)"].round(2)
        ranking_display["시작일"] = ranking_display["시작일"].dt.strftime("%Y-%m-%d")
        ranking_display["종료일"] = ranking_display["종료일"].dt.strftime("%Y-%m-%d")
    metric_dataframe(st, ranking_display, hide_index=True, width="stretch")

    st.subheader("원자료")
    display = pd.DataFrame({
        "시점": data["WRTTIME_DESC"],
        "지역": data.get("CLS_FULLNM", data.get("CLS_NM")),
        "항목": data["ITM_NM"],
        f"값 ({unit})" if unit else "값": data["값"].round(4),
        "기간 변동률 (%)": data["기간 변동률 (%)"].round(4),
        "누적 변화 (%)": data["누적 변화 (%)"].round(4),
    })
    metric_dataframe(st, display, hide_index=True, width="stretch")

    back_col, save_col = st.columns(2)
    if back_col.button("뒤로", key="reb_result_back", use_container_width=True):
        st.session_state["current_page"] = "reb_statistics_conditions"; st.rerun()
    save_col.download_button(
        "결과 저장",
        data=_excel_bytes(display.assign(날짜=data["날짜"])),
        file_name="r-one-real-estate-statistics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="reb_result_save",
        type="primary",
        use_container_width=True,
    )


def render(page: str) -> None:
    if page == "reb_statistics_conditions":
        _conditions()
    elif page == "reb_statistics_results":
        _results()
    else:
        st.session_state["current_page"] = "reb_statistics_conditions"
        st.rerun()

