"""NiceGUI page for the single-asset split-trading backtest."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date, datetime
import math
from zoneinfo import ZoneInfo

import pandas as pd
from nicegui import app, run, ui

from common.snowball.layout import shell
from .analytics import (
    annual_performance,
    execution_warnings,
    operational_metrics,
    period_validation,
)
from .data import load_adjusted_close, search_symbols
from .engine import SplitTradingConfig, run_split_backtest
from .optimization import (
    rate_sweep,
    run_combined_sweep,
    run_ratio_sweep,
    run_tranche_sweep,
)
from .persistence import (
    deserialize_optimization,
    deserialize_result,
    serialize_optimization,
    serialize_result,
)
from .sheets import connect_spreadsheet, export_result


FORM_STORAGE_KEY = "split_trading_form_v1"
LAST_RESULT_STORAGE_KEY = "split_trading_last_result_v1"
COMPARISON_STORAGE_KEY = "split_trading_comparison_v1"
FORM_VERSION = 2


def _money(value: float, country: str) -> str:
    return f"₩{value:,.0f}" if country == "한국" else f"${value:,.2f}"


def _actual_money(raw_value: float, country: str) -> float:
    """Interpret the input directly in the selected market currency."""
    return round(float(raw_value), 0 if country == "한국" else 2)


def _tranche_from_count(capital: float, count: int, country: str) -> float:
    if count < 1:
        raise ValueError("분할횟수는 1 이상의 정수여야 합니다.")
    unit = 1 if country == "한국" else .01
    return math.floor((capital / count + 1e-12) / unit) * unit


def _number(value: float, country: str) -> str:
    """Format market-currency prices and amounts without changing calculations."""
    return f"{float(value):,.0f}" if country == "한국" else f"{float(value):,.2f}"


def _table_columns(names: list[str]) -> list[dict[str, object]]:
    return [{"name": name, "label": name, "field": name, "align": "left", "sortable": True}
            for name in names]


def _render_normalized_chart(result: dict, country: str) -> None:
    """Compare both NAV series as if each started at 100."""
    initial = float(result.get("initial_capital", result["full_nav"].iloc[0]))
    full = result["full_nav"] / initial * 100
    split = result["split_nav"] / initial * 100
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in split.index]
    buy_points, sell_points = [], []
    for row in result["trades"].to_dict("records"):
        day = pd.Timestamp(row["거래일"])
        if day not in split.index:
            continue
        point = {
            "value": [day.strftime("%Y-%m-%d"), round(float(split.loc[day]), 3)],
            "name": f"{row['매수/매도']} · {row['차수']}차 · {_number(row['시뮬레이션 체결가'], country)}",
        }
        (buy_points if row["매수/매도"] == "매수" else sell_points).append(point)
    ui.label("초기자산 100 기준 비교").classes("font-bold")
    ui.echart({
        "tooltip": {"trigger": "axis"}, "legend": {"bottom": 0},
        "grid": {"left": 62, "right": 20, "top": 20, "bottom": 64},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
        "yAxis": {"type": "value", "name": "100 기준", "scale": True},
        "dataZoom": [{"type": "inside"}],
        "series": [
            {"name": "전액 투자", "type": "line", "showSymbol": False,
             "data": full.round(3).tolist()},
            {"name": "분할매매", "type": "line", "showSymbol": False,
             "data": split.round(3).tolist(), "lineStyle": {"width": 3}},
            {"name": "매수", "type": "scatter", "data": buy_points,
             "symbol": "triangle", "symbolSize": 10, "itemStyle": {"color": "#d95c5c"}},
            {"name": "매도", "type": "scatter", "data": sell_points,
             "symbol": "triangle", "symbolRotate": 180, "symbolSize": 10,
             "itemStyle": {"color": "#3f9b65"}},
        ],
    }).classes("dashboard-chart w-full")


def _render_result(result: dict, country: str, symbol_label: str) -> None:
    ui.label(f"{symbol_label} 결과").classes("section-title")
    _render_normalized_chart(result, country)
    operations = operational_metrics(result)
    cards = (
        ("전액 투자 최종 평가액", _money(result["full_final"], country)),
        ("전액 투자 총수익률", f"{result['full_return']:.2%}"),
        ("분할매매 최종 평가액", _money(result["split_final"], country)),
        ("분할매매 총수익률", f"{result['split_return']:.2%}"),
        ("수익률 차이", f"{result['return_difference'] * 100:+.2f}%p"),
        ("최종 현금잔액", _money(result["final_cash"], country)),
        ("최종 보유분 평가액", _money(result["final_holdings_value"], country)),
        ("현재 보유분 매입원금", _money(result.get("open_cost_basis", 0), country)),
        ("매수 횟수", f"{operations['매수 횟수']}회"),
        ("매도 횟수", f"{operations['매도 횟수']}회"),
        ("현재 보유 차수", f"{operations['미청산 보유 차수']}개"),
        ("남은 매수 가능 횟수", f"{operations['남은 매수 가능 횟수']}회"),
        ("동시 보유 가능 차수", f"{result['max_slots']}차"),
        ("전액 투자 MDD", f"{result['full_max_drawdown']:.2%}"),
        ("분할매매 MDD", f"{result['split_max_drawdown']:.2%}"),
        ("분할매매 현재 낙폭", f"{result['split_current_drawdown']:.2%}"),
        (
            "분할매매 수익률/MDD",
            "-" if result["split_return_mdd"] is None else f"{result['split_return_mdd']:.2f}",
        ),
    )
    with ui.grid().classes("result-metric-grid w-full gap-3"):
        for label, value in cards:
            with ui.card().classes("metric-card w-full"):
                with ui.row().classes("metric-row w-full items-center justify-between no-wrap"):
                    ui.label(label).classes("metric-label")
                    ui.label(value).classes("metric-value")
    recovery = result["split_mdd_recovery_date"] or "회복 전"
    ui.label(
        "분할매매 최대 낙폭 구간 · "
        f"{result['split_mdd_peak_date']} → {result['split_mdd_trough_date']} · "
        f"회복 {recovery} · {result['split_mdd_duration_days']}일"
    ).classes("text-sm text-grey-6")
    with ui.expansion("운용 품질·기간 검증·경고", icon="fact_check", value=False).classes(
        "strategy-expansion w-full"
    ):
        operation_cards = (
            ("매수 횟수", f"{operations['매수 횟수']}회"),
            ("매도 횟수", f"{operations['매도 횟수']}회"),
            ("수익 거래 비율", f"{operations['수익 거래 비율']:.1%}"),
            ("평균 보유기간", f"{operations['평균 보유기간']:.1f}일"),
            ("평균 투자집행률", f"{operations['평균 투자집행률']:.1%}"),
            ("평균 현금비중", f"{operations['평균 현금비중']:.1%}"),
            ("미청산 차수", f"{operations['미청산 보유 차수']}개"),
            ("총 거래비용", _money(operations["총 거래비용"], country)),
        )
        with ui.grid().classes("result-metric-grid w-full gap-2"):
            for label, value in operation_cards:
                with ui.card().classes("metric-card w-full"):
                    with ui.row().classes("metric-row w-full items-center justify-between no-wrap"):
                        ui.label(label).classes("metric-label")
                        ui.label(value).classes("metric-value")
        ui.label("실행 가능성 점검").classes("font-bold")
        for warning in execution_warnings(result):
            ui.label(f"• {warning}").classes("text-sm")
        for title, frame in (
            ("연도별 성과", annual_performance(result)),
            ("기간·시장국면 검증", period_validation(result)),
        ):
            ui.label(title).classes("font-bold mt-2")
            display_frame = frame.copy()
            for column in ("수익률", "MDD", "평균 현금비중"):
                if column in display_frame:
                    display_frame[column] = display_frame[column].map(
                        lambda value: f"{float(value):.2%}"
                    )
            ui.table(
                columns=_table_columns(list(display_frame.columns)),
                rows=display_frame.to_dict("records"),
                pagination={"rowsPerPage": 10},
            ).props("flat bordered dense").classes("w-full account-table")
    ui.label("체결 내역").classes("section-title")
    display = result["trades"].copy()
    for column in (
        "시뮬레이션 체결가", "체결금액", "실현손익", "체결 후 현금잔액",
        "거래비용",
    ):
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else _number(float(value), country)
        )
    quantity_digits = 0 if result["quantity_mode"] == "whole" else 6
    display["수량"] = display["수량"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):,.{quantity_digits}f}"
    )
    display["실현수익률"] = display["실현수익률"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
    )
    ui.table(
        columns=_table_columns(list(display.columns)),
        rows=display.fillna("").to_dict("records"), pagination={"rowsPerPage": 15},
    ).props("flat bordered dense").classes("w-full account-table")
def _optimization_table(frame: pd.DataFrame, country: str) -> pd.DataFrame:
    display = frame.copy()
    for column in ("최종 평가액", "최종 현금", "회당 매수금액"):
        if column in display:
            display[column] = display[column].map(lambda value: _number(value, country))
    for column in (
        "총수익률", "MDD", "현재 낙폭", "매수·매도율", "수익 거래 비율",
        "평균 투자집행률", "평균 현금비중",
    ):
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value):.2%}")
    if "수익률/MDD" in display:
        display["수익률/MDD"] = display["수익률/MDD"].map(lambda value: f"{float(value):.2f}")
    for column in ("개별 종합점수", "안정구간 점수"):
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value):.1f}")
    return display


def _top_strategy_chart(optimization: dict, country: str) -> None:
    top = optimization["runs"][:5]
    index = top[0]["result"]["split_nav"].index
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in index]
    digits = 0 if country == "한국" else 2
    series = []
    for rank, item in enumerate(top, start=1):
        parameters = item["parameters"]
        if "분할 조건" in parameters and "매수·매도율" in parameters:
            name = f"{rank}위 · {parameters['분할 조건']} / {parameters['매수·매도율']:.0%}"
        elif "매수·매도율" in parameters:
            name = f"{rank}위 · ±{parameters['매수·매도율']:.0%}"
        else:
            name = f"{rank}위 · {parameters['분할 조건']}"
        series.append({
            "name": name, "type": "line", "showSymbol": False,
            "data": item["result"]["split_nav"].round(digits).tolist(),
            "lineStyle": {"width": 3 if rank == 1 else 1.5},
        })
    ui.label("상위 5개 전략 평가액").classes("section-title")
    ui.echart({
        "tooltip": {"trigger": "axis"}, "legend": {"bottom": 0, "type": "scroll"},
        "grid": {"left": 72, "right": 24, "top": 24, "bottom": 74},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
        "yAxis": {"type": "value", "scale": True},
        "dataZoom": [{"type": "inside"}], "series": series,
    }).classes("dashboard-chart w-full")


def _paired_rate_chart(optimization: dict) -> None:
    frame = optimization["table"].sort_values("매수·매도율")
    labels = [f"{value:.0%}" for value in frame["매수·매도율"]]
    ui.label("매수·매도율별 수익률과 MDD").classes("section-title")
    ui.echart({
        "tooltip": {"trigger": "axis"}, "legend": {"bottom": 0},
        "grid": {"left": 58, "right": 24, "top": 24, "bottom": 64},
        "xAxis": {"type": "category", "name": "매수·매도율", "data": labels},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}%"}},
        "series": [
            {"name": "총수익률", "type": "line", "data": (frame["총수익률"] * 100).round(2).tolist()},
            {"name": "MDD", "type": "line", "data": (frame["MDD"] * 100).round(2).tolist()},
        ],
    }).classes("monthly-distribution w-full")


def _combined_heatmap(
    optimization: dict, metric: str, title: str, country: str,
) -> None:
    frame = optimization["table"]
    condition_rows = (
        frame[["분할 조건", "회당 매수금액", "동시 보유 차수"]]
        .drop_duplicates("분할 조건")
        .sort_values("동시 보유 차수")
    )
    conditions = condition_rows["분할 조건"].tolist()
    amount_labels = {
        row["분할 조건"]: f"{row['분할 조건']}\n{_money(row['회당 매수금액'], country)}"
        for row in condition_rows.to_dict("records")
    }
    rates = sorted(frame["매수·매도율"].unique())
    expected_cells = len(conditions) * len(rates)
    unique_cells = frame[["분할 조건", "매수·매도율"]].drop_duplicates()
    if len(frame) != expected_cells or len(unique_cells) != expected_cells:
        raise ValueError(
            "히트맵 조합 수가 올바르지 않습니다: "
            f"{len(conditions)}개 횟수 × {len(rates)}개 비율 = "
            f"{expected_cells}개여야 하지만 {len(frame)}개가 계산되었습니다."
        )
    condition_pos = {value: index for index, value in enumerate(conditions)}
    rate_pos = {value: index for index, value in enumerate(rates)}
    data = [
        [rate_pos[row["매수·매도율"]], condition_pos[row["분할 조건"]],
         round(float(row[metric]) * (100 if metric in {"총수익률", "MDD"} else 1), 2)]
        for row in frame.to_dict("records")
    ]
    values = [item[2] for item in data]
    visual_min, visual_max = min(values), max(values)
    if visual_min == visual_max:
        visual_min -= 1
        visual_max += 1
    ui.label(title).classes("section-title")
    ui.label(
        f"{len(conditions)}개 분할횟수 × {len(rates)}개 매수·매도율 "
        f"= 총 {expected_cells}개 조건"
    ).classes("text-sm text-grey-6")
    minimum_width = max(720, len(rates) * 76 + 210)
    with ui.element("div").classes("w-full overflow-x-auto"):
        ui.echart({
            "tooltip": {"position": "top"},
            "grid": {"left": 130, "right": 20, "top": 20, "bottom": 45},
            "xAxis": {
                "type": "category", "name": "매수·매도율",
                "data": [f"{v:.0%}" for v in rates],
            },
            "yAxis": {
                "type": "category", "name": "회당 매수금액",
                "data": [amount_labels[value] for value in conditions],
                "axisLabel": {"fontSize": 10, "lineHeight": 13},
            },
            "visualMap": {
                "show": False, "min": visual_min, "max": visual_max,
                "inRange": {"color": ["#d95c5c", "#fff7f7", "#f3fbf5", "#3f9b65"]},
            },
            "series": [{
                "type": "heatmap", "data": data,
                "label": {"show": True, "fontSize": 10},
            }],
        }).style(
            f"min-width:{minimum_width}px;"
            f"height:{max(430, len(conditions) * 40 + 110)}px"
        )


def _render_result_preview(result: dict, country: str) -> None:
    """Render an already-calculated winner without its long trade table."""
    _render_normalized_chart(result, country)
    operations = operational_metrics(result)
    metrics = (
        ("최종 평가액", _money(result["split_final"], country)),
        ("총수익률", f"{result['split_return']:.2%}"),
        ("MDD", f"{result['split_max_drawdown']:.2%}"),
        ("현재 낙폭", f"{result['split_current_drawdown']:.2%}"),
        (
            "수익률/MDD",
            "-" if result["split_return_mdd"] is None else f"{result['split_return_mdd']:.2f}",
        ),
        ("매수 횟수", f"{operations['매수 횟수']}회"),
        ("매도 횟수", f"{operations['매도 횟수']}회"),
        ("현재 보유 차수", f"{operations['미청산 보유 차수']}개"),
        ("남은 매수 가능 횟수", f"{operations['남은 매수 가능 횟수']}회"),
        ("현재 보유분 매입원금", _money(operations["현재 보유분 매입원금"], country)),
        ("현재 보유분 평가금", _money(operations["현재 보유분 평가금"], country)),
        ("현금자산", _money(operations["현금자산"], country)),
    )
    with ui.grid().classes("result-metric-grid w-full gap-2"):
        for label, value in metrics:
            with ui.card().classes("metric-card w-full"):
                with ui.row().classes("metric-row w-full items-center justify-between no-wrap"):
                    ui.label(label).classes("metric-label")
                    ui.label(value).classes("metric-value")


def _best_heatmap_condition(
    optimization: dict, metric: str, title: str, country: str,
    on_save: Callable[[dict, dict], None] | None = None,
) -> None:
    frame = optimization["table"]
    best = frame.loc[frame[metric].astype(float).idxmax()]
    value = (
        f"{float(best[metric]):.2%}"
        if metric in {"총수익률", "MDD"}
        else f"{float(best[metric]):.1f}점"
        if metric == "안정구간 점수"
        else f"{float(best[metric]):.2f}"
    )
    with ui.card().classes("metric-card w-full"):
        ui.label(f"{title} 최고 조건").classes("font-bold text-primary")
        with ui.row().classes("w-full items-center justify-between gap-3"):
            ui.label(
                f"{best['분할 조건']} · 회당 {_money(float(best['회당 매수금액']), country)} "
                f"· 매수·매도율 {float(best['매수·매도율']):.0%}"
            ).classes("text-sm")
            ui.label(value).classes("metric-value")
        preview_result = optimization.get("metric_winners", {}).get(metric)
        if preview_result is not None:
            with ui.column().classes("w-full gap-2"):
                toggle = ui.button(
                    "상세 결과 펼치기", icon="expand_more",
                ).props("outline no-caps").classes("w-full")
                detail = ui.column().classes("strategy-expansion w-full gap-3")
                detail.set_visibility(False)
                loaded = {"value": False}

                def toggle_detail() -> None:
                    opening = not bool(detail.visible)
                    detail.set_visibility(opening)
                    toggle.set_text("상세 결과 접기" if opening else "상세 결과 펼치기")
                    if opening and not loaded["value"]:
                        with detail:
                            _render_result_preview(preview_result, country)
                        loaded["value"] = True

                toggle.on("click", toggle_detail)
                if on_save is not None:
                    ui.button(
                        "조건 비교함에 저장", icon="bookmark_add",
                        on_click=lambda: on_save(best.to_dict(), preview_result),
                    ).props("outline no-caps").classes("w-full")


def _allocation_chart(optimization: dict) -> None:
    frame = optimization["table"].sort_values("동시 보유 차수")
    labels = frame["분할 조건"].astype(str).tolist()
    ui.label("분할 규모별 수익률과 MDD").classes("section-title")
    ui.echart({
        "tooltip": {"trigger": "axis"}, "legend": {"bottom": 0},
        "grid": {"left": 58, "right": 24, "top": 24, "bottom": 70},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 30}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}%"}},
        "series": [
            {"name": "총수익률", "type": "line", "data": (frame["총수익률"] * 100).round(2).tolist()},
            {"name": "MDD", "type": "line", "data": (frame["MDD"] * 100).round(2).tolist()},
        ],
    }).classes("monthly-distribution w-full")


def _render_optimization(
    optimization: dict, country: str, symbol_label: str, mode: str,
    on_save: Callable[[dict, dict], None] | None = None,
) -> None:
    ui.label(f"{symbol_label} 조건 비교 결과").classes("section-title")
    ui.label(
        f"총 {len(optimization['table']):,}개 조건을 계산했습니다. "
        "수익률과 MDD를 함께 비교해 원하는 조건을 선택하세요."
    ).classes("page-subtitle")
    if mode == "ratio":
        _paired_rate_chart(optimization)
    elif mode == "combined":
        _best_heatmap_condition(
            optimization, "안정구간 점수", "안정 구간 추천", country, on_save,
        )
        _best_heatmap_condition(
            optimization, "총수익률", "총수익률", country, on_save,
        )
        _combined_heatmap(
            optimization, "총수익률",
            "회당 매수금액 × 매수·매도율 · 총수익률", country,
        )
        _best_heatmap_condition(
            optimization, "MDD", "MDD", country, on_save,
        )
        _combined_heatmap(
            optimization, "MDD",
            "회당 매수금액 × 매수·매도율 · MDD", country,
        )
        _best_heatmap_condition(
            optimization, "수익률/MDD", "수익률/MDD", country, on_save,
        )
        _combined_heatmap(
            optimization, "수익률/MDD",
            "회당 매수금액 × 매수·매도율 · 수익률/MDD", country,
        )
    else:
        _allocation_chart(optimization)
    ui.label("전체 조합 결과").classes("section-title")
    raw_table = optimization["table"].drop(columns=["순위"], errors="ignore")
    sort_columns = [
        column for column in ("동시 보유 차수", "매수·매도율")
        if column in raw_table.columns
    ]
    if sort_columns:
        raw_table = raw_table.sort_values(sort_columns)
    display = _optimization_table(raw_table, country)
    ui.table(
        columns=_table_columns(list(display.columns)),
        rows=display.fillna("").to_dict("records"), pagination={"rowsPerPage": 15},
    ).props("flat bordered dense").classes("w-full account-table")
    ui.button(
        "최적화 전체 결과 CSV", icon="download",
        on_click=lambda: ui.download.content(
            optimization["table"].to_csv(index=False).encode("utf-8-sig"),
            filename="split_trading_optimization.csv", media_type="text/csv",
        ),
    ).props("outline no-caps")


@ui.page("/split-trading")
def split_trading_page() -> None:
    shell()
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    default_start = (pd.Timestamp(today) - pd.DateOffset(years=3)).date()
    stored_value = app.storage.user.get(FORM_STORAGE_KEY, {})
    saved = stored_value if isinstance(stored_value, dict) else {}
    saved_form_version = int(saved.get("form_version", 1))
    saved_country = str(saved.get("country", "한국"))
    if saved_country not in {"한국", "미국"}:
        saved_country = "한국"
    saved_symbol = str(saved.get("symbol", "")).strip()
    saved_symbol_label = str(saved.get("symbol_label", saved_symbol)).strip()
    saved_period = str(saved.get("period", "3"))
    if saved_period not in {*map(str, range(1, 11)), "custom"}:
        saved_period = "3"
    saved_test_mode = str(saved.get("test_mode", "single"))
    if saved_test_mode not in {"single", "ratio", "allocation", "combined"}:
        saved_test_mode = "single"
    state: dict[str, object] = {
        "result": None, "conditions": None, "search_version": 0, "date_sync": False,
        "restoring": True,
    }

    with ui.column().classes("page-container w-full gap-4"):
        with ui.row().classes("items-center gap-2"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("round flat")
            with ui.column().classes("gap-0"):
                ui.label("분할매매 백테스트").classes("page-title")
                ui.label("1차는 보유하고 2차 이후 매수·매도를 반복해 전액 투자와 비교합니다.").classes("page-subtitle")

        with ui.card().classes("backtest-controls w-full"):
            test_mode = ui.select(
                {
                    "single": "단일 백테스트",
                    "ratio": "매수·매도율 비교",
                    "allocation": "분할횟수 비교",
                    "combined": "통합 비교 · 횟수 × 매수·매도율",
                },
                value=saved_test_mode, label="테스트 유형",
            ).props("outlined").classes("w-full")
            country = ui.select(["한국", "미국"], value=saved_country, label="국가").props("outlined").classes("control-grow")
            symbol_query = ui.input(
                "종목명 또는 코드", value=str(saved.get("symbol_query", "")),
                placeholder="입력하면 종목 후보를 찾습니다",
            ).props("outlined clearable").classes("w-full")
            symbol_options = {saved_symbol: saved_symbol_label} if saved_symbol else {}
            symbol_select = ui.select(
                symbol_options, value=saved_symbol or None, label="검색 결과",
            ).props("outlined options-dense").classes("w-full")
            search_status = ui.label(
                "저장된 종목을 복원했습니다." if saved_symbol else "종목명이나 코드를 입력하세요."
            ).classes("status-text")
            period = ui.select(
                {**{str(year): f"{year}년" for year in range(1, 11)}, "custom": "직접 설정"},
                value=saved_period, label="기간",
            ).props("outlined").classes("control-grow")
            with ui.row().classes("backtest-field-grid backtest-date-grid w-full"):
                start_input = ui.input(
                    "From", value=str(saved.get("start", default_start.isoformat())),
                ).props("type=date").classes("date-input")
                end_input = ui.input(
                    "To", value=str(saved.get("end", today.isoformat())),
                ).props("type=date").classes("date-input")
            with ui.row().classes("backtest-field-grid w-full"):
                initial_raw = ui.number(
                    "최초 투자총액", value=float(saved.get("initial", 10_000_000)), min=.01, step=1,
                ).props("outlined").classes("control-grow")
                count_custom = ui.number(
                    "분할횟수 (최적화 시 최대횟수)",
                    value=int(saved.get("count_custom", 10)), min=1, max=100, step=1,
                ).props("outlined").classes("control-grow")
            tranche_display = ui.input("계산된 회당 매수금액").props("outlined readonly").classes("w-full")
            with ui.row().classes("backtest-field-grid w-full"):
                rate_preset = ui.select(
                    ["직접입력", *range(5, 51, 5)], value=saved.get("rate_preset", 10),
                    label="매수·매도율",
                ).props("outlined").classes("control-grow")
                rate_custom = ui.number(
                    "직접 매수·매도율(%)", value=float(saved.get("rate_custom", 10)),
                    min=.01, max=99.99, step=.1,
                ).props("outlined").classes("control-grow")
            with ui.card().classes("editor-section w-full") as rate_sweep_card:
                ui.label("매수·매도율 탐색 범위 · 5%p 단위").classes("font-bold")
                with ui.row().classes("backtest-field-grid w-full"):
                    rate_min = ui.number(
                        "최소 매수·매도율(%)", value=float(saved.get("rate_min", 10)),
                        min=.01, max=99.99, step=5,
                    ).props("outlined").classes("control-grow")
                    rate_max = ui.number(
                        "최대 매수·매도율(%)", value=float(saved.get("rate_max", 30)),
                        min=.01, max=99.99, step=5,
                    ).props("outlined").classes("control-grow")
            quantity_mode = ui.select(
                {
                    "whole": "1주 단위 정수 매매",
                    "fractional_6": "금액 기준 소수점 매매(6자리)",
                },
                value=(
                    str(saved.get("quantity_mode", "whole"))
                    if saved_form_version >= FORM_VERSION else "whole"
                ),
                label="수량 방식",
            ).props("outlined").classes("w-full")
            with ui.expansion("현실 비용 반영 (선택)", icon="payments").classes(
                "strategy-expansion w-full"
            ):
                ui.label("모든 값은 체결금액 대비 비율이며 기본값 0%입니다.").classes(
                    "text-xs text-grey-6"
                )
                with ui.column().classes("w-full gap-3"):
                    buy_fee = ui.number(
                        "매수수수료(%)", value=float(saved.get("buy_fee", 0)),
                        min=0, max=99, step=.01,
                    ).props("outlined").classes("w-full")
                    sell_fee = ui.number(
                        "매도수수료(%)", value=float(saved.get("sell_fee", 0)),
                        min=0, max=99, step=.01,
                    ).props("outlined").classes("w-full")
                    sell_tax = ui.number(
                        "매도세금(%)", value=float(saved.get("sell_tax", 0)),
                        min=0, max=99, step=.01,
                    ).props("outlined").classes("w-full")
                    slippage = ui.number(
                        "슬리피지(%)", value=float(saved.get("slippage", 0)),
                        min=0, max=99, step=.01,
                    ).props("outlined").classes("w-full")
                    fx_cost = ui.number(
                        "환전비용(%)", value=float(saved.get("fx_cost", 0)),
                        min=0, max=99, step=.01,
                    ).props("outlined").classes("w-full")
            currency_help = ui.label("").classes("text-xs text-grey-6")
            run_button = ui.button("백테스트 실행", icon="play_arrow").classes("primary-action w-full")

        status = ui.label("조건을 입력한 뒤 실행하세요.").classes("status-text")
        comparison_area = ui.column().classes("w-full gap-3")
        result_area = ui.column().classes("w-full gap-4")

        def persist_form() -> None:
            if state["restoring"]:
                return
            selected_symbol = str(symbol_select.value or "")
            options = symbol_select.options if isinstance(symbol_select.options, dict) else {}
            app.storage.user[FORM_STORAGE_KEY] = {
                "form_version": FORM_VERSION,
                "test_mode": str(test_mode.value),
                "country": str(country.value),
                "symbol_query": str(symbol_query.value or ""),
                "symbol": selected_symbol,
                "symbol_label": str(options.get(selected_symbol, selected_symbol)),
                "period": str(period.value),
                "start": str(start_input.value),
                "end": str(end_input.value),
                "initial": float(initial_raw.value or 0),
                "count_custom": int(count_custom.value or 1),
                "rate_preset": rate_preset.value,
                "rate_custom": float(rate_custom.value or 0),
                "rate_min": float(rate_min.value or 0),
                "rate_max": float(rate_max.value or 0),
                "quantity_mode": str(quantity_mode.value),
                "buy_fee": float(buy_fee.value or 0),
                "sell_fee": float(sell_fee.value or 0),
                "sell_tax": float(sell_tax.value or 0),
                "slippage": float(slippage.value or 0),
                "fx_cost": float(fx_cost.value or 0),
            }

        def update_currency_inputs() -> None:
            is_us = country.value == "미국"
            step = .01 if is_us else 1
            digits = 2 if is_us else 0
            for control in (initial_raw,):
                if control.value is not None:
                    normalized = round(float(control.value), digits)
                    if float(control.value) != normalized:
                        control.value = normalized
            initial_raw.props(f"step={step}")
            currency_help.set_text(
                "달러 금액은 USD를 직접 입력하며 소수점 둘째 자리까지 사용할 수 있습니다."
                if is_us else
                "원화 금액은 원 단위 정수로 입력합니다."
            )

        def update_dynamic_controls() -> None:
            update_currency_inputs()
            mode = str(test_mode.value)
            optimizing_counts = mode in {"allocation", "combined"}
            optimizing_rates = mode in {"ratio", "combined"}
            fixed_ratio = mode in {"single", "allocation"}
            rate_sweep_card.set_visibility(optimizing_rates)
            tranche_display.set_visibility(not optimizing_counts)
            rate_preset.set_visibility(fixed_ratio)
            rate_custom.set_visibility(
                fixed_ratio and rate_preset.value == "직접입력"
            )
            raw_total = float(initial_raw.value or 0)
            count_value = int(count_custom.value or 1)
            raw_tranche = _tranche_from_count(
                raw_total, max(1, count_value), str(country.value),
            )
            tranche_display.value = _money(raw_tranche, str(country.value))
            run_button.set_text(
                "백테스트 실행" if mode == "single" else "조건 비교 실행"
            )
            persist_form()

        def apply_period() -> None:
            if period.value == "custom":
                return
            end = datetime.now(ZoneInfo("Asia/Seoul")).date()
            state["date_sync"] = True
            end_input.value = end.isoformat()
            start_input.value = (pd.Timestamp(end) - pd.DateOffset(years=int(period.value))).date().isoformat()
            ui.timer(.05, lambda: state.__setitem__("date_sync", False), once=True)

        def mark_custom() -> None:
            if not state["date_sync"]:
                period.value = "custom"

        async def search() -> None:
            state["search_version"] = int(state["search_version"]) + 1
            version = int(state["search_version"])
            query = str(symbol_query.value or "").strip()
            if not query:
                symbol_select.options = {}
                symbol_select.value = None
                symbol_select.update()
                return
            await asyncio.sleep(.45)
            if version != state["search_version"]:
                return
            search_status.set_text("종목을 검색하고 있습니다…")
            try:
                options = await run.io_bound(search_symbols, query, str(country.value))
            except Exception as error:
                search_status.set_text(f"종목 검색 실패: {error}")
                return
            if version != state["search_version"]:
                return
            symbol_select.options = options
            symbol_select.value = next(iter(options), None)
            symbol_select.update()
            search_status.set_text(f"{len(options)}개 후보를 찾았습니다." if options else "일치하는 종목이 없습니다.")
            persist_form()

        def reset_symbol() -> None:
            symbol_query.value = ""
            symbol_select.options = {}
            symbol_select.value = None
            symbol_select.update()
            update_dynamic_controls()
            persist_form()

        def render_comparison_box() -> None:
            comparison_area.clear()
            raw_items = app.storage.user.get(COMPARISON_STORAGE_KEY, [])
            items = raw_items if isinstance(raw_items, list) else []
            if not items:
                return
            with comparison_area:
                with ui.expansion(
                    f"조건 비교함 · {len(items)}개", icon="compare_arrows", value=True,
                ).classes("strategy-expansion w-full"):
                    chart_series = []
                    rows = []
                    for index, item in enumerate(items):
                        restored = deserialize_result(item["result"])
                        chart_series.append({
                            "name": item["name"], "type": "line", "showSymbol": False,
                            "data": [
                                [timestamp.isoformat(), round(float(value), 2)]
                                for timestamp, value in restored["split_nav"].items()
                            ],
                        })
                        rows.append({
                            "조건": item["name"],
                            "총수익률": f"{float(item['return']):.2%}",
                            "MDD": f"{float(item['mdd']):.2%}",
                            "수익률/MDD": f"{float(item['return_mdd']):.2f}",
                            "안정점수": f"{float(item.get('stability', 0)):.1f}",
                        })
                    ui.echart({
                        "tooltip": {"trigger": "axis"}, "legend": {"bottom": 0, "type": "scroll"},
                        "grid": {"left": 70, "right": 20, "top": 20, "bottom": 72},
                        "xAxis": {"type": "time"}, "yAxis": {"type": "value", "scale": True},
                        "dataZoom": [{"type": "inside"}], "series": chart_series,
                    }).classes("dashboard-chart w-full")
                    ui.table(
                        columns=_table_columns(list(rows[0])), rows=rows,
                        pagination={"rowsPerPage": 8},
                    ).props("flat bordered dense").classes("w-full account-table")
                    with ui.row().classes("w-full gap-2"):
                        for index, item in enumerate(items):
                            def remove(saved_index: int = index) -> None:
                                current = list(app.storage.user.get(COMPARISON_STORAGE_KEY, []))
                                if 0 <= saved_index < len(current):
                                    current.pop(saved_index)
                                app.storage.user[COMPARISON_STORAGE_KEY] = current
                                render_comparison_box()
                            ui.button(
                                f"{index + 1}번 삭제", icon="delete", on_click=remove,
                            ).props("flat dense no-caps")

        def save_comparison(
            row: dict, result: dict, selected_country: str,
            label: str, conditions: dict,
        ) -> None:
            count = int(row["동시 보유 차수"])
            rate = float(row["매수·매도율"])
            name = f"{label.split(' · ', 1)[0]} · {count}회 · {rate:.0%}"
            item = {
                "key": (
                    f"{label}|{conditions.get('요청 기간')}|{count}|{rate:.8f}|"
                    f"{conditions.get('수량 방식')}|{conditions.get('비용 설정')}"
                ),
                "name": name, "country": selected_country,
                "return": float(row["총수익률"]), "mdd": float(row["MDD"]),
                "return_mdd": float(row["수익률/MDD"]),
                "stability": float(row.get("안정구간 점수", 0)),
                "result": serialize_result(result),
            }
            items = list(app.storage.user.get(COMPARISON_STORAGE_KEY, []))
            items = [saved for saved in items if saved.get("key") != item["key"]]
            items.append(item)
            app.storage.user[COMPARISON_STORAGE_KEY] = items[-8:]
            render_comparison_box()
            ui.notify("조건 비교함에 저장했습니다.", type="positive")

        def show_output(
            result: dict | None,
            optimization: dict | None,
            selected_country: str,
            label: str,
            selected_test_mode: str,
            conditions: dict,
            actual_start: object,
            actual_end: object,
        ) -> None:
            result_area.clear()
            status.set_text(f"{actual_start} ~ {actual_end} 수정종가 기준")
            with result_area:
                if optimization is None and result is not None:
                    _render_result(result, selected_country, label)
                elif optimization is not None:
                    _render_optimization(
                        optimization, selected_country, label, selected_test_mode,
                        lambda row, winner: save_comparison(
                            row, winner, selected_country, label, conditions,
                        ),
                    )
                if optimization is None and result is not None:
                    export_status = ui.label(
                        "현재 결과를 연결된 Google Sheets의 새 탭으로 저장할 수 있습니다."
                    ).classes("status-text")
                    export_button = ui.button(
                        "Google Sheets로 내보내기", icon="table_view",
                    ).classes("primary-action")

                    async def export_sheet() -> None:
                        export_button.disable()
                        export_status.set_text("새 Google Sheets 탭을 만드는 중입니다…")
                        try:
                            spreadsheet = await run.io_bound(connect_spreadsheet)
                            title, url = await run.io_bound(
                                export_result, spreadsheet,
                                symbol_name=label.split(" · ", 1)[-1],
                                conditions=conditions, result=result,
                            )
                        except Exception as error:
                            export_status.set_text(f"내보내기 실패: {error}")
                        else:
                            export_status.set_text(f"'{title}' 탭에 저장했습니다.")
                            ui.link("새 탭 열기", url, new_tab=True).classes(
                                "text-primary font-bold"
                            )
                        finally:
                            export_button.enable()

                    export_button.on("click", export_sheet)

        async def execute() -> None:
            persist_form()
            run_button.disable()
            result_area.clear()
            status.set_text("수정종가를 불러와 분할매매를 계산하고 있습니다…")
            try:
                if not symbol_select.value:
                    raise ValueError("검색 결과에서 종목을 선택해 주세요.")
                start = date.fromisoformat(str(start_input.value))
                end = date.fromisoformat(str(end_input.value))
                if start > end:
                    raise ValueError("From은 To보다 늦을 수 없습니다.")
                selected_country = str(country.value)
                selected_test_mode = str(test_mode.value)
                capital = _actual_money(float(initial_raw.value or 0), selected_country)
                count_value = int(count_custom.value or 1)
                if count_value < 1:
                    raise ValueError("분할횟수는 1 이상의 정수여야 합니다.")
                if count_value > 100:
                    raise ValueError("분할횟수는 최대 100회까지 비교할 수 있습니다.")
                tranche = _tranche_from_count(capital, count_value, selected_country)
                base_slot_count = count_value
                split_input = f"{count_value}회 · 회당 {_money(tranche, selected_country)}"
                if selected_test_mode in {"allocation", "combined"}:
                    split_input = f"1~{count_value}회 자동 비교"
                if selected_test_mode in {"ratio", "combined"}:
                    rate = .10
                else:
                    rate = float(
                        rate_custom.value if rate_preset.value == "직접입력"
                        else rate_preset.value
                    ) / 100
                buy = sell = rate
                selected_quantity_mode = str(quantity_mode.value)
                config = SplitTradingConfig(
                    capital, tranche, buy, sell, selected_quantity_mode,
                    base_slot_count,
                    float(buy_fee.value or 0) / 100,
                    float(sell_fee.value or 0) / 100,
                    float(sell_tax.value or 0) / 100,
                    float(slippage.value or 0) / 100,
                    float(fx_cost.value or 0) / 100,
                )
                symbol = str(symbol_select.value)
                close = await run.io_bound(load_adjusted_close, symbol, start, end)
                label = str(symbol_select.options.get(symbol, symbol))
                optimization = None
                if selected_test_mode == "single":
                    result = await run.io_bound(run_split_backtest, close, config)
                elif selected_test_mode == "ratio":
                    rates = rate_sweep(
                        float(rate_min.value or 0), float(rate_max.value or 0),
                    )
                    status.set_text(f"{len(rates)}개 매수·매도율을 계산하고 있습니다…")
                    optimization = await run.io_bound(
                        run_ratio_sweep, close, config, rates, "return",
                    )
                    result = optimization["best"]
                    rate = float(optimization["best_parameters"]["매수·매도율"])
                    buy = sell = rate
                    split_input += (
                        f" · ±{float(rate_min.value):g}~{float(rate_max.value):g}% 탐색"
                    )
                else:
                    counts = list(range(1, count_value + 1))
                    candidates = [
                        (
                            f"{count_value}회",
                            _tranche_from_count(capital, count_value, selected_country),
                            count_value,
                        )
                        for count_value in counts
                    ]
                    estimated_slots = sum(
                        slot_count
                        for _, tranche_value, slot_count in candidates if tranche_value > 0
                    )
                    rates = (
                        rate_sweep(
                            float(rate_min.value or 0), float(rate_max.value or 0),
                        )
                        if selected_test_mode == "combined" else []
                    )
                    rate_multiplier = len(rates) if selected_test_mode == "combined" else 1
                    if estimated_slots * rate_multiplier > 20_000:
                        raise ValueError(
                            "분할횟수 범위의 총 계산량이 너무 큽니다. 범위를 줄여주세요."
                        )
                    if selected_test_mode == "combined":
                        status.set_text(
                            f"{len(candidates) * len(rates)}개 횟수·비율 조합을 계산하고 있습니다…"
                        )
                        optimization = await run.io_bound(
                            run_combined_sweep, close, config, candidates, rates,
                            "return",
                        )
                        rate = float(optimization["best_parameters"]["매수·매도율"])
                        buy = sell = rate
                    else:
                        status.set_text(f"{len(candidates)}개 분할횟수를 계산하고 있습니다…")
                        optimization = await run.io_bound(
                            run_tranche_sweep, close, config, candidates,
                            "return",
                        )
                    result = optimization["best"]
                    tranche = float(optimization["best_parameters"]["회당 매수금액"])
                conditions = {
                    "국가": selected_country, "종목": label,
                    "테스트 유형": str(test_mode.options.get(selected_test_mode, selected_test_mode)),
                    "요청 기간": f"{start} ~ {end}",
                    "실제 데이터 기간": f"{result['actual_start']} ~ {result['actual_end']}",
                    "최초 투자총액": capital, "분할 방식": split_input,
                    "매수·매도율 탐색 범위": (
                        f"{float(rate_min.value):g}% ~ {float(rate_max.value):g}% · 5%p 단위"
                        if selected_test_mode in {"ratio", "combined"} else "해당 없음"
                    ),
                    "회당 매수금액": tranche, "매수·매도율": rate,
                    "매수 하락률": buy, "매도 상승률": sell,
                    "가격 기준": "yfinance auto_adjust 수정종가",
                    "수량 방식": (
                        "1주 단위 정수 매매" if selected_quantity_mode == "whole"
                        else "금액 기준 소수점 매매(6자리)"
                    ),
                    "비용 설정": (
                        f"매수 {float(buy_fee.value or 0):g}% · "
                        f"매도 {float(sell_fee.value or 0):g}% · "
                        f"세금 {float(sell_tax.value or 0):g}% · "
                        f"슬리피지 {float(slippage.value or 0):g}% · "
                        f"환전 {float(fx_cost.value or 0):g}%"
                    ),
                }
                state.update(
                    result=result, conditions=conditions, symbol_label=label,
                    optimization=optimization, test_mode=selected_test_mode,
                )
            except Exception as error:
                status.set_text("백테스트를 완료하지 못했습니다.")
                with result_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            else:
                app.storage.user[LAST_RESULT_STORAGE_KEY] = {
                    "version": 1,
                    "saved_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                    "country": selected_country,
                    "label": label,
                    "test_mode": selected_test_mode,
                    "conditions": conditions,
                    "actual_start": str(result["actual_start"]),
                    "actual_end": str(result["actual_end"]),
                    "result": serialize_result(result) if optimization is None else None,
                    "optimization": serialize_optimization(optimization),
                }
                show_output(
                    result, optimization, selected_country, label,
                    selected_test_mode, conditions,
                    result["actual_start"], result["actual_end"],
                )
            finally:
                run_button.enable()

        def period_changed() -> None:
            apply_period()
            persist_form()

        def date_changed() -> None:
            mark_custom()
            persist_form()

        async def query_changed() -> None:
            persist_form()
            await search()

        country.on_value_change(lambda _: reset_symbol())
        symbol_query.on_value_change(lambda _: query_changed())
        symbol_select.on_value_change(lambda _: persist_form())
        period.on_value_change(lambda _: period_changed())
        start_input.on_value_change(lambda _: date_changed())
        end_input.on_value_change(lambda _: date_changed())
        for control in (
            test_mode,
            initial_raw, count_custom, rate_preset, rate_custom, quantity_mode,
            rate_min, rate_max, buy_fee, sell_fee, sell_tax, slippage, fx_cost,
        ):
            control.on_value_change(lambda _: update_dynamic_controls())
        run_button.on("click", execute)
        state["restoring"] = False
        update_dynamic_controls()
        render_comparison_box()
        saved_output = app.storage.user.get(LAST_RESULT_STORAGE_KEY)
        if isinstance(saved_output, dict) and saved_output.get("version") == 1:
            try:
                restored_result = (
                    deserialize_result(saved_output["result"])
                    if saved_output.get("result") is not None else None
                )
                restored_optimization = deserialize_optimization(
                    saved_output.get("optimization")
                )
                show_output(
                    restored_result,
                    restored_optimization,
                    str(saved_output["country"]),
                    str(saved_output["label"]),
                    str(saved_output["test_mode"]),
                    dict(saved_output.get("conditions", {})),
                    saved_output["actual_start"],
                    saved_output["actual_end"],
                )
                status.set_text(
                    f"{saved_output['actual_start']} ~ {saved_output['actual_end']} "
                    "수정종가 기준 · 마지막 계산 결과 복원"
                )
            except Exception:
                app.storage.user.pop(LAST_RESULT_STORAGE_KEY, None)

