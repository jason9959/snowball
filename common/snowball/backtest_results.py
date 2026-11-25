"""Shared Snowball backtest result presentation for NiceGUI features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from nicegui import ui

from common.snowball.serialization import dataframe_records

from snowball.exports import (
    diagnostic_frames,
    diagnostic_zip,
    market_data_export,
    monthly_return_export,
)


def _percent(value: float | None) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):.2%}"


def _metric_card(label: str, value: str) -> None:
    with ui.card().classes("metric-card w-full"):
        with ui.row().classes("metric-row w-full items-center justify-between no-wrap"):
            ui.label(label).classes("metric-label")
            ui.label(value).classes("metric-value")


def _table_columns(names: list[str]) -> list[dict[str, object]]:
    return [
        {"name": name, "label": name, "field": name, "align": "left", "sortable": True}
        for name in names
    ]


def _result_table(frame: pd.DataFrame, rows_per_page: int = 12) -> None:
    if frame is None or frame.empty:
        ui.label("표시할 자료가 없습니다.").classes("empty-state")
        return
    visible = frame.copy().fillna("")
    for column in visible.columns:
        visible[column] = visible[column].astype(str)
    ui.table(
        columns=_table_columns(list(visible.columns)), rows=dataframe_records(visible),
        pagination={"rowsPerPage": rows_per_page},
    ).props("flat bordered dense").classes("w-full account-table")


def _monthly_result_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    months = frame.copy()
    periods = pd.PeriodIndex(months["적용 월"], freq="M")
    months["연도"] = periods.year
    months["월"] = periods.month
    pivot = months.pivot(index="연도", columns="월", values="수익률 원시값")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [f"{month}월" for month in pivot.columns]
    pivot = pivot.reset_index()
    for column in pivot.columns[1:]:
        pivot[column] = pivot[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
        )
    return pivot


def _monthly_heatmap_options(frame: pd.DataFrame) -> dict:
    """Build a red/white/green monthly-return heatmap for ECharts."""
    if frame.empty:
        return {}
    months = frame.copy()
    periods = pd.PeriodIndex(months["적용 월"], freq="M")
    values = pd.to_numeric(months["수익률 원시값"], errors="coerce") * 100.0
    years = sorted(set(periods.year))
    year_positions = {year: index for index, year in enumerate(years)}
    data = []
    for period, value in zip(periods, values):
        if pd.isna(value):
            continue
        numeric = round(float(value), 2)
        data.append({
            "value": [int(period.month) - 1, year_positions[int(period.year)], numeric],
            "label": {"show": True, "formatter": f"{numeric:.1f}%", "fontSize": 11},
        })
    magnitudes = [abs(item["value"][2]) for item in data]
    bound = max(1.0, float(np.percentile(magnitudes, 90))) if magnitudes else 1.0
    for item in data:
        item["label"]["color"] = "#ffffff" if abs(item["value"][2]) >= bound * 0.62 else "#273142"
    return {
        "tooltip": {"position": "top"},
        "grid": {"left": 52, "right": 18, "top": 18, "bottom": 54},
        "xAxis": {
            "type": "category", "data": [f"{month}월" for month in range(1, 13)],
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category", "data": [str(year) for year in years],
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": -bound, "max": bound, "calculable": True,
            "orient": "horizontal", "left": "center", "bottom": 0,
            "inRange": {"color": ["#d95c5c", "#fff7f7", "#ffffff", "#f3fbf5", "#3f9b65"]},
        },
        "series": [{
            "name": "월 수익률", "type": "heatmap", "data": data,
            "emphasis": {"itemStyle": {"shadowBlur": 8, "shadowColor": "rgba(0,0,0,.25)"}},
        }],
    }


def _monthly_distribution_options(frame: pd.DataFrame, bins: int = 18) -> dict:
    """Build a compact histogram showing monthly-return frequency ranges."""
    values = pd.to_numeric(frame.get("수익률 원시값"), errors="coerce").dropna().to_numpy() * 100.0
    if not len(values):
        return {}
    counts, edges = np.histogram(values, bins=min(bins, max(6, len(values) // 4)))
    labels = [f"{left:.1f}~{right:.1f}%" for left, right in zip(edges[:-1], edges[1:])]
    colors = ["#d95c5c" if (left + right) / 2 < 0 else "#3f9b65" for left, right in zip(edges[:-1], edges[1:])]
    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 48, "right": 18, "top": 24, "bottom": 78},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 38, "fontSize": 10}},
        "yAxis": {"type": "value", "minInterval": 1, "name": "발생 개월 수"},
        "series": [{
            "name": "개월 수", "type": "bar", "data": [
                {"value": int(count), "itemStyle": {"color": color}}
                for count, color in zip(counts, colors)
            ],
        }],
    }


def render_backtest_result(result: dict, bundle) -> None:
    metrics = result["metrics"]
    metric_values = (
        ("누적 수익률", _percent(metrics.get("total_return"))),
        ("연환산 수익률", _percent(metrics.get("cagr"))),
        ("연 변동성", _percent(metrics.get("annual_volatility"))),
        ("최대 낙폭", _percent(metrics.get("max_drawdown"))),
        ("연 수익률 표준편차", _percent(metrics.get("annual_return_std"))),
        ("최악의 해", _percent(metrics.get("worst_year"))),
    )
    ui.label(f"{result['preset_name']} 결과").classes("section-title")
    ui.label(
        f"공통 평가기간 · {pd.Timestamp(result['common_start']).date()} ~ "
        f"{pd.Timestamp(result['common_end']).date()} · "
        f"연간 턴오버 {float(result.get('annual_turnover', 0)):.2%}"
    ).classes("text-sm text-grey-6")
    with ui.grid().classes("result-metric-grid w-full gap-3"):
        for label, value in metric_values:
            _metric_card(label, value)
    ui.button(
        "비교 진단자료 ZIP 다운로드", icon="folder_zip",
        on_click=lambda: ui.download.content(
            diagnostic_zip(result),
            filename=(
                f"snowball_diagnostics_{pd.Timestamp(result['common_start']).date()}_"
                f"{pd.Timestamp(result['common_end']).date()}.zip"
            ),
            media_type="application/zip",
        ),
    ).props("outline no-caps").classes("diagnostic-download")

    nav = result["nav"].dropna().astype(float)
    monthly_export = monthly_return_export(nav)
    chart = pd.DataFrame({"포트폴리오": nav / nav.iloc[0] - 1.0})
    benchmark = result.get("benchmark_ticker", "QQQ")
    if benchmark in bundle.adjusted_close:
        comparison = bundle.adjusted_close[benchmark].reindex(nav.index).ffill().dropna()
        if not comparison.empty:
            comparison = comparison / comparison.iloc[0] - 1.0
            chart["국내 나스닥100" if benchmark.endswith(".KS") else "QQQ"] = comparison
    dates = [value.strftime("%Y-%m-%d") for value in chart.index]
    series = [
        {
            "name": column, "type": "line", "showSymbol": False,
            "data": (chart[column] * 100).round(4).tolist(),
            "lineStyle": {"width": 3 if column == "포트폴리오" else 2},
        }
        for column in chart.columns
    ]
    with ui.tabs().classes("w-full dashboard-tabs") as tabs:
        returns_tab = ui.tab("수익률 차트", icon="show_chart")
        drawdown_tab = ui.tab("손실폭", icon="trending_down")
        monthly_tab = ui.tab("월별 수익률", icon="calendar_month")
        history_tab = ui.tab("리밸런싱", icon="event_repeat")
        decisions_tab = ui.tab("판단 원시값", icon="fact_check")
        snapshot_tab = ui.tab("데이터", icon="database")
        settings_tab = ui.tab("적용 설정", icon="settings")
    content = ui.column().classes("w-full dashboard-panels result-tab-content")

    def render_returns() -> None:
        with content:
            ui.echart({
                "tooltip": {"trigger": "axis"},
                "legend": {"bottom": 0, "data": list(chart.columns)},
                "grid": {"left": 58, "right": 24, "top": 24, "bottom": 58},
                "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
                "yAxis": {"type": "value", "scale": True, "axisLabel": {"formatter": "{value}%"}},
                "dataZoom": [{"type": "inside"}], "series": series,
            }).classes("dashboard-chart w-full")

    def render_drawdown() -> None:
        with content:
            drawdown = result["drawdown"].reindex(nav.index).fillna(0.0) * 100
            ui.echart({
                "tooltip": {"trigger": "axis"},
                "grid": {"left": 54, "right": 24, "top": 24, "bottom": 48},
                "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
                "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}%"}},
                "dataZoom": [{"type": "inside"}],
                "series": [{
                    "name": "손실폭", "type": "line", "showSymbol": False,
                    "data": drawdown.round(3).tolist(), "areaStyle": {"opacity": .18},
                }],
            }).classes("dashboard-chart w-full")

    def render_monthly() -> None:
        with content:
            ui.label("월별 수익률").classes("section-title")
            ui.label("상승 월은 초록색, 하락 월은 빨간색이며 색이 진할수록 변동 폭이 큽니다.").classes("status-text")
            ui.echart(_monthly_heatmap_options(monthly_export)).classes("monthly-heatmap w-full")
            ui.label("월별 수익률 분포").classes("section-title")
            ui.echart(_monthly_distribution_options(monthly_export)).classes("monthly-distribution w-full")
            with ui.expansion("월별 수익률 표로 보기", icon="table_rows").classes("strategy-expansion w-full"):
                _result_table(_monthly_result_table(monthly_export), 20)
            ui.button(
                "월별 수익률 CSV", icon="download",
                on_click=lambda: ui.download.content(
                    monthly_export.to_csv(index=False).encode("utf-8-sig"),
                    filename="monthly_returns.csv", media_type="text/csv",
                ),
            ).props("outline no-caps")

    def render_history() -> None:
        with content:
            history = result["history"].copy()
            display = [
                column for column in (
                    "적용일", "판단 기준일", "선택된 종목", "턴오버", "거래비용", "평가금액"
                ) if column in history
            ]
            table = history[display].sort_values("적용일", ascending=False).copy()
            _result_table(table)
            ui.button(
                "리밸런싱 CSV", icon="download",
                on_click=lambda: ui.download.content(
                    history.to_csv(index=False).encode("utf-8-sig"),
                    filename="rebalancing.csv", media_type="text/csv",
                ),
            ).props("outline no-caps")

    def render_decisions() -> None:
        with content:
            keys = list(result.get("candidate_snapshots", {}))
            if not keys:
                ui.label("판단 원시값이 없습니다.").classes("empty-state")
            else:
                decision_date = ui.select(
                    keys, value=keys[-1], label="리밸런싱 적용일",
                ).props("outlined options-dense").classes("decision-select")
                decision_area = ui.column().classes("w-full gap-4")

                def render_decision() -> None:
                    decision_area.clear()
                    selected = str(decision_date.value)
                    with decision_area:
                        ui.label("자산 선택 계산값").classes("font-bold")
                        _result_table(result["candidate_snapshots"].get(selected, pd.DataFrame()))
                        canary = result.get("canary_audits", {}).get(selected, pd.DataFrame())
                        if not canary.empty:
                            ui.label("카나리아 원시 신호").classes("font-bold")
                            _result_table(canary)
                        market = result.get("market_audits", {}).get(selected, pd.DataFrame())
                        if not market.empty:
                            ui.label("시장 데이터 원시 신호").classes("font-bold")
                            _result_table(market)

                decision_date.on_value_change(lambda _: render_decision())
                render_decision()
                def decision_csv() -> bytes:
                    exports = diagnostic_frames(result)
                    raw = pd.concat([
                        exports["03_asset_signals.csv"], exports["04_canary_signals.csv"],
                        exports["05_market_signals.csv"],
                    ], ignore_index=True, sort=False)
                    return raw.to_csv(index=False).encode("utf-8-sig")

                ui.button(
                    "전체 판단 원시값 CSV", icon="download",
                    on_click=lambda: ui.download.content(
                        decision_csv(),
                        filename="decision_raw_values.csv", media_type="text/csv",
                    ),
                ).props("outline no-caps")

    def render_snapshot() -> None:
        with content:
            metadata = result.get("data_snapshot", {})
            ui.label("이번 실행에 사용된 데이터 버전").classes("font-bold")
            ui.code(json.dumps(metadata, ensure_ascii=False, indent=2, default=str)).classes(
                "snapshot-code w-full"
            )
            market_snapshot = market_data_export(result)
            if not market_snapshot.empty:
                _result_table(market_snapshot, 15)
                ui.button(
                    "시장 데이터 CSV", icon="download",
                    on_click=lambda: ui.download.content(
                        market_snapshot.to_csv(index=False).encode("utf-8-sig"),
                        filename="market_data_snapshot.csv", media_type="text/csv",
                    ),
                ).props("outline no-caps")
            with ui.row().classes("w-full gap-3"):
                ui.button(
                    "수정종가 CSV", icon="download",
                    on_click=lambda: ui.download.content(
                        bundle.adjusted_close.rename_axis("Date").to_csv().encode("utf-8-sig"),
                        filename="adjusted_close.csv", media_type="text/csv",
                    ),
                ).props("outline no-caps")
                ui.button(
                    "수정시가 CSV", icon="download",
                    on_click=lambda: ui.download.content(
                        bundle.adjusted_open.rename_axis("Date").to_csv().encode("utf-8-sig"),
                        filename="adjusted_open.csv", media_type="text/csv",
                    ),
                ).props("outline no-caps")

    def render_settings() -> None:
        with content:
            config = dict(result.get("engine_config", {}))
            config.pop("execution_prices", None)
            config.pop("_market_data", None)
            ui.code(json.dumps(config, ensure_ascii=False, indent=2, default=str)).classes(
                "snapshot-code w-full"
            )

    renderers = {
        "returns": render_returns,
        "drawdown": render_drawdown,
        "monthly": render_monthly,
        "history": render_history,
        "decisions": render_decisions,
        "snapshot": render_snapshot,
        "settings": render_settings,
    }

    def render_selected(name: str) -> None:
        content.clear()
        renderers[name]()

    for tab, name in (
        (returns_tab, "returns"), (drawdown_tab, "drawdown"),
        (monthly_tab, "monthly"), (history_tab, "history"),
        (decisions_tab, "decisions"), (snapshot_tab, "snapshot"),
        (settings_tab, "settings"),
    ):
        tab.on("click", lambda _, selected=name: render_selected(selected))
    render_selected("returns")
