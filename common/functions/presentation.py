"""Shared NiceGUI presentation and persistence for every Functions result."""

from __future__ import annotations

import io
from io import StringIO

import numpy as np
import pandas as pd
from nicegui import ui

from common.functions.analytics import ToolResult
from common.snowball.serialization import dataframe_records, json_safe


LABELS = {
    "낙폭": "포트폴리오 낙폭",
    "date": "날짜", "cycle": "사이클", "side": "매수/매도",
    "price": "체결가", "amount": "체결금액", "shares": "수량",
    "note": "거래 사유", "Close": "종가", "High": "고가", "Low": "저가",
    "cash": "현금", "cost_basis": "매입원금", "avg_price": "평균단가",
    "T": "진행 차수(T)", "star_pct": "별%", "portfolio_value": "평가액",
    "initial_capital": "초기 투자금", "final_value": "최종 평가액",
    "profit": "평가손익", "return_pct": "총수익률",
    "cagr_pct": "연환산 수익률", "mdd_pct": "최대 낙폭(MDD)",
    "completed_cycles": "완료 사이클", "max_t": "최대 진행 차수(T)",
    "ending_cash": "최종 현금", "ending_shares": "최종 보유수량",
    "ending_avg": "최종 평균단가", "effective_start_date": "실제 시작일",
    "effective_end_date": "실제 종료일", "initial_amount": "초기 투자금",
    "additional_contributions": "추가 납입금", "total_contributions": "총 투입금",
    "buy_hold_final_value": "장기보유 최종 평가액",
    "buy_hold_return_pct": "장기보유 수익률", "buy_count": "매수 횟수",
    "sell_count": "매도 횟수", "position_changes": "포지션 변경 횟수",
    "confirmation_count": "확인 횟수", "signal_limit_days": "신호 대기일",
    "ending_position": "최종 포지션", "sharpe": "샤프지수",
    "sortino": "소르티노지수", "calmar": "칼마지수", "xirr": "내부수익률(XIRR)",
    "event": "거래 종류", "type": "유형", "contribution": "추가 납입금",
    "scheduled_date": "예정일", "actual_date": "실제 거래일",
    "signal_date": "신호일", "regime": "시장 국면", "turnover": "회전율",
    "fee": "거래비용", "nav": "순자산가치", "Frequency": "측정 주기",
    "Model": "모형", "CAGR": "연환산 수익률", "MaxDD": "최대 낙폭(MDD)",
    "Volatility": "변동성", "Sharpe_rf0": "샤프지수",
    "Sortino": "소르티노지수", "Calmar": "칼마지수",
    "Start": "시작일", "End": "종료일", "action": "거래 내용",
    "rate": "환율", "score": "신호 점수", "total_value": "총 평가액",
    "converted_value": "환전금액", "cash_before": "환전 전 원화",
    "usd_value_before": "환전 전 달러 평가액",
    "fear_and_greed_historical": "종합 공포·탐욕 지수",
    "market_momentum_sp500": "S&P500 시장 모멘텀",
    "market_volatility_vix": "VIX 시장 변동성",
}


def label(value: object) -> str:
    return LABELS.get(str(value), str(value))


def _metric_value(metric: object, value: object) -> str:
    if str(metric) == "ending_position":
        return {"LONG": "투자 중", "STOCK": "투자 중", "CASH": "현금 보유"}.get(
            str(value), str(value)
        )
    return str(value)


def localized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.index.name = label(output.index.name) if output.index.name else output.index.name
    output = output.rename(columns={column: label(column) for column in output.columns})
    if "매수/매도" in output:
        output["매수/매도"] = output["매수/매도"].replace({"BUY": "매수", "SELL": "매도"})
    if "거래 종류" in output:
        output["거래 종류"] = output["거래 종류"].replace({
            "BUY": "매수", "SELL": "매도", "CONTRIBUTION": "추가 납입",
        })
    if "최종 포지션" in output:
        output["최종 포지션"] = output["최종 포지션"].replace({
            "LONG": "투자 중", "STOCK": "투자 중", "CASH": "현금 보유",
        })
    if "시장 국면" in output:
        output["시장 국면"] = output["시장 국면"].replace({
            "reflation": "리플레이션", "goldilocks": "골디락스",
            "stagflation": "스태그플레이션", "slowdown": "경기 둔화", "cash": "현금",
        })
    return output


def _is_drawdown_name(name: object) -> bool:
    raw = str(name).lower().replace(" ", "")
    return "낙폭" in raw or "drawdown" in raw or raw in {"mdd", "maxdd"}


def _performance_series(series: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {name: values for name, values in series.items() if not _is_drawdown_name(name)}


def drawdown_series(series: dict[str, pd.Series]) -> dict[str, pd.Series]:
    explicit = {name: values for name, values in series.items() if _is_drawdown_name(name)}
    if explicit:
        output = {}
        for name, values in explicit.items():
            clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
            if clean.empty:
                continue
            # Native engines store drawdown either as a decimal or already as percent.
            output[label(name)] = clean * 100.0 if clean.abs().max() <= 1.5 else clean
        return output

    excluded = ("투입금", "환율", "원/달러", "적정환율", "종가", "공포", "탐욕")
    output = {}
    for name, values in _performance_series(series).items():
        if any(term in str(name) for term in excluded):
            continue
        clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
        if clean.empty or (clean <= 0).any():
            continue
        output[f"{label(name)} 낙폭"] = (clean / clean.cummax() - 1.0) * 100.0
    return output


def _chart_options(series: dict[str, pd.Series], *, percent_axis: bool = False) -> dict:
    all_indexes: list[str] = []
    normalized: dict[str, pd.Series] = {}
    for name, raw in series.items():
        clean = pd.Series(raw).replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) > 900:
            positions = np.linspace(0, len(clean) - 1, 900).astype(int)
            clean = clean.iloc[np.unique(positions)]
        normalized[name] = clean
        all_indexes.extend(
            str(value.date() if hasattr(value, "date") else value) for value in clean.index
        )
    categories = list(dict.fromkeys(all_indexes))
    output = []
    for name, clean in normalized.items():
        values = {
            str(index.date() if hasattr(index, "date") else index): round(float(value), 5)
            for index, value in clean.items()
        }
        output.append({
            "name": label(name), "type": "line", "showSymbol": False,
            "data": [values.get(item) for item in categories],
            "areaStyle": {"opacity": .08} if percent_axis else None,
        })
    y_axis = {"type": "value", "scale": not percent_axis}
    if percent_axis:
        y_axis.update({"name": "낙폭(%)", "max": 0, "axisLabel": {"formatter": "{value}%"}})
    return {
        "tooltip": {"trigger": "axis"}, "legend": {"type": "scroll", "bottom": 0},
        "grid": {"left": 62, "right": 20, "top": 25, "bottom": 58},
        "xAxis": {"type": "category", "data": categories, "axisLabel": {"hideOverlap": True}},
        "yAxis": y_axis, "dataZoom": [{"type": "inside"}], "series": output,
    }


def serialize_tool_result(result: ToolResult) -> dict:
    return json_safe({
        "title": result.title, "metrics": dict(result.metrics), "note": result.note,
        "series": {
            str(name): {
                "index": [
                    value.isoformat() if hasattr(value, "isoformat") else value
                    for value in values.index
                ],
                "datetime_index": isinstance(values.index, pd.DatetimeIndex),
                "values": [None if pd.isna(value) else float(value) for value in values],
                "name": values.name,
            }
            for name, values in result.series.items()
        },
        "table": result.table.to_json(orient="split", date_format="iso"),
    })


def deserialize_tool_result(payload: dict) -> ToolResult:
    series = {}
    for name, item in payload.get("series", {}).items():
        index = (
            pd.to_datetime(item["index"])
            if item.get("datetime_index") else pd.Index(item["index"])
        )
        series[name] = pd.Series(item["values"], index=index, name=item.get("name"))
    table_raw = str(payload.get("table", ""))
    table = pd.read_json(StringIO(table_raw), orient="split") if table_raw else pd.DataFrame()
    return ToolResult(
        str(payload.get("title", "분석 결과")),
        {str(key): str(value) for key, value in payload.get("metrics", {}).items()},
        series, table, str(payload.get("note", "")),
    )


def render_tool_result(result: ToolResult) -> None:
    ui.label(result.title).classes("section-title")
    if result.metrics:
        with ui.grid().classes("result-metric-grid w-full gap-3"):
            for metric, value in result.metrics.items():
                with ui.card().classes("metric-card"):
                    with ui.row().classes("metric-row items-center justify-between no-wrap"):
                        ui.label(label(metric)).classes("metric-label")
                        ui.label(_metric_value(metric, value)).classes("metric-value")

    performance = _performance_series(result.series)
    drawdowns = drawdown_series(result.series)
    with ui.tabs().classes("dashboard-tabs w-full") as tabs:
        chart_tab = ui.tab("차트", icon="show_chart")
        mdd_tab = ui.tab("MDD", icon="trending_down")
        detail_tab = ui.tab("세부내역", icon="table_rows")
        download_tab = ui.tab("다운로드", icon="download")
    with ui.tab_panels(tabs, value=chart_tab).classes("dashboard-panels w-full"):
        with ui.tab_panel(chart_tab):
            if performance:
                ui.echart(_chart_options(performance)).classes("w-full dashboard-chart")
            else:
                ui.label("이 결과에는 일반 차트 자료가 없습니다.").classes("empty-state")
        with ui.tab_panel(mdd_tab):
            if drawdowns:
                ui.echart(_chart_options(drawdowns, percent_axis=True)).classes(
                    "w-full dashboard-chart"
                )
            else:
                ui.label("이 결과에는 MDD를 계산할 평가액 자료가 없습니다.").classes("empty-state")
        with ui.tab_panel(detail_tab):
            if not result.table.empty:
                frame = localized_frame(result.table).tail(500)
                frame.columns = [str(column) for column in frame.columns]
                frame = frame.reset_index(drop=False)
                frame.columns = [str(column) for column in frame.columns]
                rows = dataframe_records(frame)
                columns = [
                    {"name": column, "label": column, "field": column, "sortable": True}
                    for column in frame.columns
                ]
                ui.table(columns=columns, rows=rows, pagination=20).classes(
                    "w-full account-table"
                )
            else:
                ui.label("표시할 세부내역이 없습니다.").classes("empty-state")
        with ui.tab_panel(download_tab):
            with ui.row().classes("gap-3"):
                if not result.table.empty:
                    csv_data = localized_frame(result.table).to_csv(index=True).encode("utf-8-sig")
                    ui.button(
                        "CSV 다운로드", icon="download",
                        on_click=lambda: ui.download(csv_data, "result.csv"),
                    )
                if result.series or not result.table.empty:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        if result.series:
                            pd.concat(
                                {label(name): values for name, values in result.series.items()}, axis=1,
                            ).to_excel(writer, sheet_name="차트 데이터")
                        if not result.table.empty:
                            localized_frame(result.table).to_excel(
                                writer, sheet_name="세부내역", index=True,
                            )
                    excel_data = output.getvalue()
                    ui.button(
                        "Excel 다운로드", icon="download",
                        on_click=lambda: ui.download(excel_data, "result.xlsx"),
                    )
    if result.note:
        ui.label(result.note).classes("status-text")
