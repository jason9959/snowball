"""Result views for feature 04."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common.metrics import cashflow_xirr, mdd_recovery_days


COLORS = {"포트폴리오": "#F57C00", "벤치마크": "#5965D8", "동일 비중": "#2E9D68"}


def _format_percent(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}%}"


def _metric_summary(nav: pd.Series, annual_turnover: float | None = None) -> dict[str, Any]:
    nav = nav.dropna().astype(float)
    returns = nav.pct_change().dropna()
    if len(nav) < 2:
        return {}
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1 / 365.25)
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else np.nan
    drawdown = nav / nav.cummax() - 1.0
    mdd = float(drawdown.min())
    sharpe = cagr / volatility if volatility and volatility > 0 else np.nan
    downside = float(np.sqrt(np.mean(np.square(np.minimum(returns, 0.0)))) * np.sqrt(252)) if len(returns) else np.nan
    sortino = cagr / downside if downside and downside > 0 else np.nan
    ulcer = float(np.sqrt(np.mean(np.square(drawdown))))
    upi = cagr / ulcer if years >= 1 and ulcer > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    recovery_days = mdd_recovery_days(nav)
    monthly = nav.resample("ME").last().pct_change().dropna()
    positive = int((monthly > 0).sum())
    current_month = float(nav.loc[nav.index >= nav.index[-1].to_period("M").start_time].iloc[-1] / nav.loc[nav.index >= nav.index[-1].to_period("M").start_time].iloc[0] - 1.0)
    current_year_values = nav.loc[nav.index.year == nav.index[-1].year]
    current_year = float(current_year_values.iloc[-1] / current_year_values.iloc[0] - 1.0)
    result = {
        "기간 수익률": total_return,
        "연환산 수익률": cagr,
        "이번 달 수익률": current_month,
        "올해 수익률": current_year,
        "월 최고 수익률": float(monthly.max()) if len(monthly) else np.nan,
        "월 최저 수익률": float(monthly.min()) if len(monthly) else np.nan,
        "수익 월 비중": f"{positive} / {len(monthly)} ({positive / len(monthly):.1%})" if len(monthly) else "-",
        "연 변동성": volatility,
        "최대 낙폭 (MDD)": mdd,
        "MDD 시점": drawdown.idxmin().strftime("%Y-%m") if len(drawdown) else "-",
        "샤프 지수": sharpe,
        "소르티노 지수": sortino,
        "UPI 지수": upi,
        "Calmar 지수": calmar,
        "MDD 회복기간": "미회복" if not np.isfinite(recovery_days) else f"{recovery_days:.0f}일",
        "연간 턴오버": annual_turnover if annual_turnover is not None else np.nan,
    }
    for year_count in (1, 3, 5):
        cutoff = nav.index[-1] - pd.DateOffset(years=year_count)
        window = nav.loc[nav.index >= cutoff]
        enough = nav.index[0] <= cutoff and len(window) > 1
        result[f"{year_count}년 수익률"] = float(window.iloc[-1] / window.iloc[0] - 1.0) if enough else np.nan
        window_returns = window.pct_change().dropna()
        result[f"{year_count}년 표준편차"] = (
            float(window_returns.std(ddof=1) * np.sqrt(252)) if enough and len(window_returns) > 1 else np.nan
        )
    return result


def _summary_table(
    series_map: dict[str, pd.Series], turnover: float, portfolio_extra: dict[str, Any] | None = None
) -> pd.DataFrame:
    metrics = {name: _metric_summary(series, turnover if name == "포트폴리오" else None) for name, series in series_map.items()}
    if portfolio_extra:
        metrics["포트폴리오"].update(portfolio_extra)
    order = [
        "기간 수익률", "연환산 수익률", "이번 달 수익률", "올해 수익률",
        "월 최고 수익률", "월 최저 수익률", "수익 월 비중", "연 변동성",
        "최대 낙폭 (MDD)", "MDD 시점", "MDD 회복기간",
        "1년 수익률", "3년 수익률", "5년 수익률",
        "1년 표준편차", "3년 표준편차", "5년 표준편차",
        "샤프 지수", "소르티노 지수", "Calmar 지수", "UPI 지수",
        "연간 턴오버", "적립식 XIRR",
    ]
    rows = []
    percent_rows = {
        "기간 수익률", "연환산 수익률", "이번 달 수익률", "올해 수익률",
        "월 최고 수익률", "월 최저 수익률", "연 변동성", "최대 낙폭 (MDD)", "연간 턴오버",
        "1년 수익률", "3년 수익률", "5년 수익률",
        "1년 표준편차", "3년 표준편차", "5년 표준편차", "적립식 XIRR",
    }
    for metric in order:
        row = {"지표": metric}
        for name, values in metrics.items():
            value = values.get(metric, np.nan)
            if metric in percent_rows:
                row[name] = _format_percent(value)
            elif isinstance(value, (float, np.floating)):
                row[name] = "-" if not np.isfinite(value) else f"{value:.2f}"
            else:
                row[name] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _line_figure(frame: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    for column in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=column,
                line={"width": 2.4, "color": COLORS.get(column)},
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title=y_title,
        hovermode="x unified",
        height=430,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def _drawdown_episodes(nav: pd.Series) -> pd.DataFrame:
    drawdown = nav / nav.cummax() - 1.0
    episodes = []
    in_episode = False
    start = trough = None
    trough_value = 0.0
    for date, value in drawdown.items():
        if value < 0 and not in_episode:
            in_episode = True
            prior = nav.loc[:date].iloc[:-1]
            start = prior.idxmax() if len(prior) else date
            trough, trough_value = date, float(value)
        elif in_episode and value < trough_value:
            trough, trough_value = date, float(value)
        if in_episode and value >= -1e-12:
            episodes.append(
                {
                    "시작": start.date(),
                    "저점": trough.date(),
                    "회복시점": date.date(),
                    "회복기간(일)": int((date - trough).days),
                    "손실기간(일)": int((date - start).days),
                    "드로우다운": trough_value,
                }
            )
            in_episode = False
    if in_episode:
        episodes.append(
            {
                "시작": start.date(),
                "저점": trough.date(),
                "회복시점": "진행 중",
                "회복기간(일)": "진행 중",
                "손실기간(일)": int((nav.index[-1] - start).days),
                "드로우다운": trough_value,
            }
        )
    result = pd.DataFrame(episodes)
    if not result.empty:
        result = result.sort_values("드로우다운").head(10).reset_index(drop=True)
        result.insert(0, "순위", np.arange(1, len(result) + 1))
        result["드로우다운"] = result["드로우다운"].map(lambda value: f"{value:.1%}")
    return result


def _stress_table(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    benchmark = series_map["벤치마크"].dropna()
    drawdown = benchmark / benchmark.cummax() - 1.0
    rows = []
    in_episode = False
    start = trough = None
    trough_value = 0.0
    for date, value in drawdown.items():
        if value < 0 and not in_episode:
            in_episode = True
            prior = benchmark.loc[:date].iloc[:-1]
            start = prior.idxmax() if len(prior) else date
            trough, trough_value = date, float(value)
        elif in_episode and value < trough_value:
            trough, trough_value = date, float(value)
        if in_episode and value >= -1e-12:
            if trough_value <= -0.10:
                row = {"시작일": start.date(), "저점일": trough.date(), "벤치마크 낙폭": trough_value}
                for name in ("포트폴리오", "동일 비중"):
                    series = series_map[name]
                    row[f"{name} 수익률"] = float(series.asof(trough) / series.asof(start) - 1.0)
                rows.append(row)
            in_episode = False
    if in_episode and trough_value <= -0.10:
        row = {
            "시작일": start.date(),
            "저점일": trough.date(),
            "벤치마크 낙폭": trough_value,
        }
        for name in ("포트폴리오", "동일 비중"):
            series = series_map[name]
            row[f"{name} 수익률"] = float(series.asof(trough) / series.asof(start) - 1.0)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("벤치마크 낙폭").head(5)
        for column in ["벤치마크 낙폭", "포트폴리오 수익률", "동일 비중 수익률"]:
            result[column] = result[column].map(lambda value: f"{value:.1%}")
    return result


def render(result: dict[str, Any], config: dict[str, Any]) -> None:
    st.title("🔄 동적 포트폴리오 백테스트 결과")
    st.caption(
        f"평가 기간: {result['common_start']:%Y-%m-%d} ~ {result['common_end']:%Y-%m-%d} · "
        f"기준통화: {config['currency']} · 리밸런싱: {config['rebalance_frequency']}"
    )
    series_map = {
        "포트폴리오": result["nav"],
        "벤치마크": result["benchmark_nav"],
        "동일 비중": result["equal_weight_nav"],
    }
    aligned = pd.concat(series_map, axis=1).dropna()
    portfolio_extra: dict[str, Any] = {}
    if float(config.get("contribution_amount", 0.0)) > 0:
        portfolio_extra["적립식 XIRR"] = cashflow_xirr(
            result["values"].index,
            result["flows"],
            float(result["values"].iloc[-1]),
            float(config["initial_investment"]),
        )
    tabs = st.tabs(["수익률 차트", "손실폭", "포트폴리오 요약", "자산 세부내역", "리밸런싱 추이"])

    with tabs[0]:
        summary = _metric_summary(result["nav"], result["annual_turnover"])
        cols = st.columns(4)
        cols[0].metric("최종 평가금액", f"{result['values'].iloc[-1]:,.0f} {config['currency']}")
        cols[1].metric("총 투입금", f"{result['invested'].iloc[-1]:,.0f} {config['currency']}")
        cols[2].metric("기간 수익률", _format_percent(summary.get("기간 수익률")))
        cols[3].metric("연환산 수익률", _format_percent(summary.get("연환산 수익률")))
        st.plotly_chart(_line_figure(aligned * 100.0, "수익률 비교", "성과지수 (시작=100)"), use_container_width=True)

        monthly = aligned.resample("ME").last().pct_change().dropna()
        annual = aligned.resample("YE").last().pct_change().dropna()
        st.subheader("연도별 수익률")
        annual_show = annual.copy()
        annual_show.index = annual_show.index.year
        st.dataframe(annual_show.style.format("{:.1%}"), use_container_width=True)
        st.subheader("월별 수익률")
        monthly_show = monthly[["포트폴리오"]].copy()
        monthly_show["연도"] = monthly_show.index.year
        monthly_show["월"] = monthly_show.index.month
        pivot = monthly_show.pivot(index="연도", columns="월", values="포트폴리오")
        st.dataframe(pivot.style.format("{:.1%}").background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
        st.subheader("월별 수익률 분포")
        distribution = px.histogram(monthly, x="포트폴리오", nbins=24, labels={"포트폴리오": "월 수익률"})
        distribution.add_vline(x=0, line_dash="dash", line_color="#D04444")
        distribution.update_layout(height=360, yaxis_title="빈도", xaxis_tickformat=".1%")
        st.plotly_chart(distribution, use_container_width=True)

    with tabs[1]:
        drawdowns = aligned / aligned.cummax() - 1.0
        st.plotly_chart(_line_figure(drawdowns * 100.0, "최대 손실폭", "고점 대비 하락률 (%)"), use_container_width=True)
        st.subheader("폭락 시장 포트폴리오 성과")
        stress = _stress_table(series_map)
        if stress.empty:
            st.info("해당되는 기간이 없습니다.")
        else:
            st.dataframe(stress, hide_index=True, use_container_width=True)
        st.subheader("포트폴리오 드로우다운")
        episodes = _drawdown_episodes(result["nav"])
        if episodes.empty:
            st.info("표시할 손실 구간이 없습니다.")
        else:
            st.dataframe(episodes, hide_index=True, use_container_width=True)

    with tabs[2]:
        st.subheader("주요 지표 비교")
        st.dataframe(
            _summary_table(series_map, result["annual_turnover"], portfolio_extra),
            hide_index=True,
            use_container_width=True,
        )

    with tabs[3]:
        st.subheader("현재 목표 비중")
        weights = pd.DataFrame(
            [{"자산": ticker, "목표 비중": weight} for ticker, weight in result["final_weights"].items() if weight > 0]
        )
        if not weights.empty:
            st.dataframe(weights.style.format({"목표 비중": "{:.1%}"}), hide_index=True, use_container_width=True)
        st.subheader("자산별 수익률")
        asset_nav = result["asset_nav"]
        st.plotly_chart(_line_figure(asset_nav * 100.0, "자산별 성과", "성과지수 (시작=100)"), use_container_width=True)
        st.subheader("자산별 상관관계")
        correlation = asset_nav.pct_change().dropna().corr()
        st.dataframe(correlation.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1), use_container_width=True)

    with tabs[4]:
        st.subheader("전체 기간 리밸런싱 이력")
        history = result["history"].copy()
        if history.empty:
            st.info("표시할 리밸런싱 이력이 없습니다.")
        else:
            display_history = history.copy()
            for column in [
                "공격자산 비중", "방어자산 비중", "현금 비중", "턴오버",
                "카나리아 전환율", "카나리아 전환 비중",
                "카나리아 적용 전 공격자산 비중", "카나리아 적용 후 공격자산 비중",
                "시장 전환 비중",
            ]:
                if column not in display_history:
                    continue
                display_history[column] = display_history[column].map(lambda value: f"{value:.1%}")
            display_history["거래비용"] = display_history["거래비용"].map(lambda value: f"{value:,.0f}")
            st.dataframe(display_history.iloc[::-1], hide_index=True, use_container_width=True, height=360)
            selected_date = st.selectbox("계산 과정을 확인할 적용일", list(result["candidate_snapshots"].keys())[::-1])
            selected_row = history.loc[history["적용일"].astype(str) == selected_date]
            if not selected_row.empty:
                row = selected_row.iloc[0]
                st.caption(
                    f"판단 기준일 {row['판단 기준일']} · 예정일 {row['예정일']} · 실제 적용일 {row['적용일']}"
                )
            detail = result["candidate_snapshots"][selected_date].copy()
            for column in ["상대 계산값", "절대 계산값", "BIL 기준값", "최종 목표 비중"]:
                if column in detail:
                    detail[column] = pd.to_numeric(detail[column], errors="coerce")
            for column in ["초기 배분 비중", "최종 목표 비중"]:
                if column in detail:
                    detail[column] = detail[column].map(
                        lambda value: "-" if pd.isna(value) else f"{float(value):.1%}"
                    )
            st.dataframe(detail, hide_index=True, use_container_width=True)
            canary_audit = result.get("canary_audits", {}).get(selected_date)
            if canary_audit is not None and not canary_audit.empty:
                st.subheader("카나리아 위험 회피 상세")
                canary_show = canary_audit.copy()
                if "모멘텀 값" in canary_show:
                    canary_show["모멘텀 값"] = canary_show["모멘텀 값"].map(
                        lambda value: "-" if pd.isna(value) else f"{float(value):.2%}"
                    )
                st.dataframe(canary_show, hide_index=True, use_container_width=True)
            market_audit = result.get("market_audits", {}).get(selected_date)
            if market_audit is not None and not market_audit.empty:
                st.subheader("시장 데이터 위험 회피 상세")
                market_show = market_audit.copy()
                if "원시값" in market_show:
                    market_show["원시값"] = market_show["원시값"].map(
                        lambda value: "-" if pd.isna(value) else f"{float(value):.2f}"
                    )
                st.dataframe(market_show, hide_index=True, use_container_width=True)
            st.subheader("최종 배분 검증")
            total = sum(result["final_weights"].values())
            st.success(f"공격자산·방어자산·현금 비중 합계: {total:.1%}")


def download_summary_csv(result: dict[str, Any]) -> bytes:
    history = result["history"].copy()
    return history.to_csv(index=False).encode("utf-8-sig")
