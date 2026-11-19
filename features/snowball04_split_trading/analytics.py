"""Operational and robustness analytics for split-trading results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _return_and_mdd(nav: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(nav, errors="coerce").dropna().astype(float)
    if values.empty:
        return 0.0, 0.0
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    mdd = float((values / values.cummax() - 1.0).min())
    return total_return, mdd


def operational_metrics(result: dict) -> dict[str, float | int]:
    trades = result["trades"]
    sells = trades[trades["매수/매도"] == "매도"]
    buys = trades[trades["매수/매도"] == "매수"]
    holding_days = []
    if not sells.empty:
        buy_dates = {
            row["거래 ID"]: pd.Timestamp(row["거래일"])
            for row in buys.to_dict("records")
        }
        holding_days = [
            (pd.Timestamp(row["거래일"]) - buy_dates[row["거래 ID"]]).days
            for row in sells.to_dict("records") if row["거래 ID"] in buy_dates
        ]
    nav = result["split_nav"].replace(0, np.nan)
    cash_ratio = result.get("cash_nav", pd.Series(0.0, index=nav.index)) / nav
    invested_ratio = result.get("holdings_nav", nav) / nav
    position_count = result.get("position_count", pd.Series(1, index=nav.index))
    return {
        "거래 횟수": int(len(trades)),
        "매수 횟수": int(len(buys)),
        "매도 횟수": int(len(sells)),
        "완료 매도 횟수": int(len(sells)),
        "수익 거래 비율": float((sells["실현손익"] > 0).mean()) if len(sells) else 0.0,
        "평균 보유기간": float(np.mean(holding_days)) if holding_days else 0.0,
        "평균 투자집행률": float(invested_ratio.mean()),
        "평균 현금비중": float(cash_ratio.mean()),
        "최대 현금비중": float(cash_ratio.max()),
        "최대 동시 보유 차수": int(position_count.max()),
        "미청산 보유 차수": int(result.get("open_positions", 0)),
        "남은 매수 가능 횟수": int(result.get("remaining_buy_slots", 0)),
        "현재 보유분 매입원금": float(result.get("open_cost_basis", 0.0)),
        "현재 보유분 평가금": float(result.get("final_holdings_value", 0.0)),
        "현금자산": float(result.get("final_cash", 0.0)),
        "총 거래비용": float(result.get("total_trading_cost", 0.0)),
    }


def annual_performance(result: dict) -> pd.DataFrame:
    nav = result["split_nav"]
    rows = []
    for year, values in nav.groupby(nav.index.year):
        total_return, mdd = _return_and_mdd(values)
        trades = result["trades"]
        trade_dates = pd.to_datetime(trades["거래일"])
        year_trades = int((trade_dates.dt.year == year).sum()) if len(trades) else 0
        cash = result.get("cash_nav", pd.Series(0.0, index=nav.index)).reindex(values.index)
        rows.append({
            "연도": int(year), "수익률": total_return, "MDD": mdd,
            "거래 횟수": year_trades,
            "평균 현금비중": float((cash / values.replace(0, np.nan)).mean()),
        })
    return pd.DataFrame(rows)


def period_validation(result: dict) -> pd.DataFrame:
    nav = result["split_nav"]
    midpoint = max(1, len(nav) // 2)
    periods: list[tuple[str, pd.Series]] = [
        ("전반기", nav.iloc[:midpoint]), ("후반기", nav.iloc[midpoint - 1:]),
    ]
    end = nav.index[-1]
    for years in (1, 3, 5):
        start = end - pd.DateOffset(years=years)
        recent = nav.loc[nav.index >= start]
        if len(recent) >= 2:
            periods.append((f"최근 {years}년", recent))
    rows = []
    for label, values in periods:
        total_return, mdd = _return_and_mdd(values)
        rows.append({"구간": label, "수익률": total_return, "MDD": mdd})

    strategy_monthly = nav.resample("ME").last().pct_change().dropna()
    benchmark_monthly = result["full_nav"].resample("ME").last().pct_change().dropna()
    aligned = pd.concat([strategy_monthly, benchmark_monthly], axis=1).dropna()
    aligned.columns = ["strategy", "benchmark"]
    for label, mask in (
        ("상승장 월", aligned["benchmark"] >= 0),
        ("하락장 월", aligned["benchmark"] < 0),
    ):
        values = aligned.loc[mask, "strategy"]
        if len(values):
            rows.append({
                "구간": label,
                "수익률": float((1.0 + values).prod() - 1.0),
                "MDD": float(values.min()),
            })
    return pd.DataFrame(rows)


def add_stability_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Score each count/rate cell together with its immediate neighbors."""
    table = frame.copy()
    if not {"동시 보유 차수", "매수·매도율"}.issubset(table.columns):
        return table
    return_rank = table["총수익률"].rank(pct=True)
    mdd_rank = table["MDD"].rank(pct=True)
    efficiency_rank = table["수익률/MDD"].rank(pct=True)
    table["개별 종합점수"] = (return_rank * .4 + mdd_rank * .3 + efficiency_rank * .3) * 100
    scores, warnings = [], []
    for _, row in table.iterrows():
        count = int(row["동시 보유 차수"])
        rate = float(row["매수·매도율"])
        neighbor = table[
            table["동시 보유 차수"].between(count - 1, count + 1)
            & (table["매수·매도율"] - rate).abs().le(.0500001)
        ]
        neighbor_score = float(neighbor["개별 종합점수"].mean())
        scores.append(neighbor_score)
        warnings.append(
            "과최적화 주의"
            if float(row["개별 종합점수"]) - neighbor_score >= 20 else "안정"
        )
    table["안정구간 점수"] = scores
    table["안정성 판정"] = warnings
    return table


def execution_warnings(result: dict) -> list[str]:
    metrics = operational_metrics(result)
    warnings = []
    if metrics["완료 매도 횟수"] < 3:
        warnings.append("완료된 매도가 3회 미만이라 결과 신뢰도가 낮습니다.")
    if metrics["평균 투자집행률"] < .5:
        warnings.append("평균 투자집행률이 50% 미만이라 낮은 MDD가 현금 보유의 영향일 수 있습니다.")
    if metrics["미청산 보유 차수"] > 1:
        warnings.append(f"종료일 현재 {metrics['미청산 보유 차수']}개 차수가 미청산 상태입니다.")
    annual = annual_performance(result)
    if len(annual) >= 2:
        positive = annual[annual["수익률"] > 0]["수익률"]
        if len(positive) and positive.max() > max(.01, positive.sum() * .65):
            warnings.append("전체 양(+)의 연간 성과가 특정 연도에 크게 의존합니다.")
    if result["split_return"] <= 0:
        warnings.append("전체 기간 수익률이 0% 이하입니다.")
    return warnings or ["현재 기준에서 뚜렷한 실행 가능성 경고가 없습니다."]
