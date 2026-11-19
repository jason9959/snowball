"""Parameter sweep helpers built on the shared split-trading engine."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import pandas as pd

from .analytics import add_stability_metrics, operational_metrics
from .engine import SplitTradingConfig, run_split_backtest


CRITERIA = {
    "return": "총수익률 최대",
    "mdd": "MDD 최소",
    "return_mdd": "수익률/MDD 최대",
}


def rate_sweep(minimum: float, maximum: float) -> list[float]:
    """Return paired buy/sell rates in 5 percentage-point increments."""
    if minimum > maximum:
        raise ValueError("최소 매수·매도율은 최대 매수·매도율보다 클 수 없습니다.")
    if minimum <= 0 or maximum >= 100:
        raise ValueError("매수·매도율은 0% 초과 100% 미만이어야 합니다.")
    values = []
    value = minimum
    while value <= maximum + 1e-9:
        values.append(round(value / 100, 10))
        value += 5
    maximum_rate = round(maximum / 100, 10)
    if values[-1] != maximum_rate:
        values.append(maximum_rate)
    return values


def _score(result: dict, criterion: str) -> float:
    if criterion == "return":
        return float(result["split_return"])
    if criterion == "mdd":
        return float(result["split_max_drawdown"])
    if criterion == "return_mdd":
        value = result.get("split_return_mdd")
        return float(value) if value is not None else float("-inf")
    raise ValueError("지원하지 않는 최적화 기준입니다.")


def _summary(result: dict) -> dict[str, float | int]:
    operations = operational_metrics(result)
    summary = {
        "최종 평가액": float(result["split_final"]),
        "총수익률": float(result["split_return"]),
        "MDD": float(result["split_max_drawdown"]),
        "현재 낙폭": float(result["split_current_drawdown"]),
        "수익률/MDD": float(result["split_return_mdd"] or 0.0),
        "거래 횟수": int(len(result["trades"])),
        "최종 현금": float(result["final_cash"]),
        "완료 매도 횟수": int(operations["완료 매도 횟수"]),
        "수익 거래 비율": float(operations["수익 거래 비율"]),
        "평균 투자집행률": float(operations["평균 투자집행률"]),
        "평균 현금비중": float(operations["평균 현금비중"]),
        "미청산 보유 차수": int(operations["미청산 보유 차수"]),
    }
    if result.get("funding_mode") == "required":
        summary.update({
            "최소 필요자금": float(result["minimum_required_capital"]),
            "최대 자금 필요일": result["minimum_required_capital_date"],
            "안전 여유 10%": float(result["recommended_capital_10"]),
            "안전 여유 20%": float(result["recommended_capital_20"]),
            "안전 여유 30%": float(result["recommended_capital_30"]),
        })
    return summary


def _finish(runs: list[dict], criterion: str) -> dict:
    if not runs:
        raise ValueError("계산 가능한 최적화 조합이 없습니다.")
    runs.sort(key=lambda item: _score(item["result"], criterion), reverse=True)
    rows = []
    for rank, item in enumerate(runs, start=1):
        rows.append({"순위": rank, **item["parameters"], **_summary(item["result"])})
    table = add_stability_metrics(pd.DataFrame(rows))
    metric_winners = {}
    for metric in ("총수익률", "MDD", "수익률/MDD", "안정구간 점수"):
        if metric not in table:
            continue
        winner = table.loc[table[metric].astype(float).idxmax()]
        metric_winners[metric] = runs[int(winner["순위"]) - 1]["result"]
    return {
        "criterion": criterion,
        "runs": runs,
        "best": runs[0]["result"],
        "best_parameters": runs[0]["parameters"],
        "table": table,
        "metric_winners": metric_winners,
    }


def run_ratio_sweep(
    closes: pd.Series,
    base: SplitTradingConfig,
    rates: Iterable[float],
    criterion: str,
) -> dict:
    runs = []
    for rate in rates:
        config = replace(base, buy_drop=float(rate), sell_rise=float(rate))
        result = run_split_backtest(closes, config)
        runs.append({
            "parameters": {"매수·매도율": float(rate)},
            "result": result,
        })
    return _finish(runs, criterion)


def run_combined_sweep(
    closes: pd.Series,
    base: SplitTradingConfig,
    candidates: Iterable[tuple[str, float, int]],
    rates: Iterable[float],
    criterion: str,
) -> dict:
    runs = []
    for label, tranche, slot_count in candidates:
        if tranche <= 0 or (
            base.funding_mode == "limited" and tranche > base.initial_capital
        ):
            continue
        for rate in rates:
            config = replace(
                base, tranche_amount=float(tranche),
                buy_drop=float(rate), sell_rise=float(rate),
                slot_count=int(slot_count),
            )
            try:
                result = run_split_backtest(closes, config)
            except ValueError as error:
                if "1주 이상" in str(error):
                    continue
                raise
            runs.append({
                "parameters": {
                    "분할 조건": str(label),
                    "회당 매수금액": float(tranche),
                    "동시 보유 차수": int(config.max_slots),
                    "매수·매도율": float(rate),
                },
                "result": result,
            })
    return _finish(runs, criterion)


def run_tranche_sweep(
    closes: pd.Series,
    base: SplitTradingConfig,
    candidates: Iterable[tuple[str, float, int]],
    criterion: str,
) -> dict:
    runs = []
    for label, tranche, slot_count in candidates:
        if tranche <= 0 or (
            base.funding_mode == "limited" and tranche > base.initial_capital
        ):
            continue
        config = replace(
            base, tranche_amount=float(tranche), slot_count=int(slot_count),
        )
        try:
            result = run_split_backtest(closes, config)
        except ValueError as error:
            if "1주 이상" in str(error):
                continue
            raise
        runs.append({
            "parameters": {
                "분할 조건": str(label),
                "회당 매수금액": float(tranche),
                "동시 보유 차수": int(config.max_slots),
            },
            "result": result,
        })
    return _finish(runs, criterion)

