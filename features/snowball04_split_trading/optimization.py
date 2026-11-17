"""Parameter sweep helpers built on the shared split-trading engine."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import pandas as pd

from .engine import SplitTradingConfig, run_split_backtest


CRITERIA = {
    "return": "총수익률 최대",
    "mdd": "MDD 최소",
    "return_mdd": "수익률/MDD 최대",
}


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
    return {
        "최종 평가액": float(result["split_final"]),
        "총수익률": float(result["split_return"]),
        "MDD": float(result["split_max_drawdown"]),
        "현재 낙폭": float(result["split_current_drawdown"]),
        "수익률/MDD": float(result["split_return_mdd"] or 0.0),
        "거래 횟수": int(len(result["trades"])),
        "최종 현금": float(result["final_cash"]),
    }


def _finish(runs: list[dict], criterion: str) -> dict:
    if not runs:
        raise ValueError("계산 가능한 최적화 조합이 없습니다.")
    runs.sort(key=lambda item: _score(item["result"], criterion), reverse=True)
    rows = []
    for rank, item in enumerate(runs, start=1):
        rows.append({"순위": rank, **item["parameters"], **_summary(item["result"])})
    return {
        "criterion": criterion,
        "runs": runs,
        "best": runs[0]["result"],
        "best_parameters": runs[0]["parameters"],
        "table": pd.DataFrame(rows),
    }


def run_ratio_sweep(
    closes: pd.Series,
    base: SplitTradingConfig,
    buy_rates: Iterable[float],
    sell_rates: Iterable[float],
    criterion: str,
) -> dict:
    runs = []
    for buy_rate in buy_rates:
        for sell_rate in sell_rates:
            config = replace(base, buy_drop=float(buy_rate), sell_rise=float(sell_rate))
            result = run_split_backtest(closes, config)
            runs.append({
                "parameters": {
                    "매수 하락률": float(buy_rate),
                    "매도 상승률": float(sell_rate),
                },
                "result": result,
            })
    return _finish(runs, criterion)


def run_tranche_sweep(
    closes: pd.Series,
    base: SplitTradingConfig,
    candidates: Iterable[tuple[str, float]],
    criterion: str,
) -> dict:
    runs = []
    for label, tranche in candidates:
        if tranche <= 0 or tranche > base.initial_capital:
            continue
        config = replace(base, tranche_amount=float(tranche))
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

