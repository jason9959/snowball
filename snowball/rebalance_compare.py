"""Compare a model target allocation with one account's current valuation."""

from __future__ import annotations

import re

import pandas as pd


_WEIGHT_PATTERN = re.compile(r"^(?P<label>.+?)\s+(?P<weight>-?\d+(?:\.\d+)?)%$")


def parse_target_selection(selection: str) -> dict[str, tuple[str, float]]:
    """Parse the mailing selection text into code -> (label, decimal weight)."""
    targets: dict[str, tuple[str, float]] = {}
    for item in str(selection or "").split(","):
        match = _WEIGHT_PATTERN.match(item.strip())
        if match is None:
            continue
        label = match.group("label").strip()
        code = label.split("·", 1)[0].strip()
        if not code:
            continue
        weight = float(match.group("weight")) / 100.0
        previous_label, previous_weight = targets.get(code, (label, 0.0))
        targets[code] = (previous_label, previous_weight + weight)
    if not targets:
        raise ValueError("리밸런싱 목표 비중을 읽을 수 없습니다.")
    total = sum(weight for _, weight in targets.values())
    if abs(total - 1.0) > 0.011:
        raise ValueError(f"리밸런싱 목표 비중 합계가 100%가 아닙니다: {total:.1%}")
    return targets


def build_rebalance_comparison(
    holdings: pd.DataFrame,
    *,
    cash: float,
    total_value: float,
    target_selection: str,
) -> pd.DataFrame:
    """Return current/target weights and KRW trade gaps, including cash."""
    if total_value <= 0:
        raise ValueError("총 평가액이 0보다 커야 리밸런싱 금액을 계산할 수 있습니다.")
    targets = parse_target_selection(target_selection)
    current: dict[str, tuple[str, float]] = {"현금": ("현금", float(cash))}
    if holdings is not None and not holdings.empty:
        required = {"종목코드", "종목명", "평가금액"}
        if not required.issubset(holdings.columns):
            raise ValueError("보유종목 표에 종목코드·종목명·평가금액이 필요합니다.")
        for _, row in holdings.iterrows():
            code = str(row["종목코드"]).strip()
            if not code:
                continue
            label = str(row["종목명"] or code).strip()
            previous_label, previous_value = current.get(code, (label, 0.0))
            current[code] = (previous_label, previous_value + float(row["평가금액"]))

    rows: list[dict[str, object]] = []
    for code in dict.fromkeys((*targets.keys(), *current.keys())):
        target_label, target_weight = targets.get(code, (code, 0.0))
        current_label, current_value = current.get(code, (target_label, 0.0))
        target_value = total_value * target_weight
        gap = target_value - current_value
        tolerance = max(total_value * 0.0005, 1.0)
        action = "유지" if abs(gap) <= tolerance else ("매수" if gap > 0 else "매도")
        rows.append({
            "종목코드": code,
            "종목명": target_label if code in targets else current_label,
            "현재 평가금액": current_value,
            "현재 비중": current_value / total_value,
            "목표 비중": target_weight,
            "목표 평가금액": target_value,
            "차이 금액": gap,
            "조정": action,
        })
    return pd.DataFrame(rows).sort_values(
        ["목표 비중", "현재 비중"], ascending=False, kind="stable",
    ).reset_index(drop=True)
