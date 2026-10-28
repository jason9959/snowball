"""Canonical, validated research presets.

UI widgets must be derived from these objects. They are not allowed to own a
second, partially overlapping copy of a strategy definition.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .core.engine import MomentumConfig, normalize_market_signals


@dataclass(frozen=True)
class MomentumSpec:
    kind: str = "수익률"
    months: int = 12
    score_type: str = "1-3-6-12개월 단순"

    def build(self) -> MomentumConfig:
        if self.kind not in {"수익률", "이동평균", "모멘텀 스코어"}:
            raise ValueError(f"지원하지 않는 모멘텀 종류: {self.kind}")
        if self.months < 1:
            raise ValueError("모멘텀 기간은 1개월 이상이어야 합니다.")
        return MomentumConfig(kind=self.kind, months=self.months, score_type=self.score_type)


@dataclass(frozen=True)
class SpreadRule:
    lag_months: int
    active_months: int
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        if self.lag_months < 0 or self.active_months < 1:
            raise ValueError("금리차 반영 시점과 기간이 올바르지 않습니다.")
        return {"lag": self.lag_months, "active": self.active_months, "threshold": self.threshold}


@dataclass(frozen=True)
class PortfolioPreset:
    name: str
    attack_tickers: tuple[str, ...]
    defense_tickers: tuple[str, ...]
    attack_mode: str
    attack_select_count: int
    attack_relative: MomentumSpec
    attack_absolute: MomentumSpec
    defense_mode: str
    defense_select_count: int
    defense_relative: MomentumSpec
    defense_absolute: MomentumSpec
    defense_same_momentum: bool = False
    include_cash: bool = False
    weight_mode: str = "고정 범위"
    fixed_defense_weight: float = 0.0
    negative_threshold: int = 1
    negative_same_momentum: bool = True
    positive_same_momentum: bool = True
    canary_tickers: tuple[str, ...] = ()
    canary_apply_mode: str = "방어형"
    canary_direction: str = "마이너스(-)"
    canary_momentum_type: str = "절대 모멘텀"
    canary_same_momentum: bool = False
    canary_momentum: MomentumSpec = field(default_factory=lambda: MomentumSpec("이동평균", 12))
    # Number of decimal places retained after converting the canary signal to
    # percentage points. ``None`` restores the original raw-value decision.
    canary_decision_percent_digits: int | None = None
    market_combine: str = "OR"
    market_signals: tuple[str, ...] = ()
    spread_10y2y: SpreadRule | None = None
    spread_10y3m: SpreadRule | None = None
    dividend_threshold: float = 1.6
    attack_use_bil: bool = False
    defense_use_bil: bool = False
    cash_replacement_ticker: str | None = None

    def validate(self) -> None:
        if not self.name.strip() or not self.attack_tickers:
            raise ValueError("프리셋 이름과 공격자산은 필수입니다.")
        if len(set(self.attack_tickers)) != len(self.attack_tickers):
            raise ValueError(f"{self.name}: 중복 공격자산이 있습니다.")
        if not 1 <= self.attack_select_count <= len(self.attack_tickers):
            raise ValueError(f"{self.name}: 공격자산 선택 수가 후보군 범위를 벗어났습니다.")
        if self.defense_tickers and not 1 <= self.defense_select_count <= len(self.defense_tickers):
            raise ValueError(f"{self.name}: 방어자산 선택 수가 후보군 범위를 벗어났습니다.")
        if not 0.0 <= self.fixed_defense_weight <= 1.0:
            raise ValueError("고정 방어 비중은 0~1 범위여야 합니다.")
        if self.canary_decision_percent_digits is not None and not 0 <= self.canary_decision_percent_digits <= 6:
            raise ValueError("카나리아 판정 반올림 자릿수는 0~6 또는 None이어야 합니다.")
        if self.weight_mode == "마이너스 절대 모멘텀 수" and not 1 <= self.negative_threshold <= len(self.attack_tickers):
            raise ValueError(f"{self.name}: 마이너스 절대 모멘텀 기준 수가 잘못됐습니다.")
        normalize_market_signals(list(self.market_signals))
        for momentum in (
            self.attack_relative, self.attack_absolute, self.defense_relative,
            self.defense_absolute, self.canary_momentum,
        ):
            momentum.build()
        if "장단기 금리차 역전 (10년-2년)" in self.market_signals and self.spread_10y2y is None:
            raise ValueError(f"{self.name}: 10년-2년 금리차 설정이 없습니다.")
        if "장단기 금리차 역전 (10년-3개월)" in self.market_signals and self.spread_10y3m is None:
            raise ValueError(f"{self.name}: 10년-3개월 금리차 설정이 없습니다.")

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.attack_tickers + self.defense_tickers + self.canary_tickers))

    def engine_config(
        self,
        start_date: dt.date,
        end_date: dt.date,
        *,
        initial_investment: float = 10_000.0,
        transaction_cost_rate: float = 0.0,
    ) -> dict[str, Any]:
        self.validate()
        spread_settings: dict[str, dict[str, Any]] = {}
        if self.spread_10y2y:
            spread_settings["10년-2년"] = self.spread_10y2y.as_dict()
        if self.spread_10y3m:
            spread_settings["10년-3개월"] = self.spread_10y3m.as_dict()
        return {
            "start_date": start_date,
            "end_date": end_date,
            "initial_investment": float(initial_investment),
            "investment_type": "거치식",
            "rebalance_frequency": "월별",
            "contribution_amount": 0.0,
            "contribution_frequency": "매월",
            "attack_tickers": list(self.attack_tickers),
            "defense_tickers": list(self.defense_tickers),
            "attack_mode": self.attack_mode,
            "attack_select_count": self.attack_select_count,
            "attack_relative": self.attack_relative.build(),
            "attack_absolute": self.attack_absolute.build(),
            "attack_use_bil": self.attack_use_bil,
            "defense_enabled": bool(self.defense_tickers),
            "defense_mode": self.defense_mode,
            "defense_select_count": self.defense_select_count,
            "defense_relative": self.defense_relative.build(),
            "defense_absolute": self.defense_absolute.build(),
            "defense_same_momentum": self.defense_same_momentum,
            "defense_use_bil": self.defense_use_bil,
            "include_cash": self.include_cash,
            "weight_mode": self.weight_mode,
            "fixed_defense_weight": self.fixed_defense_weight,
            "negative_threshold": self.negative_threshold,
            "negative_same_momentum": self.negative_same_momentum,
            "positive_same_momentum": self.positive_same_momentum,
            "risk_avoidance": {
                "enabled": bool(self.canary_tickers),
                "tickers": list(self.canary_tickers),
                "apply_mode": self.canary_apply_mode,
                "direction": self.canary_direction,
                "momentum_type": self.canary_momentum_type,
                "same_momentum": self.canary_same_momentum,
                "custom_momentum": self.canary_momentum.build(),
                "decision_percent_digits": self.canary_decision_percent_digits,
            },
            "market_risk": {
                "enabled": bool(self.market_signals),
                "combine": self.market_combine,
                "signals": list(self.market_signals),
                "spread_settings": spread_settings,
                "dividend_threshold": self.dividend_threshold,
                "unemployment_months": 12,
            },
            "cash_replacement_ticker": self.cash_replacement_ticker,
            "bil_ticker": "BIL",
            "use_transaction_cost": transaction_cost_rate > 0,
            "transaction_cost_rate": float(transaction_cost_rate),
        }


S13612 = MomentumSpec("모멘텀 스코어", score_type="1-3-6-12개월 단순")
S3612 = MomentumSpec("모멘텀 스코어", score_type="3-6-12개월 단순")
S136912 = MomentumSpec("모멘텀 스코어", score_type="1-3-6-9-12개월 단순")
MA6 = MomentumSpec("이동평균", 6)

PRESETS: dict[str, PortfolioPreset] = {
    "1. 분산투자 방어형": PortfolioPreset(
        name="1. 분산투자 방어형",
        attack_tickers=("QQQ", "SOXX", "XLE", "XLK", "XLV", "EDV", "GLD", "DBC"),
        defense_tickers=("BIL", "IEF", "HYG"),
        attack_mode="상대 모멘텀", attack_select_count=4,
        attack_relative=S13612, attack_absolute=S13612,
        defense_mode="상대 모멘텀", defense_select_count=1,
        defense_relative=S13612, defense_absolute=S13612, defense_same_momentum=True,
        weight_mode="마이너스 절대 모멘텀 수", negative_threshold=5,
        canary_tickers=("SPY", "TIP"), canary_momentum=MomentumSpec("이동평균", 11),
        canary_decision_percent_digits=None,
        market_signals=("S&P500 배당수익률",), dividend_threshold=1.32,
    ),
    "2. 집중투자 공격형": PortfolioPreset(
        name="2. 집중투자 공격형",
        attack_tickers=("QQQ", "SCHD", "VWO"),
        defense_tickers=("GLD", "HYG", "PDBC"),
        attack_mode="상대 모멘텀", attack_select_count=1,
        attack_relative=S3612, attack_absolute=S3612,
        defense_mode="상대 & 절대 모멘텀 + 현금 스위치", defense_select_count=1,
        defense_relative=MA6, defense_absolute=MA6,
        canary_tickers=("TIP",), canary_momentum=MomentumSpec("이동평균", 12),
        canary_decision_percent_digits=None,
        market_signals=("장단기 금리차 역전 (10년-3개월)", "S&P500 배당수익률"),
        spread_10y3m=SpreadRule(7, 8, -0.5), dividend_threshold=1.6,
    ),
    "7. 집중투자 공격형 1-3-6-9-12": PortfolioPreset(
        name="7. 집중투자 공격형 1-3-6-9-12",
        attack_tickers=("QQQ", "SCHD", "VWO"),
        defense_tickers=("GLD", "HYG", "PDBC"),
        attack_mode="상대 모멘텀", attack_select_count=1,
        attack_relative=S136912, attack_absolute=S136912,
        defense_mode="상대 & 절대 모멘텀 + 현금 스위치", defense_select_count=1,
        defense_relative=MA6, defense_absolute=MA6,
        canary_tickers=("TIP",), canary_momentum=MomentumSpec("이동평균", 12),
        canary_decision_percent_digits=None,
        market_signals=("장단기 금리차 역전 (10년-3개월)", "S&P500 배당수익률"),
        spread_10y3m=SpreadRule(7, 8, -0.5), dividend_threshold=1.6,
    ),
}

for _preset in PRESETS.values():
    _preset.validate()


def get_preset(name: str) -> PortfolioPreset:
    try:
        return PRESETS[name]
    except KeyError as error:
        raise ValueError(f"알 수 없는 프리셋: {name}") from error
