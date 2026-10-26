"""In-memory saved-run records used by the Streamlit presentation layer."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .presets import PortfolioPreset


@dataclass(frozen=True)
class SavedRun:
    preset: PortfolioPreset
    start_date: dt.date
    end_date: dt.date
    initial_investment: float
    transaction_cost_rate: float

    def validate(self) -> None:
        self.preset.validate()
        if self.start_date >= self.end_date:
            raise ValueError("시작일은 종료일보다 앞서야 합니다.")
        if self.initial_investment <= 0:
            raise ValueError("초기 투자금은 0보다 커야 합니다.")
        if not 0 <= self.transaction_cost_rate <= 0.05:
            raise ValueError("거래비용이 올바르지 않습니다.")
