"""Pure calculation engine for the single-asset split-trading strategy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import pandas as pd


TRADE_COLUMNS = (
    "거래일", "당일 순번", "매수/매도", "차수", "거래 ID",
    "시뮬레이션 체결가", "수량", "체결금액", "실현손익", "실현수익률",
    "체결 후 현금잔액",
)


@dataclass(frozen=True)
class SplitTradingConfig:
    initial_capital: float
    tranche_amount: float
    buy_drop: float
    sell_rise: float

    def validate(self) -> None:
        if self.initial_capital <= 0 or self.tranche_amount <= 0:
            raise ValueError("투자총액과 회당 매수금액은 0보다 커야 합니다.")
        if self.tranche_amount > self.initial_capital:
            raise ValueError("회당 매수금액은 최초 투자총액을 초과할 수 없습니다.")
        if not 0 < self.buy_drop < 1:
            raise ValueError("매수 하락률은 0% 초과 100% 미만이어야 합니다.")
        if self.sell_rise <= 0:
            raise ValueError("매도 상승률은 0%보다 커야 합니다.")

    @property
    def max_slots(self) -> int:
        return max(1, int(self.initial_capital // self.tranche_amount))


@dataclass
class _Position:
    slot: int
    trade_id: str
    price: float
    quantity: float


def _close_series(closes: pd.Series) -> pd.Series:
    series = pd.to_numeric(closes, errors="coerce").dropna().astype(float)
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError("요청 기간에 사용할 종가가 없습니다.")
    if (series <= 0).any():
        raise ValueError("가격은 0보다 커야 합니다.")
    return series


def run_split_backtest(closes: pd.Series, config: SplitTradingConfig) -> dict:
    """Compare full investment with the specified split-trading state machine."""
    config.validate()
    prices = _close_series(closes)
    first_date, first_price = prices.index[0], float(prices.iloc[0])
    slots = config.max_slots
    ids = count(1)
    cash = float(config.initial_capital)
    positions: dict[int, _Position] = {}
    trades: list[dict] = []
    daily_sequence: dict[pd.Timestamp, int] = {}

    def record(side: str, slot: int, price: float, position: _Position, *, profit=None) -> None:
        sequence = daily_sequence.get(current_date, 0) + 1
        daily_sequence[current_date] = sequence
        trades.append({
            "거래일": current_date.date().isoformat(), "당일 순번": sequence,
            "매수/매도": side, "차수": slot, "거래 ID": position.trade_id,
            "시뮬레이션 체결가": float(price), "수량": position.quantity,
            "체결금액": float(price * position.quantity),
            "실현손익": profit if side == "매도" else None,
            "실현수익률": (price / position.price - 1.0) if side == "매도" else None,
            "체결 후 현금잔액": cash,
        })

    current_date = first_date
    first_quantity = config.tranche_amount / first_price
    cash -= config.tranche_amount
    first_position = _Position(1, f"T{next(ids):06d}", first_price, first_quantity)
    positions[1] = first_position
    record("매수", 1, first_price, first_position)

    # Temporary base follows highs while only the permanent first slot remains.
    temporary_base = first_price
    confirmed_base: float | None = None
    split_values: list[float] = []
    full_quantity = config.initial_capital / first_price

    for index, (current_date, close) in enumerate(prices.items()):
        close = float(close)
        if index:
            previous = float(prices.iloc[index - 1])
            extras = {slot: position for slot, position in positions.items() if slot > 1}
            if not extras:
                temporary_base = max(temporary_base, close if close >= previous else previous)

            if close < previous and slots > 1:
                base = temporary_base if not extras else float(confirmed_base or temporary_base)
                boundaries = {slot: base * (1.0 - config.buy_drop) ** (slot - 1)
                              for slot in range(2, slots + 1)}
                # A falling interval meets the highest boundary first.
                for slot, boundary in sorted(boundaries.items(), key=lambda item: item[1], reverse=True):
                    if slot in positions or not close <= boundary <= previous:
                        continue
                    if cash + 1e-9 < config.tranche_amount:
                        continue
                    if confirmed_base is None:
                        confirmed_base = base
                    quantity = config.tranche_amount / boundary
                    position = _Position(slot, f"T{next(ids):06d}", boundary, quantity)
                    cash -= config.tranche_amount
                    positions[slot] = position
                    record("매수", slot, boundary, position)

            elif close > previous:
                sellable = []
                for slot, position in positions.items():
                    if slot == 1:
                        continue
                    target = position.price * (1.0 + config.sell_rise)
                    if previous <= target <= close:
                        sellable.append((target, slot, position))
                for target, slot, position in sorted(sellable):
                    proceeds = position.quantity * target
                    profit = proceeds - position.quantity * position.price
                    cash += proceeds
                    del positions[slot]
                    record("매도", slot, target, position, profit=profit)
                if not any(slot > 1 for slot in positions):
                    temporary_base = max(float(confirmed_base or first_price), close)
                    confirmed_base = None

        split_values.append(cash + sum(position.quantity * close for position in positions.values()))

    split_nav = pd.Series(split_values, index=prices.index, name="분할매매")
    full_nav = (prices * full_quantity).rename("전액 투자")
    final_close = float(prices.iloc[-1])
    holdings_value = sum(position.quantity * final_close for position in positions.values())
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return {
        "prices": prices,
        "full_nav": full_nav,
        "split_nav": split_nav,
        "trades": trade_frame,
        "max_slots": slots,
        "final_cash": cash,
        "final_holdings_value": holdings_value,
        "full_final": float(full_nav.iloc[-1]),
        "split_final": float(split_nav.iloc[-1]),
        "full_return": float(full_nav.iloc[-1] / config.initial_capital - 1.0),
        "split_return": float(split_nav.iloc[-1] / config.initial_capital - 1.0),
        "return_difference": float((split_nav.iloc[-1] - full_nav.iloc[-1]) / config.initial_capital),
        "actual_start": prices.index[0].date(),
        "actual_end": prices.index[-1].date(),
    }

