"""Pure calculation engine for the single-asset split-trading strategy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import math

import pandas as pd


TRADE_COLUMNS = (
    "거래일", "당일 순번", "매수/매도", "차수", "거래 ID",
    "시뮬레이션 체결가", "수량", "체결금액", "실현손익", "실현수익률",
    "체결 후 현금잔액", "거래비용",
)


@dataclass(frozen=True)
class SplitTradingConfig:
    initial_capital: float
    tranche_amount: float
    buy_drop: float
    sell_rise: float
    quantity_mode: str = "fractional_6"
    slot_count: int | None = None
    buy_fee_rate: float = 0.0
    sell_fee_rate: float = 0.0
    sell_tax_rate: float = 0.0
    slippage_rate: float = 0.0
    fx_cost_rate: float = 0.0

    def validate(self) -> None:
        if self.initial_capital <= 0 or self.tranche_amount <= 0:
            raise ValueError("투자총액과 회당 매수금액은 0보다 커야 합니다.")
        if self.tranche_amount > self.initial_capital:
            raise ValueError("회당 매수금액은 최초 투자총액을 초과할 수 없습니다.")
        if not 0 < self.buy_drop < 1:
            raise ValueError("매수 하락률은 0% 초과 100% 미만이어야 합니다.")
        if self.sell_rise <= 0:
            raise ValueError("매도 상승률은 0%보다 커야 합니다.")
        if self.quantity_mode not in {"whole", "fractional_6"}:
            raise ValueError("지원하지 않는 수량 방식입니다.")
        if self.slot_count is not None and self.slot_count < 1:
            raise ValueError("분할횟수는 1 이상의 정수여야 합니다.")
        if self.max_slots > 1_000:
            raise ValueError("동시 보유 차수는 최대 1,000차까지 계산할 수 있습니다.")
        for name, value in (
            ("매수수수료", self.buy_fee_rate), ("매도수수료", self.sell_fee_rate),
            ("매도세금", self.sell_tax_rate), ("슬리피지", self.slippage_rate),
            ("환전비용", self.fx_cost_rate),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name}은 0% 이상 100% 미만이어야 합니다.")

    @property
    def max_slots(self) -> int:
        if self.slot_count is not None:
            return int(self.slot_count)
        return max(1, int(self.initial_capital // self.tranche_amount))


@dataclass
class _Position:
    slot: int
    trade_id: str
    price: float
    quantity: float
    cost_basis: float


def _order_quantity(budget: float, price: float, mode: str) -> float:
    raw = budget / price
    if mode == "whole":
        return float(math.floor(raw + 1e-12))
    return math.floor((raw + 1e-12) * 1_000_000) / 1_000_000


def _close_series(closes: pd.Series) -> pd.Series:
    series = pd.to_numeric(closes, errors="coerce").dropna().astype(float)
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError("요청 기간에 사용할 종가가 없습니다.")
    if (series <= 0).any():
        raise ValueError("가격은 0보다 커야 합니다.")
    return series


def _drawdown_metrics(nav: pd.Series) -> dict[str, object]:
    running_peak = nav.cummax()
    drawdown = nav / running_peak - 1.0
    trough_date = pd.Timestamp(drawdown.idxmin())
    peak_date = pd.Timestamp(nav.loc[:trough_date].idxmax())
    peak_value = float(nav.loc[peak_date])
    recovered = nav.loc[trough_date:]
    recovered = recovered[recovered >= peak_value - 1e-9]
    recovery_date = pd.Timestamp(recovered.index[0]) if not recovered.empty else None
    end_date = recovery_date or pd.Timestamp(nav.index[-1])
    return {
        "max_drawdown": float(drawdown.min()),
        "current_drawdown": float(drawdown.iloc[-1]),
        "mdd_peak_date": peak_date.date(),
        "mdd_trough_date": trough_date.date(),
        "mdd_recovery_date": recovery_date.date() if recovery_date is not None else None,
        "mdd_duration_days": int((end_date - peak_date).days),
    }


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

    def record(
        side: str, slot: int, price: float, position: _Position,
        *, profit=None, trading_cost: float = 0.0,
    ) -> None:
        sequence = daily_sequence.get(current_date, 0) + 1
        daily_sequence[current_date] = sequence
        trades.append({
            "거래일": current_date.date().isoformat(), "당일 순번": sequence,
            "매수/매도": side, "차수": slot, "거래 ID": position.trade_id,
            "시뮬레이션 체결가": float(price), "수량": position.quantity,
            "체결금액": float(price * position.quantity),
            "실현손익": profit if side == "매도" else None,
            "실현수익률": (
                float(profit) / position.cost_basis
                if side == "매도" and profit is not None else None
            ),
            "체결 후 현금잔액": cash, "거래비용": trading_cost,
        })

    current_date = first_date
    buy_cost_rate = config.buy_fee_rate + config.fx_cost_rate
    sell_cost_rate = config.sell_fee_rate + config.sell_tax_rate + config.fx_cost_rate
    first_execution = first_price * (1.0 + config.slippage_rate)
    first_unit_cost = first_execution * (1.0 + buy_cost_rate)
    first_quantity = _order_quantity(config.tranche_amount, first_unit_cost, config.quantity_mode)
    if first_quantity <= 0:
        raise ValueError("회당 매수금액으로 첫 거래일에 1주 이상 매수할 수 없습니다.")
    first_gross = first_quantity * first_execution
    first_cost = first_gross * buy_cost_rate
    first_spend = first_gross + first_cost
    cash -= first_spend
    first_position = _Position(
        1, f"T{next(ids):06d}", first_execution, first_quantity, first_spend,
    )
    positions[1] = first_position
    record("매수", 1, first_execution, first_position, trading_cost=first_cost)

    # Temporary base follows highs while only the permanent first slot remains.
    temporary_base = first_price
    confirmed_base: float | None = None
    split_values: list[float] = []
    cash_values: list[float] = []
    holding_values: list[float] = []
    position_counts: list[int] = []
    full_quantity = _order_quantity(config.initial_capital, first_unit_cost, config.quantity_mode)
    full_spend = full_quantity * first_unit_cost
    full_cash = config.initial_capital - full_spend

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
                    execution = boundary * (1.0 + config.slippage_rate)
                    unit_cost = execution * (1.0 + buy_cost_rate)
                    quantity = _order_quantity(config.tranche_amount, unit_cost, config.quantity_mode)
                    if quantity <= 0:
                        continue
                    gross = quantity * execution
                    trading_cost = gross * buy_cost_rate
                    spend = gross + trading_cost
                    position = _Position(
                        slot, f"T{next(ids):06d}", execution, quantity, spend,
                    )
                    cash -= spend
                    positions[slot] = position
                    record(
                        "매수", slot, execution, position, trading_cost=trading_cost,
                    )

            elif close > previous:
                sellable = []
                for slot, position in positions.items():
                    if slot == 1:
                        continue
                    target = position.price * (1.0 + config.sell_rise)
                    if previous <= target <= close:
                        sellable.append((target, slot, position))
                for target, slot, position in sorted(sellable):
                    execution = target * (1.0 - config.slippage_rate)
                    gross = position.quantity * execution
                    trading_cost = gross * sell_cost_rate
                    proceeds = gross - trading_cost
                    profit = proceeds - position.cost_basis
                    cash += proceeds
                    del positions[slot]
                    record(
                        "매도", slot, execution, position,
                        profit=profit, trading_cost=trading_cost,
                    )
                if not any(slot > 1 for slot in positions):
                    temporary_base = max(float(confirmed_base or first_price), close)
                    confirmed_base = None

        holdings = sum(position.quantity * close for position in positions.values())
        split_values.append(cash + holdings)
        cash_values.append(cash)
        holding_values.append(holdings)
        position_counts.append(len(positions))

    split_nav = pd.Series(split_values, index=prices.index, name="분할매매")
    cash_nav = pd.Series(cash_values, index=prices.index, name="현금")
    holdings_nav = pd.Series(holding_values, index=prices.index, name="보유분")
    position_count = pd.Series(position_counts, index=prices.index, name="보유차수")
    full_nav = (prices * full_quantity + full_cash).rename("전액 투자")
    final_close = float(prices.iloc[-1])
    holdings_value = sum(position.quantity * final_close for position in positions.values())
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    full_return = float(full_nav.iloc[-1] / config.initial_capital - 1.0)
    split_return = float(split_nav.iloc[-1] / config.initial_capital - 1.0)
    full_risk = _drawdown_metrics(full_nav)
    split_risk = _drawdown_metrics(split_nav)
    result = {
        "prices": prices,
        "full_nav": full_nav,
        "split_nav": split_nav,
        "cash_nav": cash_nav,
        "holdings_nav": holdings_nav,
        "position_count": position_count,
        "trades": trade_frame,
        "max_slots": slots,
        "quantity_mode": config.quantity_mode,
        "final_cash": cash,
        "final_holdings_value": holdings_value,
        "full_final": float(full_nav.iloc[-1]),
        "split_final": float(split_nav.iloc[-1]),
        "full_return": full_return,
        "split_return": split_return,
        "return_difference": float((split_nav.iloc[-1] - full_nav.iloc[-1]) / config.initial_capital),
        "actual_start": prices.index[0].date(),
        "actual_end": prices.index[-1].date(),
        "open_positions": len(positions),
        "total_trading_cost": float(trade_frame["거래비용"].sum()),
    }
    result.update({f"full_{key}": value for key, value in full_risk.items()})
    result.update({f"split_{key}": value for key, value in split_risk.items()})
    result["full_return_mdd"] = (
        full_return / abs(float(full_risk["max_drawdown"]))
        if float(full_risk["max_drawdown"]) < 0 else None
    )
    result["split_return_mdd"] = (
        split_return / abs(float(split_risk["max_drawdown"]))
        if float(split_risk["max_drawdown"]) < 0 else None
    )
    return result

