"""Shared-cash multi-asset split-trading calculation engine."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import pandas as pd

from features.snowball04_split_trading.engine import (
    _close_series,
    _drawdown_metrics,
    _order_quantity,
)


MULTI_TRADE_COLUMNS = (
    "거래일", "당일 순번", "종목", "매수/매도", "차수", "거래 ID",
    "시뮬레이션 체결가", "주문 기준가", "복원 목표가", "거래 상태",
    "수량", "체결금액", "실현손익", "실현수익률",
    "체결 후 현금잔액", "거래비용",
)


@dataclass(frozen=True)
class AssetRule:
    symbol: str
    tranche_amount: float
    slot_count: int
    buy_drop: float
    sell_rise: float
    allocation: float

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValueError("종목 코드가 비어 있습니다.")
        if self.tranche_amount <= 0:
            raise ValueError(f"{self.symbol}: 회당 매수금액은 0보다 커야 합니다.")
        if not 1 <= self.slot_count <= 1_000:
            raise ValueError(f"{self.symbol}: 분할횟수는 1~1,000회여야 합니다.")
        if not 0 < self.buy_drop < 1:
            raise ValueError(f"{self.symbol}: 매수 하락률은 0% 초과 100% 미만이어야 합니다.")
        if self.sell_rise <= 0:
            raise ValueError(f"{self.symbol}: 매도 상승률은 0%보다 커야 합니다.")
        if self.allocation <= 0:
            raise ValueError(f"{self.symbol}: 배정금액은 0보다 커야 합니다.")


@dataclass(frozen=True)
class MultiSplitConfig:
    initial_capital: float
    funding_mode: str = "limited"
    quantity_mode: str = "whole"
    buy_fee_rate: float = 0.0
    sell_fee_rate: float = 0.0
    sell_tax_rate: float = 0.0
    slippage_rate: float = 0.0
    fx_cost_rate: float = 0.0

    def validate(self) -> None:
        if self.funding_mode not in {"limited", "required"}:
            raise ValueError("지원하지 않는 자금 운용 방식입니다.")
        if self.funding_mode == "limited" and self.initial_capital <= 0:
            raise ValueError("투자총액은 0보다 커야 합니다.")
        if self.quantity_mode not in {"whole", "fractional_6"}:
            raise ValueError("지원하지 않는 수량 방식입니다.")
        for name, value in (
            ("매수수수료", self.buy_fee_rate), ("매도수수료", self.sell_fee_rate),
            ("매도세금", self.sell_tax_rate), ("슬리피지", self.slippage_rate),
            ("환전비용", self.fx_cost_rate),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name}은 0% 이상 100% 미만이어야 합니다.")


@dataclass
class _Position:
    slot: int
    trade_id: str
    price: float
    quantity: float
    cost_basis: float
    reference_price: float | None = None
    restore_target: float | None = None
    role: str = "normal"

    @property
    def strategy_price(self) -> float:
        return float(self.reference_price or self.price)


@dataclass
class _PendingOrder:
    slot: int
    target: float
    created: pd.Timestamp
    recovery_child: bool = False


@dataclass
class _AssetState:
    rule: AssetRule
    prices: pd.Series
    positions: dict[int, _Position]
    temporary_base: float
    confirmed_base: float | None
    previous_close: float
    pending: dict[int, _PendingOrder]
    recovery_slot: int | None = None
    recovery_peak: float | None = None


def _aligned_prices(prices: dict[str, pd.Series]) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    if len(prices) < 2:
        raise ValueError("다중 종목 포트폴리오는 종목을 2개 이상 선택해야 합니다.")
    clean = {symbol: _close_series(series) for symbol, series in prices.items()}
    common = None
    for series in clean.values():
        common = series.index if common is None else common.intersection(series.index)
    if common is None or common.empty:
        raise ValueError("선택한 모든 종목의 공통 거래일이 없습니다.")
    start, end = common[0], common[-1]
    union = clean[next(iter(clean))].index
    for series in list(clean.values())[1:]:
        union = union.union(series.index)
    union = union[(union >= start) & (union <= end)].sort_values()
    frame = pd.DataFrame({symbol: series.reindex(union).ffill() for symbol, series in clean.items()})
    frame = frame.dropna()
    return clean, frame


def run_multi_split_backtest(
    prices: dict[str, pd.Series], rules: list[AssetRule], config: MultiSplitConfig,
) -> dict:
    """Run multiple independent signal states against one shared cash balance."""
    config.validate()
    if len(rules) < 2:
        raise ValueError("종목별 조건을 2개 이상 입력해 주세요.")
    for rule in rules:
        rule.validate()
    symbols = [rule.symbol.strip().upper() for rule in rules]
    if len(set(symbols)) != len(symbols):
        raise ValueError("같은 종목을 중복해서 설정할 수 없습니다.")
    missing = [symbol for symbol in symbols if symbol not in prices]
    if missing:
        raise ValueError("가격 데이터가 없는 종목: " + ", ".join(missing))

    clean, frame = _aligned_prices({symbol: prices[symbol] for symbol in symbols})
    first_date = frame.index[0]
    reverse = config.funding_mode == "required"
    cash = 0.0 if reverse else float(config.initial_capital)
    buy_cost_rate = config.buy_fee_rate + config.fx_cost_rate
    sell_cost_rate = config.sell_fee_rate + config.sell_tax_rate + config.fx_cost_rate
    ids = count(1)
    trades: list[dict] = []
    daily_sequence: dict[pd.Timestamp, int] = {}
    states: dict[str, _AssetState] = {}

    def record(
        day: pd.Timestamp, symbol: str, side: str, slot: int, price: float,
        position: _Position, *, profit: float | None = None, cost: float = 0.0,
    ) -> None:
        sequence = daily_sequence.get(day, 0) + 1
        daily_sequence[day] = sequence
        trades.append({
            "거래일": day.date().isoformat(), "당일 순번": sequence,
            "종목": symbol, "매수/매도": side, "차수": slot,
            "거래 ID": position.trade_id, "시뮬레이션 체결가": float(price),
            "주문 기준가": position.strategy_price,
            "복원 목표가": position.restore_target,
            "거래 상태": {
                "recovery_base": "복원 대기 물량",
                "recovery_child": "복원 중 추가 물량",
            }.get(position.role, "정상 물량"),
            "수량": position.quantity, "체결금액": float(price * position.quantity),
            "실현손익": profit if side == "매도" else None,
            "실현수익률": (
                float(profit) / position.cost_basis
                if side == "매도" and profit is not None else None
            ),
            "체결 후 현금잔액": cash, "거래비용": cost,
        })

    # Start every strategy on the first common trading day.  When fixed cash is
    # tight, allocate the initial requests proportionally to avoid ticker-order bias.
    initial_orders = []
    for rule in rules:
        symbol = rule.symbol.strip().upper()
        close = float(frame.loc[first_date, symbol])
        initial_orders.append((symbol, rule, close))
    scale = 1.0
    requested = sum(rule.tranche_amount for _, rule, _ in initial_orders)
    if not reverse and requested > cash:
        scale = max(0.0, cash / requested)
    for symbol, rule, close in initial_orders:
        execution = close * (1.0 + config.slippage_rate)
        unit_cost = execution * (1.0 + buy_cost_rate)
        budget = rule.tranche_amount * scale
        quantity = _order_quantity(budget, unit_cost, config.quantity_mode)
        if quantity <= 0:
            raise ValueError(f"{symbol}: 회당 매수금액으로 첫 거래일에 매수할 수 없습니다.")
        gross = quantity * execution
        cost = gross * buy_cost_rate
        spend = gross + cost
        cash -= spend
        position = _Position(
            1, f"{symbol}-T{next(ids):06d}", execution, quantity, spend,
            reference_price=execution,
        )
        states[symbol] = _AssetState(
            rule, clean[symbol], {1: position}, close, None, close, {},
        )
        record(first_date, symbol, "매수", 1, execution, position, cost=cost)

    split_values: list[float] = []
    cash_values: list[float] = []
    holding_values: list[float] = []
    position_counts: list[int] = []
    per_asset_values = {symbol: [] for symbol in symbols}
    recovery_started = 0
    recovery_completed = 0

    for day in frame.index:
        if day != first_date:
            newly_triggered: list[tuple[str, _PendingOrder]] = []
            # Sells across every asset are credited before any same-day buys.
            for symbol in symbols:
                state = states[symbol]
                if day not in state.prices.index:
                    continue
                close = float(state.prices.loc[day])
                previous = state.previous_close
                recovery = (
                    state.positions.get(state.recovery_slot)
                    if state.recovery_slot is not None else None
                )
                if close > previous:
                    sellable = []
                    for slot, position in state.positions.items():
                        if slot == 1:
                            continue
                        # The delayed fill is the temporary first position of
                        # its recovery cycle.  It must survive until the
                        # originally missed price is restored; only its child
                        # positions rotate in the meantime.
                        if slot == state.recovery_slot and position.restore_target is not None:
                            continue
                        target = position.strategy_price * (1.0 + state.rule.sell_rise)
                        if close + 1e-12 >= target:
                            sellable.append((target, slot, position))
                    for target, slot, position in sorted(sellable):
                        execution = target * (1.0 - config.slippage_rate)
                        gross = position.quantity * execution
                        cost = gross * sell_cost_rate
                        proceeds = gross - cost
                        profit = proceeds - position.cost_basis
                        cash += proceeds
                        del state.positions[slot]
                        record(day, symbol, "매도", slot, execution, position, profit=profit, cost=cost)
                    if (
                        sellable and state.recovery_slot is None
                        and not any(slot > 1 for slot in state.positions)
                    ):
                        state.temporary_base = max(float(state.confirmed_base or close), close)
                        state.confirmed_base = None

                recovery = (
                    state.positions.get(state.recovery_slot)
                    if state.recovery_slot is not None else None
                )
                if recovery is not None and recovery.restore_target is not None:
                    restore_target = float(recovery.restore_target)
                    state.recovery_peak = max(
                        float(state.recovery_peak or recovery.price),
                        min(close, restore_target),
                    )
                    # Sell processing intentionally happens first.  With an
                    # equal buy/sell rate every child target is below the
                    # restore target: H(1-r)(1+r)=H(1-r²)<H.
                    child_slots = [
                        slot for slot in state.positions if slot > int(state.recovery_slot)
                    ]
                    if close + 1e-12 >= restore_target and not child_slots:
                        recovery.reference_price = restore_target
                        recovery.restore_target = None
                        recovery.role = "normal"
                        state.recovery_slot = None
                        state.recovery_peak = None
                        recovery_completed += 1
                    elif close < previous and state.rule.slot_count > int(state.recovery_slot):
                        peak = float(state.recovery_peak)
                        boundaries = {
                            slot: peak * (1.0 - state.rule.buy_drop) ** (
                                slot - int(state.recovery_slot)
                            )
                            for slot in range(int(state.recovery_slot) + 1, state.rule.slot_count + 1)
                        }
                        for slot, boundary in sorted(
                            boundaries.items(), key=lambda item: item[1], reverse=True,
                        ):
                            if (
                                slot not in state.positions and slot not in state.pending
                                and close <= boundary <= previous
                            ):
                                newly_triggered.append((
                                    symbol,
                                    _PendingOrder(slot, boundary, pd.Timestamp(day), True),
                                ))
                else:
                    extras = {
                        slot: pos for slot, pos in state.positions.items() if slot > 1
                    }
                    if not extras:
                        state.temporary_base = max(
                            state.temporary_base,
                            close if close >= previous else previous,
                        )
                    if close < previous and state.rule.slot_count > 1:
                        base = (
                            state.temporary_base
                            if not extras
                            else float(state.confirmed_base or state.temporary_base)
                        )
                        boundaries = {
                            slot: base * (1.0 - state.rule.buy_drop) ** (slot - 1)
                            for slot in range(2, state.rule.slot_count + 1)
                        }
                        for slot, boundary in sorted(
                            boundaries.items(), key=lambda item: item[1], reverse=True,
                        ):
                            if (
                                slot not in state.positions and slot not in state.pending
                                and close <= boundary <= previous
                            ):
                                newly_triggered.append((
                                    symbol,
                                    _PendingOrder(slot, boundary, pd.Timestamp(day), False),
                                ))
                state.previous_close = close

            for symbol, order in newly_triggered:
                states[symbol].pending.setdefault(order.slot, order)

            # Pending requests are not reduced proportionally.  Available cash
            # goes first to the asset that has fallen the most from its prior
            # actually held step; the remaining requests stay pending.
            eligible: list[tuple[float, pd.Timestamp, str, _PendingOrder]] = []
            for symbol in symbols:
                state = states[symbol]
                close = float(frame.loc[day, symbol])
                for order in state.pending.values():
                    if order.slot in state.positions or close > order.target + 1e-12:
                        continue
                    prior = [
                        (slot, position.strategy_price)
                        for slot, position in state.positions.items() if slot < order.slot
                    ]
                    anchor = (
                        max(prior, key=lambda item: item[0])[1]
                        if prior else state.temporary_base
                    )
                    decline = close / float(anchor) - 1.0
                    eligible.append((decline, order.created, symbol, order))

            for _, _, symbol, order in sorted(
                eligible, key=lambda item: (item[0], item[1], item[2], item[3].slot),
            ):
                state = states[symbol]
                if order.slot not in state.pending or order.slot in state.positions:
                    continue
                close = float(frame.loc[day, symbol])
                delayed = pd.Timestamp(day) > order.created
                raw_execution = min(close, order.target) if delayed else order.target
                execution = raw_execution * (1.0 + config.slippage_rate)
                unit_cost = execution * (1.0 + buy_cost_rate)
                budget = state.rule.tranche_amount
                quantity = _order_quantity(budget, unit_cost, config.quantity_mode)
                if quantity <= 0:
                    if reverse:
                        raise ValueError(f"{symbol}: 기준 1회 매수금액으로 신호를 체결할 수 없습니다.")
                    continue
                gross = quantity * execution
                cost = gross * buy_cost_rate
                spend = gross + cost
                if not reverse and spend > cash + 1e-9:
                    continue
                if state.confirmed_base is None:
                    state.confirmed_base = state.temporary_base
                cash -= spend
                starts_recovery = (
                    not reverse and delayed and not order.recovery_child
                    and raw_execution < order.target - 1e-9
                )
                position = _Position(
                    order.slot, f"{symbol}-T{next(ids):06d}", execution, quantity, spend,
                    reference_price=(raw_execution if starts_recovery else order.target),
                    restore_target=(order.target if starts_recovery else None),
                    role=(
                        "recovery_base" if starts_recovery
                        else "recovery_child" if order.recovery_child else "normal"
                    ),
                )
                state.positions[order.slot] = position
                del state.pending[order.slot]
                if starts_recovery:
                    state.recovery_slot = order.slot
                    state.recovery_peak = raw_execution
                    # Deeper fixed-grid requests are superseded by the dynamic
                    # high-watermark child ladder of the recovery position.
                    state.pending = {
                        slot: pending for slot, pending in state.pending.items()
                        if slot < order.slot
                    }
                    recovery_started += 1
                record(day, symbol, "매수", order.slot, execution, position, cost=cost)

        holdings = 0.0
        positions = 0
        for symbol in symbols:
            value = sum(
                position.quantity * float(frame.loc[day, symbol])
                for position in states[symbol].positions.values()
            )
            per_asset_values[symbol].append(value)
            holdings += value
            positions += len(states[symbol].positions)
        split_values.append(cash + holdings)
        cash_values.append(cash)
        holding_values.append(holdings)
        position_counts.append(positions)

    observed_cash = [(first_date, 0.0)] + list(zip(frame.index, cash_values))
    observed_cash.extend(
        (pd.Timestamp(item["거래일"]), float(item["체결 후 현금잔액"])) for item in trades
    )
    funding_date, funding_cash = min(observed_cash, key=lambda item: item[1])
    minimum_required = max(0.0, -float(funding_cash)) if reverse else None
    effective_capital = float(minimum_required) if reverse else float(config.initial_capital)
    if effective_capital <= 0:
        raise ValueError("필요 초기자금을 계산하지 못했습니다.")
    if reverse:
        cash += effective_capital
        cash_values = [value + effective_capital for value in cash_values]
        split_values = [value + effective_capital for value in split_values]
        for item in trades:
            item["체결 후 현금잔액"] = float(item["체결 후 현금잔액"]) + effective_capital

    split_nav = pd.Series(split_values, index=frame.index, name="다종목 분할매매")
    cash_nav = pd.Series(cash_values, index=frame.index, name="공유 현금")
    holdings_nav = pd.Series(holding_values, index=frame.index, name="전체 보유분")
    position_count = pd.Series(position_counts, index=frame.index, name="전체 보유차수")

    # Buy-and-hold benchmark uses each rule's declared allocation proportion.
    allocation_total = sum(rule.allocation for rule in rules)
    benchmark = pd.Series(effective_capital, index=frame.index, dtype=float)
    benchmark[:] = 0.0
    benchmark_cash = effective_capital
    for rule in rules:
        symbol = rule.symbol.strip().upper()
        budget = (
            rule.allocation
            if config.funding_mode == "limited"
            else effective_capital * rule.allocation / allocation_total
        )
        first_price = float(frame.iloc[0][symbol]) * (1.0 + config.slippage_rate)
        unit_cost = first_price * (1.0 + buy_cost_rate)
        quantity = _order_quantity(budget, unit_cost, config.quantity_mode)
        spend = quantity * unit_cost
        benchmark_cash -= spend
        benchmark += frame[symbol] * quantity
    benchmark += benchmark_cash
    benchmark.name = "동일 비중 장기보유"

    trade_frame = pd.DataFrame(trades, columns=MULTI_TRADE_COLUMNS)
    full_return = float(benchmark.iloc[-1] / effective_capital - 1.0)
    split_return = float(split_nav.iloc[-1] / effective_capital - 1.0)
    full_risk = _drawdown_metrics(benchmark)
    split_risk = _drawdown_metrics(split_nav)
    final_prices = frame.iloc[-1]
    summary_rows = []
    for rule in rules:
        symbol = rule.symbol.strip().upper()
        state = states[symbol]
        symbol_trades = trade_frame[trade_frame["종목"] == symbol]
        realized = float(symbol_trades["실현손익"].fillna(0).sum())
        final_value = sum(pos.quantity * float(final_prices[symbol]) for pos in state.positions.values())
        open_cost = sum(pos.cost_basis for pos in state.positions.values())
        summary_rows.append({
            "종목": symbol, "배정 기준금액": rule.allocation,
            "회당 매수금액": rule.tranche_amount, "분할횟수": rule.slot_count,
            "매수·매도율": rule.buy_drop, "매수 횟수": int((symbol_trades["매수/매도"] == "매수").sum()),
            "매도 횟수": int((symbol_trades["매수/매도"] == "매도").sum()),
            "실현손익": realized, "미실현손익": final_value - open_cost,
            "최종 평가액": final_value, "현재 보유 차수": len(state.positions),
            "대기 주문": len(state.pending),
            "복원 대기 차수": state.recovery_slot or "-",
        })
    asset_summary = pd.DataFrame(summary_rows)
    final_holdings = float(holdings_nav.iloc[-1])
    open_cost_basis = float(sum(
        pos.cost_basis for state in states.values() for pos in state.positions.values()
    ))
    result = {
        "initial_capital": effective_capital, "funding_mode": config.funding_mode,
        "tranche_amount": float(sum(rule.tranche_amount for rule in rules)),
        "minimum_required_capital": minimum_required,
        "minimum_required_capital_date": pd.Timestamp(funding_date).date() if reverse else None,
        "recommended_capital_10": effective_capital * 1.10 if reverse else None,
        "recommended_capital_20": effective_capital * 1.20 if reverse else None,
        "recommended_capital_30": effective_capital * 1.30 if reverse else None,
        "prices": (benchmark / benchmark.iloc[0] * 100).rename("다종목 기준지수"),
        "full_nav": benchmark, "split_nav": split_nav, "cash_nav": cash_nav,
        "holdings_nav": holdings_nav, "position_count": position_count,
        "trades": trade_frame, "asset_summary": asset_summary,
        "max_slots": int(sum(rule.slot_count for rule in rules)),
        "quantity_mode": config.quantity_mode, "final_cash": cash,
        "final_holdings_value": final_holdings, "open_cost_basis": open_cost_basis,
        "full_final": float(benchmark.iloc[-1]), "split_final": float(split_nav.iloc[-1]),
        "full_return": full_return, "split_return": split_return,
        "return_difference": split_return - full_return,
        "actual_start": frame.index[0].date(), "actual_end": frame.index[-1].date(),
        "open_positions": int(sum(len(state.positions) for state in states.values())),
        "pending_orders": int(sum(len(state.pending) for state in states.values())),
        "active_recoveries": int(sum(state.recovery_slot is not None for state in states.values())),
        "recovery_started": int(recovery_started),
        "recovery_completed": int(recovery_completed),
        "remaining_buy_slots": int(sum(
            rule.slot_count - len(states[rule.symbol.strip().upper()].positions) for rule in rules
        )),
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
