from __future__ import annotations

from dataclasses import dataclass

from quantflow.data_model.events import BarEvent, FillEvent, OrderSide, PositionState
from quantflow.execution.simulator import ExecutionSimulator

EPSILON = 1e-12


@dataclass(slots=True)
class BracketState:
    symbol: str
    sign: int
    entry_bar_index: int
    stop_distance: float
    target_distance: float
    time_stop_bars: int
    entry_reason: str | None
    atr_entry: float | None
    risk_amount: float | None
    stop_price: float
    target_price: float
    context: dict[str, float | int | str | None]


class BracketManager:
    def __init__(self) -> None:
        self._states: dict[str, BracketState] = {}

    def reset(self) -> None:
        self._states.clear()

    def sync_after_fill(
        self,
        fill: FillEvent,
        position: PositionState | None,
        current_index: int,
    ) -> None:
        if position is None or abs(position.quantity) < EPSILON:
            self._states.pop(fill.symbol, None)
            return

        sign = 1 if position.quantity > 0.0 else -1
        existing = self._states.get(fill.symbol)

        stop_distance = _context_float(fill.context.get("stop_distance"))
        target_distance = _context_float(fill.context.get("target_distance"))
        time_stop_bars = _context_int(fill.context.get("time_stop_bars"))

        if existing is not None and existing.sign == sign:
            stop_distance = (
                stop_distance if stop_distance is not None else existing.stop_distance
            )
            target_distance = (
                target_distance if target_distance is not None else existing.target_distance
            )
            time_stop_bars = (
                time_stop_bars if time_stop_bars is not None else existing.time_stop_bars
            )

        if (
            stop_distance is None
            or target_distance is None
            or time_stop_bars is None
            or stop_distance <= 0.0
            or target_distance <= 0.0
        ):
            if existing is not None and existing.sign != sign:
                self._states.pop(fill.symbol, None)
            return

        base_context = (
            dict(existing.context)
            if existing is not None and existing.sign == sign
            else {}
        )
        base_context.update(fill.context)

        entry_bar_index = (
            existing.entry_bar_index
            if existing is not None and existing.sign == sign
            else current_index
        )
        entry_reason = _context_str(base_context.get("entry_reason")) or fill.signal_reason
        atr_entry = _context_float(base_context.get("atr_entry"))
        risk_amount = _context_float(base_context.get("risk_amount"))

        self._states[fill.symbol] = BracketState(
            symbol=fill.symbol,
            sign=sign,
            entry_bar_index=entry_bar_index,
            stop_distance=stop_distance,
            target_distance=target_distance,
            time_stop_bars=time_stop_bars,
            entry_reason=entry_reason,
            atr_entry=atr_entry,
            risk_amount=risk_amount,
            stop_price=position.avg_price - (sign * stop_distance),
            target_price=position.avg_price + (sign * target_distance),
            context=base_context,
        )

    def evaluate(
        self,
        event: BarEvent,
        position: PositionState | None,
        current_index: int,
        execution: ExecutionSimulator,
    ) -> FillEvent | None:
        if position is None or abs(position.quantity) < EPSILON:
            self._states.pop(event.symbol, None)
            return None

        state = self._states.get(event.symbol)
        if state is None:
            return None

        quantity = abs(position.quantity)
        side = OrderSide.SELL if state.sign > 0 else OrderSide.BUY
        trigger = self._trigger(event, state, current_index)
        if trigger is None:
            return None

        reason, reference_price, event_time = trigger
        context = dict(state.context)
        context.update(
            {
                "entry_reason": state.entry_reason,
                "atr_entry": state.atr_entry,
                "risk_amount": state.risk_amount,
                "stop_price": state.stop_price,
                "target_price": state.target_price,
            }
        )
        return execution.build_market_fill(
            symbol=event.symbol,
            event_time=event_time,
            side=side,
            quantity=quantity,
            reference_price=reference_price,
            target_position=0,
            signal_reason=reason,
            context=context,
        )

    def _trigger(
        self,
        event: BarEvent,
        state: BracketState,
        current_index: int,
    ) -> tuple[str, float, object] | None:
        if state.sign > 0:
            if event.open_price <= state.stop_price:
                return ("bracket_stop_loss", event.open_price, event.start_time)
            if event.open_price >= state.target_price:
                return ("bracket_take_profit", state.target_price, event.start_time)
            if event.low_price <= state.stop_price and event.high_price >= state.target_price:
                return ("bracket_stop_loss", state.stop_price, event.end_time)
            if event.low_price <= state.stop_price:
                return ("bracket_stop_loss", state.stop_price, event.end_time)
            if event.high_price >= state.target_price:
                return ("bracket_take_profit", state.target_price, event.end_time)
        else:
            if event.open_price >= state.stop_price:
                return ("bracket_stop_loss", event.open_price, event.start_time)
            if event.open_price <= state.target_price:
                return ("bracket_take_profit", state.target_price, event.start_time)
            if event.high_price >= state.stop_price and event.low_price <= state.target_price:
                return ("bracket_stop_loss", state.stop_price, event.end_time)
            if event.high_price >= state.stop_price:
                return ("bracket_stop_loss", state.stop_price, event.end_time)
            if event.low_price <= state.target_price:
                return ("bracket_take_profit", state.target_price, event.end_time)

        if (
            state.time_stop_bars > 0
            and (current_index - state.entry_bar_index) >= state.time_stop_bars
        ):
            return ("bracket_time_stop", event.open_price, event.start_time)
        return None


def _context_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _context_int(value: float | int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _context_str(value: float | int | str | None) -> str | None:
    if value is None:
        return None
    return str(value)
