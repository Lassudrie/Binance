from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from quantflow.data_model.events import FillEvent, PortfolioState, PositionState
from quantflow.utils.dates import session_from_hour


@dataclass(slots=True)
class TradeRecord:
    symbol: str
    side: str
    quantity: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    fees: float
    holding_seconds: float
    signal_reason: str | None
    entry_reason: str | None
    exit_reason: str | None
    hour_of_day: int
    session: str
    vol_regime: float | None
    atr_entry: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    risk_amount: float | None = None
    r_multiple: float | None = None
    mae_bps: float | None = None
    mfe_bps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Portfolio:
    def __init__(self, initial_cash: float) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, PositionState] = {}
        self.entry_times: dict[str, datetime] = {}
        self.entry_fees: dict[str, float] = {}
        self.entry_contexts: dict[str, dict[str, float | int | str | None]] = {}
        self.trade_excursions: dict[str, dict[str, float]] = {}
        self.fills: list[FillEvent] = []
        self.closed_trades: list[TradeRecord] = []
        self.equity_curve: list[dict[str, float | str]] = []

    def _position(self, symbol: str) -> PositionState:
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol=symbol)
        return self.positions[symbol]

    def apply_fill(self, fill: FillEvent) -> None:
        self.fills.append(fill)
        position = self._position(fill.symbol)
        prev_qty = position.quantity
        prev_avg = position.avg_price
        signed_qty = fill.signed_quantity

        self.cash -= signed_qty * fill.price + fill.fee
        position.fees_paid += fill.fee
        position.last_update = fill.event_time

        if prev_qty == 0 or prev_qty * signed_qty > 0:
            total_abs_qty = abs(prev_qty) + abs(signed_qty)
            position.quantity = prev_qty + signed_qty
            position.avg_price = (
                fill.price
                if total_abs_qty == abs(signed_qty)
                else ((abs(prev_qty) * prev_avg) + (abs(signed_qty) * fill.price)) / total_abs_qty
            )
            if prev_qty == 0:
                self.entry_times[fill.symbol] = fill.event_time
                self.entry_fees[fill.symbol] = fill.fee
                self.entry_contexts[fill.symbol] = dict(fill.context)
                self.trade_excursions[fill.symbol] = {"mae_bps": 0.0, "mfe_bps": 0.0}
            else:
                self.entry_fees[fill.symbol] = self.entry_fees.get(fill.symbol, 0.0) + fill.fee
            return

        closing_qty = min(abs(prev_qty), abs(signed_qty))
        gross_pnl = closing_qty * (fill.price - prev_avg) * (1.0 if prev_qty > 0 else -1.0)
        prior_entry_fees = self.entry_fees.get(fill.symbol, 0.0)
        entry_fee_alloc = prior_entry_fees * (closing_qty / abs(prev_qty))
        exit_fee_alloc = fill.fee * (closing_qty / abs(signed_qty))
        trade_fees = entry_fee_alloc + exit_fee_alloc
        net_pnl = gross_pnl - trade_fees
        position.realized_pnl += net_pnl

        entry_time = self.entry_times.get(fill.symbol, fill.event_time)
        context = self.entry_contexts.get(fill.symbol, dict(fill.context))
        excursions = self.trade_excursions.get(fill.symbol, {"mae_bps": 0.0, "mfe_bps": 0.0})
        risk_amount = _optional_float(context.get("risk_amount"))
        r_multiple = None
        if risk_amount is not None and risk_amount > 0.0:
            r_multiple = net_pnl / risk_amount
        self.closed_trades.append(
            TradeRecord(
                symbol=fill.symbol,
                side="long" if prev_qty > 0 else "short",
                quantity=closing_qty,
                entry_time=entry_time,
                exit_time=fill.event_time,
                entry_price=prev_avg,
                exit_price=fill.price,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                fees=trade_fees,
                holding_seconds=(fill.event_time - entry_time).total_seconds(),
                signal_reason=fill.signal_reason,
                entry_reason=_optional_str(context.get("entry_reason")),
                exit_reason=fill.signal_reason,
                hour_of_day=entry_time.hour,
                session=str(context.get("session") or session_from_hour(entry_time.hour)),
                vol_regime=(
                    None if context.get("vol_regime") is None else float(context["vol_regime"])  # type: ignore[arg-type]
                ),
                atr_entry=_optional_float(context.get("atr_entry")),
                stop_price=_optional_float(context.get("stop_price")),
                target_price=_optional_float(context.get("target_price")),
                risk_amount=risk_amount,
                r_multiple=r_multiple,
                mae_bps=float(excursions["mae_bps"]),
                mfe_bps=float(excursions["mfe_bps"]),
            )
        )

        remaining_qty = prev_qty + signed_qty
        if remaining_qty == 0:
            position.quantity = 0.0
            position.avg_price = 0.0
            self.entry_times.pop(fill.symbol, None)
            self.entry_fees.pop(fill.symbol, None)
            self.entry_contexts.pop(fill.symbol, None)
            self.trade_excursions.pop(fill.symbol, None)
            return

        if remaining_qty * prev_qty > 0:
            position.quantity = remaining_qty
            position.avg_price = prev_avg
            self.entry_fees[fill.symbol] = prior_entry_fees - entry_fee_alloc
            return

        opening_fee = fill.fee - exit_fee_alloc
        position.quantity = remaining_qty
        position.avg_price = fill.price
        self.entry_times[fill.symbol] = fill.event_time
        self.entry_fees[fill.symbol] = opening_fee
        self.entry_contexts[fill.symbol] = dict(fill.context)
        self.trade_excursions[fill.symbol] = {"mae_bps": 0.0, "mfe_bps": 0.0}

    def update_trade_excursions(self, symbol: str, high_price: float, low_price: float) -> None:
        position = self.positions.get(symbol)
        if position is None or position.quantity == 0.0 or position.avg_price <= 0.0:
            return

        excursions = self.trade_excursions.setdefault(symbol, {"mae_bps": 0.0, "mfe_bps": 0.0})
        avg_price = position.avg_price

        if position.quantity > 0.0:
            favorable = max((high_price / avg_price - 1.0) * 10_000.0, 0.0)
            adverse = max((1.0 - low_price / avg_price) * 10_000.0, 0.0)
        else:
            favorable = max((1.0 - low_price / avg_price) * 10_000.0, 0.0)
            adverse = max((high_price / avg_price - 1.0) * 10_000.0, 0.0)

        excursions["mfe_bps"] = max(excursions["mfe_bps"], favorable)
        excursions["mae_bps"] = max(excursions["mae_bps"], adverse)

    def mark_to_market(self, marks: dict[str, float], event_time: datetime) -> None:
        for symbol, position in self.positions.items():
            mark = marks.get(symbol)
            if mark is None or position.quantity == 0:
                position.unrealized_pnl = 0.0
            else:
                position.unrealized_pnl = position.quantity * (mark - position.avg_price)
            position.last_update = event_time

    def snapshot(self, event_time: datetime, marks: dict[str, float]) -> PortfolioState:
        self.mark_to_market(marks, event_time)
        gross_exposure = 0.0
        net_exposure = 0.0
        for symbol, position in self.positions.items():
            mark = marks.get(symbol, position.avg_price)
            net_exposure += position.quantity * mark
            gross_exposure += abs(position.quantity * mark)
        equity = self.cash + net_exposure
        cloned_positions = {
            symbol: PositionState(
                symbol=position.symbol,
                quantity=position.quantity,
                avg_price=position.avg_price,
                realized_pnl=position.realized_pnl,
                unrealized_pnl=position.unrealized_pnl,
                fees_paid=position.fees_paid,
                last_update=position.last_update,
            )
            for symbol, position in self.positions.items()
        }
        return PortfolioState(
            event_time=event_time,
            cash=self.cash,
            equity=equity,
            positions=cloned_positions,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
        )

    def record_equity(self, event_time: datetime, marks: dict[str, float]) -> PortfolioState:
        state = self.snapshot(event_time, marks)
        record: dict[str, float | str] = {
            "event_time": event_time.isoformat(),
            "cash": state.cash,
            "equity": state.equity,
            "gross_exposure": state.gross_exposure,
            "net_exposure": state.net_exposure,
        }
        for symbol, position in state.positions.items():
            record[f"{symbol}_qty"] = position.quantity
            record[f"{symbol}_avg_price"] = position.avg_price
            record[f"{symbol}_unrealized_pnl"] = position.unrealized_pnl
        self.equity_curve.append(record)
        return state


def _optional_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_str(value: float | int | str | None) -> str | None:
    if value is None:
        return None
    return str(value)
