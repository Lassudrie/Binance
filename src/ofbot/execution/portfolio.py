from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ofbot.execution.fills import FillEvent

EPSILON = 1e-12


@dataclass(slots=True)
class PositionSnapshot:
    symbol: str
    net_position: float
    avg_entry_price: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    last_price: float | None
    position_open_time: datetime | None
    mae_bps: float
    mfe_bps: float
    risk_context: dict[str, float | int | str | None]


@dataclass(slots=True)
class TradeRecord:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    strategy: str | None
    side: int
    qty: float
    entry_price: float
    exit_price: float
    realized_pnl: float
    gross_pnl: float
    fees: float
    spread_entry_bps: float | None
    spread_exit_bps: float | None
    holding_seconds: float
    regime: str | None
    mae_bps: float
    mfe_bps: float
    vol_bucket: float | None = None
    error_tags: str | None = None
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    entry_fill_count: int = 0
    exit_fill_count: int = 0
    entry_notional: float = 0.0
    exit_notional: float = 0.0
    closed_qty: float = 0.0


@dataclass(slots=True)
class ActiveTradeLedger:
    symbol: str
    side: int
    strategy: str | None
    regime: str | None
    entry_time: datetime
    last_fill_time: datetime
    entry_qty: float = 0.0
    closed_qty: float = 0.0
    entry_notional: float = 0.0
    exit_notional: float = 0.0
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    entry_fill_count: int = 0
    exit_fill_count: int = 0
    realized_gross_pnl: float = 0.0
    realized_net_pnl: float = 0.0
    entry_spread_weighted: float = 0.0
    entry_spread_notional: float = 0.0
    exit_spread_weighted: float = 0.0
    exit_spread_notional: float = 0.0


class PaperPortfolio:
    def __init__(self, *, initial_cash: float) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, float] = {}
        self.avg_price: dict[str, float] = {}
        self.realized_pnl: dict[str, float] = {}
        self.unrealized_pnl: dict[str, float] = {}
        self.fees_paid: dict[str, float] = {}
        self.position_open_time: dict[str, datetime] = {}
        self.max_favorable_excursion: dict[str, float] = {}
        self.max_adverse_excursion: dict[str, float] = {}
        self.risk_context: dict[str, dict[str, Any]] = {}
        self.last_price: dict[str, float] = {}
        self.active_trades: dict[str, ActiveTradeLedger] = {}
        self.closed_trades: list[TradeRecord] = []
        self.fills: list[FillEvent] = []

    def mark(self, symbol: str, price: float, mark_time: datetime) -> None:
        self.last_price[symbol] = price
        qty = self.positions.get(symbol, 0.0)
        if abs(qty) < EPSILON:
            self.unrealized_pnl[symbol] = 0.0
            return
        avg = self.avg_price.get(symbol, 0.0)
        if avg <= 0:
            self.unrealized_pnl[symbol] = 0.0
            return
        self.unrealized_pnl[symbol] = qty * (price - avg)
        self.max_favorable_excursion[symbol] = self._track_excursion(
            symbol=symbol,
            price=price,
            avg=avg,
            direction=1 if qty > 0 else -1,
            prefer_max=True,
        )
        self.max_adverse_excursion[symbol] = self._track_excursion(
            symbol=symbol,
            price=price,
            avg=avg,
            direction=1 if qty > 0 else -1,
            prefer_max=False,
        )

    def snapshot(self, symbol: str, mark_time: datetime | None = None) -> PositionSnapshot:
        if mark_time is not None and symbol in self.last_price:
            self.mark(symbol, self.last_price[symbol], mark_time)
        return PositionSnapshot(
            symbol=symbol,
            net_position=self.positions.get(symbol, 0.0),
            avg_entry_price=self.avg_price.get(symbol, 0.0),
            realized_pnl=self.realized_pnl.get(symbol, 0.0),
            unrealized_pnl=self.unrealized_pnl.get(symbol, 0.0),
            fees_paid=self.fees_paid.get(symbol, 0.0),
            last_price=self.last_price.get(symbol),
            position_open_time=self.position_open_time.get(symbol),
            mae_bps=self.max_adverse_excursion.get(symbol, 0.0),
            mfe_bps=self.max_favorable_excursion.get(symbol, 0.0),
            risk_context=self.risk_context.get(symbol, {}),
        )

    def apply_fill(self, fill: FillEvent) -> list[TradeRecord]:
        self.fills.append(fill)

        symbol = fill.symbol
        current_qty = self.positions.get(symbol, 0.0)
        current_avg = self.avg_price.get(symbol, 0.0)
        current_time = fill.event_time
        context = dict(self.risk_context.get(symbol, {}))

        fill_qty_signed = fill.fill_qty * (1 if fill.side > 0 else -1)
        self.cash -= fill_qty_signed * fill.fill_price
        self.cash -= fill.fee
        self.fees_paid[symbol] = self.fees_paid.get(symbol, 0.0) + fill.fee
        self.last_price[symbol] = fill.fill_price

        if abs(current_qty) < EPSILON:
            self._open_position(symbol, fill, current_time, context, fee=fill.fee, qty=fill.fill_qty)
            return []

        if (current_qty > 0 and fill.side > 0) or (current_qty < 0 and fill.side < 0):
            total_abs_qty = abs(current_qty) + fill.fill_qty
            if total_abs_qty > EPSILON:
                self.avg_price[symbol] = (
                    abs(current_qty) * current_avg + fill.fill_qty * fill.fill_price
                ) / total_abs_qty
            self.positions[symbol] = current_qty + fill_qty_signed
            self.unrealized_pnl[symbol] = self.positions[symbol] * (fill.fill_price - self.avg_price[symbol])
            self.position_open_time.setdefault(symbol, current_time)
            self._record_entry_fill(symbol, fill_qty=fill.fill_qty, fill_price=fill.fill_price, fee=fill.fee, spread_bps=fill.spread_bps, fill_time=current_time)
            return []

        to_close = min(abs(current_qty), fill.fill_qty)
        if to_close < EPSILON:
            return []

        gross_pnl = _position_pnl(current_qty, current_avg, fill.fill_price, to_close)
        close_fee_ratio = min(1.0, to_close / max(fill.fill_qty, EPSILON))
        realized_fee = fill.fee * close_fee_ratio
        net_pnl = gross_pnl - realized_fee
        self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + net_pnl

        self._record_exit_fill(
            symbol,
            close_qty=to_close,
            fill_price=fill.fill_price,
            fee=realized_fee,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            spread_bps=fill.spread_bps,
            fill_time=current_time,
        )

        new_qty = current_qty + fill_qty_signed
        open_time = self.position_open_time.get(symbol, current_time)
        trade: TradeRecord | None = None

        if abs(new_qty) < EPSILON:
            trade = self._finalize_trade(symbol=symbol, close_time=current_time, open_time=open_time)
            self._close_position(symbol)
        elif (current_qty > 0 > new_qty) or (current_qty < 0 < new_qty):
            trade = self._finalize_trade(symbol=symbol, close_time=current_time, open_time=open_time)
            self._close_position(symbol)
            remaining_qty = abs(new_qty)
            remaining_fee = max(0.0, fill.fee - realized_fee)
            self._open_position(
                symbol,
                fill,
                current_time,
                context,
                fee=remaining_fee,
                qty=remaining_qty,
            )
        else:
            self.positions[symbol] = new_qty
            self.avg_price[symbol] = current_avg
            self.position_open_time[symbol] = open_time
            self.unrealized_pnl[symbol] = new_qty * (fill.fill_price - self.avg_price[symbol])
            self.risk_context[symbol] = context

        if trade is None:
            return []
        self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) - trade.entry_fees
        self.closed_trades.append(trade)
        return [trade]

    @property
    def gross_notional(self) -> float:
        total = 0.0
        for symbol, qty in self.positions.items():
            if abs(qty) < EPSILON:
                continue
            total += abs(qty) * self.avg_price.get(symbol, 0.0)
        return total

    @property
    def net_exposure(self) -> float:
        total = 0.0
        for symbol, qty in self.positions.items():
            if abs(qty) < EPSILON:
                continue
            total += qty * self.avg_price.get(symbol, 0.0)
        return total

    @property
    def equity(self) -> float:
        return self.cash + sum(self.unrealized_pnl.values())

    @property
    def open_symbols(self) -> set[str]:
        return {symbol for symbol, qty in self.positions.items() if abs(qty) > EPSILON}

    def reset(self) -> None:
        self.__init__(initial_cash=self.initial_cash)

    def _open_position(
        self,
        symbol: str,
        fill: FillEvent,
        fill_time: datetime,
        context: dict[str, Any],
        *,
        fee: float,
        qty: float,
    ) -> None:
        signed_qty = qty if fill.side > 0 else -qty
        self.positions[symbol] = signed_qty
        self.avg_price[symbol] = fill.fill_price
        self.position_open_time[symbol] = fill_time
        self.realized_pnl.setdefault(symbol, 0.0)
        self.unrealized_pnl[symbol] = 0.0
        self.max_favorable_excursion[symbol] = 0.0
        self.max_adverse_excursion[symbol] = 0.0

        new_context = dict(context)
        new_context.update(
            {
                "strategy": fill.strategy_id,
                "entry_time": fill_time,
                "entry_price": fill.fill_price,
                "spread_entry_bps": fill.spread_bps,
                "regime": context.get("regime"),
                "stop_loss_bps": fill.stop_loss_bps,
                "take_profit_bps": fill.take_profit_bps,
                "trailing_stop_bps": fill.trailing_stop_bps,
                "max_holding_time_s": fill.max_holding_time_s,
            }
        )
        self.risk_context[symbol] = new_context
        self.active_trades[symbol] = ActiveTradeLedger(
            symbol=symbol,
            side=1 if fill.side > 0 else -1,
            strategy=fill.strategy_id,
            regime=str(new_context.get("regime")) if new_context.get("regime") is not None else "unlabeled",
            entry_time=fill_time,
            last_fill_time=fill_time,
        )
        self._record_entry_fill(symbol, fill_qty=qty, fill_price=fill.fill_price, fee=fee, spread_bps=fill.spread_bps, fill_time=fill_time)

    def _close_position(self, symbol: str) -> None:
        self.positions[symbol] = 0.0
        self.avg_price[symbol] = 0.0
        self.unrealized_pnl[symbol] = 0.0
        self.position_open_time.pop(symbol, None)
        self.max_favorable_excursion.pop(symbol, None)
        self.max_adverse_excursion.pop(symbol, None)
        self.risk_context.pop(symbol, None)
        self.active_trades.pop(symbol, None)

    def _record_entry_fill(
        self,
        symbol: str,
        *,
        fill_qty: float,
        fill_price: float,
        fee: float,
        spread_bps: float | None,
        fill_time: datetime,
    ) -> None:
        ledger = self.active_trades[symbol]
        notional = fill_qty * fill_price
        ledger.entry_qty += fill_qty
        ledger.entry_notional += notional
        ledger.entry_fees += fee
        ledger.entry_fill_count += 1
        ledger.last_fill_time = fill_time
        if spread_bps is not None:
            ledger.entry_spread_weighted += spread_bps * notional
            ledger.entry_spread_notional += notional

    def _record_exit_fill(
        self,
        symbol: str,
        *,
        close_qty: float,
        fill_price: float,
        fee: float,
        gross_pnl: float,
        net_pnl: float,
        spread_bps: float | None,
        fill_time: datetime,
    ) -> None:
        ledger = self.active_trades[symbol]
        notional = close_qty * fill_price
        ledger.closed_qty += close_qty
        ledger.exit_notional += notional
        ledger.exit_fees += fee
        ledger.exit_fill_count += 1
        ledger.realized_gross_pnl += gross_pnl
        ledger.realized_net_pnl += net_pnl
        ledger.last_fill_time = fill_time
        if spread_bps is not None:
            ledger.exit_spread_weighted += spread_bps * notional
            ledger.exit_spread_notional += notional

    def _finalize_trade(self, *, symbol: str, close_time: datetime, open_time: datetime) -> TradeRecord:
        ledger = self.active_trades[symbol]
        entry_price = ledger.entry_notional / max(ledger.entry_qty, EPSILON)
        exit_price = ledger.exit_notional / max(ledger.closed_qty, EPSILON)
        spread_entry = (
            ledger.entry_spread_weighted / ledger.entry_spread_notional
            if ledger.entry_spread_notional > EPSILON
            else None
        )
        spread_exit = (
            ledger.exit_spread_weighted / ledger.exit_spread_notional
            if ledger.exit_spread_notional > EPSILON
            else None
        )
        return TradeRecord(
            symbol=symbol,
            entry_time=ledger.entry_time,
            exit_time=close_time,
            strategy=ledger.strategy,
            side=ledger.side,
            qty=ledger.closed_qty,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl=ledger.realized_net_pnl - ledger.entry_fees,
            gross_pnl=ledger.realized_gross_pnl,
            fees=ledger.entry_fees + ledger.exit_fees,
            spread_entry_bps=spread_entry,
            spread_exit_bps=spread_exit,
            holding_seconds=(close_time - open_time).total_seconds(),
            regime=ledger.regime,
            mae_bps=self.max_adverse_excursion.get(symbol, 0.0),
            mfe_bps=self.max_favorable_excursion.get(symbol, 0.0),
            entry_fees=ledger.entry_fees,
            exit_fees=ledger.exit_fees,
            entry_fill_count=ledger.entry_fill_count,
            exit_fill_count=ledger.exit_fill_count,
            entry_notional=ledger.entry_notional,
            exit_notional=ledger.exit_notional,
            closed_qty=ledger.closed_qty,
        )

    def _track_excursion(self, symbol: str, price: float, avg: float, direction: int, prefer_max: bool) -> float:
        if avg <= EPSILON:
            return 0.0
        if direction > 0:
            move_bps = (price / avg - 1.0) * 10_000.0
        else:
            move_bps = (avg / price - 1.0) * 10_000.0
        if prefer_max:
            return max(self.max_favorable_excursion.get(symbol, 0.0), move_bps)
        return min(self.max_adverse_excursion.get(symbol, 0.0), move_bps)


def _position_pnl(prev_qty: float, avg_price: float, exit_price: float, closing_qty: float) -> float:
    if prev_qty > 0:
        return (exit_price - avg_price) * closing_qty
    return (avg_price - exit_price) * closing_qty
