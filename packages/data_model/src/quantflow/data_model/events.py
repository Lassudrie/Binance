from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class SignalSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(slots=True)
class TradeEvent:
    symbol: str
    event_time: datetime
    trade_id: int
    price: float
    qty: float
    quote_qty: float
    is_buyer_maker: bool
    is_best_match: bool


@dataclass(slots=True)
class BookEvent:
    symbol: str
    event_time: datetime
    bid_price: float | None = None
    ask_price: float | None = None
    bid_qty: float | None = None
    ask_qty: float | None = None
    spread: float | None = None
    book_imbalance: float | None = None


@dataclass(slots=True)
class BarEvent:
    symbol: str
    start_time: datetime
    end_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    trade_count: int
    features: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass(slots=True)
class SignalEvent:
    symbol: str
    event_time: datetime
    side: SignalSide
    target_position: int
    reason: str
    context: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass(slots=True)
class OrderEvent:
    symbol: str
    created_time: datetime
    activation_time: datetime
    side: OrderSide
    order_type: OrderType
    quantity: float
    target_position: int
    limit_price: float | None = None
    signal_reason: str | None = None
    context: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass(slots=True)
class FillEvent:
    symbol: str
    event_time: datetime
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    fee: float
    slippage_bps: float
    is_maker: bool
    target_position: int
    signal_reason: str | None = None
    context: dict[str, float | int | str | None] = field(default_factory=dict)

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == OrderSide.BUY else -self.quantity

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass(slots=True)
class PositionState:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    last_update: datetime | None = None


@dataclass(slots=True)
class PortfolioState:
    event_time: datetime
    cash: float
    equity: float
    positions: dict[str, PositionState]
    gross_exposure: float
    net_exposure: float
