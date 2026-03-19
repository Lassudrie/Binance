from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


OrderSide = str


@dataclass(slots=True)
class PaperOrder:
    order_id: str
    symbol: str
    event_time: datetime
    side: int
    order_type: str
    requested_qty: float
    remaining_qty: float
    reference_price: float
    target_qty: float
    stop_loss_bps: float | None = None
    take_profit_bps: float | None = None
    trailing_stop_bps: float | None = None
    max_holding_time_s: int | None = None
    strategy_id: str | None = None
    context: dict[str, float | int | str | None] = None  # type: ignore[assignment]
    expire_time: datetime | None = None
    activation_time: datetime | None = None

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = {}

    @property
    def is_buy(self) -> bool:
        return self.side > 0

    @property
    def is_sell(self) -> bool:
        return self.side < 0


@dataclass(slots=True)
class FillEvent:
    symbol: str
    event_time: datetime
    side: int
    fill_qty: float
    fill_price: float
    fee: float
    slippage_bps: float
    is_maker: bool
    notional: float
    strategy_id: str | None
    context: dict[str, float | int | str | None]
    order_id: str
    stop_loss_bps: float | None = None
    take_profit_bps: float | None = None
    trailing_stop_bps: float | None = None
    max_holding_time_s: int | None = None
    spread_bps: float | None = None
