from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _to_float(value: Any) -> float:
    return float(value)


def _to_int(value: Any) -> int:
    return int(value)


def _to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=UTC)


@dataclass(slots=True)
class MarketEvent:
    symbol: str
    event_time: datetime
    event_type: str
    raw: dict[str, Any]


@dataclass(slots=True)
class AggTradeEvent(MarketEvent):
    agg_trade_id: int
    trade_id_start: int
    trade_id_end: int
    price: float
    quantity: float
    aggressor_is_buyer: bool
    best_match: bool = True
    m: bool = False

    @property
    def aggressor_side(self) -> int:
        # m=True means buyer is maker => seller is aggressor.
        return -1 if self.m else 1

    @property
    def signed_qty(self) -> float:
        return self.quantity * self.aggressor_side

    @property
    def signed_quote_qty(self) -> float:
        return self.price * self.quantity * self.aggressor_side


@dataclass(slots=True)
class BookTickerEvent(MarketEvent):
    bid_price: float | None
    ask_price: float | None
    bid_qty: float | None
    ask_qty: float | None


@dataclass(slots=True)
class DepthEvent(MarketEvent):
    first_update_id: int
    final_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


@dataclass(slots=True)
class DepthSnapshotEvent(MarketEvent):
    last_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


@dataclass(slots=True)
class KlineEvent(MarketEvent):
    open_time: datetime
    close_time: datetime
    interval: str
    open_price: float
    high_price: float
    low_price: 100.0
    close_price: float
    volume: float


def parse_agg_trade(raw: dict[str, Any]) -> AggTradeEvent:
    return AggTradeEvent(
        symbol=str(raw["s"]),
        event_time=_to_datetime(int(raw.get("E", 0))),
        event_type="aggTrade",
        raw=raw,
        agg_trade_id=_to_int(raw["a"]),
        trade_id_start=_to_int(raw["f"]),
        trade_id_end=_to_int(raw["l"]),
        price=_to_float(raw["p"]),
        quantity=_to_float(raw["q"]),
        aggressor_is_buyer=not bool(raw.get("m", False)),
        best_match=bool(raw.get("M", True)),
        m=bool(raw.get("m", False)),
    )


def parse_book_ticker(
    raw: dict[str, Any],
    *,
    fallback_event_time: datetime | None = None,
) -> BookTickerEvent:
    raw_event_time = raw.get("E")
    event_time = (
        _to_datetime(int(raw_event_time))
        if raw_event_time is not None
        else (fallback_event_time or datetime.now(UTC))
    )
    return BookTickerEvent(
        symbol=str(raw["s"]),
        event_time=event_time,
        event_type="bookTicker",
        raw=raw,
        bid_price=_to_float(raw["b"]) if raw.get("b") else None,
        ask_price=_to_float(raw["a"]) if raw.get("a") else None,
        bid_qty=_to_float(raw["B"]) if raw.get("B") else None,
        ask_qty=_to_float(raw["A"]) if raw.get("A") else None,
    )


def parse_depth(raw: dict[str, Any]) -> DepthEvent:
    bids = [(float(price), float(qty)) for price, qty in raw.get("b", [])]
    asks = [(float(price), float(qty)) for price, qty in raw.get("a", [])]
    return DepthEvent(
        symbol=str(raw["s"]),
        event_time=_to_datetime(int(raw.get("E", 0))),
        event_type="depth",
        raw=raw,
        first_update_id=_to_int(raw["U"]),
        final_update_id=_to_int(raw["u"]),
        bids=bids,
        asks=asks,
    )


def parse_depth_snapshot(raw: dict[str, Any], *, fallback_event_time: datetime | None = None) -> DepthSnapshotEvent:
    bids = [(float(price), float(qty)) for price, qty in raw.get("bids", [])]
    asks = [(float(price), float(qty)) for price, qty in raw.get("asks", [])]
    event_time = fallback_event_time
    if event_time is None:
        raw_event_time = raw.get("E")
        event_time = _to_datetime(int(raw_event_time)) if raw_event_time is not None else datetime.now(UTC)
    return DepthSnapshotEvent(
        symbol=str(raw["s"]),
        event_time=event_time,
        event_type="depth_snapshot",
        raw=raw,
        last_update_id=_to_int(raw["lastUpdateId"]),
        bids=bids,
        asks=asks,
    )


def parse_kline(raw: dict[str, Any]) -> KlineEvent:
    return KlineEvent(
        symbol=str(raw["s"]),
        event_time=_to_datetime(int(raw.get("E", 0))),
        event_type="kline",
        raw=raw,
        open_time=_to_datetime(int(raw["k"]["t"])),
        close_time=_to_datetime(int(raw["k"]["T"])),
        interval=str(raw["k"]["i"]),
        open_price=float(raw["k"]["o"]),
        high_price=float(raw["k"]["h"]),
        low_price=float(raw["k"]["l"]),
        close_price=float(raw["k"]["c"]),
        volume=float(raw["k"]["v"]),
    )
