from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from ofbot.market.trades import BookTickerEvent, DepthEvent, DepthSnapshotEvent


BookSyncState = Literal["synced", "unsynced", "invalid"]
EPSILON = 1e-12


@dataclass(slots=True)
class DepthSyncResult:
    applied: bool = False
    ignored: bool = False
    needs_snapshot: bool = False
    reason: str | None = None


@dataclass(slots=True)
class OrderBookState:
    symbol: str
    require_depth_sync: bool = False
    book_ticker_divergence_bps: float = 50.0
    bid_price: float | None = None
    ask_price: float | None = None
    bid_qty: float | None = None
    ask_qty: float | None = None
    last_update_id: int | None = None
    last_event_time: datetime | None = None
    last_raw: dict[str, Any] | None = None
    sync_state: BookSyncState = "unsynced"
    last_sync_reason: str | None = None
    depth_mode_active: bool = False
    _depth_bids: dict[float, float] = field(default_factory=dict)
    _depth_asks: dict[float, float] = field(default_factory=dict)
    _pending_depth_events: deque[DepthEvent] = field(default_factory=lambda: deque(maxlen=4096))
    _book_ticker_bid_price: float | None = None
    _book_ticker_ask_price: float | None = None
    _book_ticker_bid_qty: float | None = None
    _book_ticker_ask_qty: float | None = None

    def apply_book_ticker(self, event: BookTickerEvent) -> None:
        self._book_ticker_bid_price = event.bid_price if event.bid_price is not None else self._book_ticker_bid_price
        self._book_ticker_ask_price = event.ask_price if event.ask_price is not None else self._book_ticker_ask_price
        self._book_ticker_bid_qty = event.bid_qty if event.bid_qty is not None else self._book_ticker_bid_qty
        self._book_ticker_ask_qty = event.ask_qty if event.ask_qty is not None else self._book_ticker_ask_qty
        self.last_event_time = event.event_time
        self.last_raw = event.raw
        if not self.depth_mode_active or not self.is_synced:
            self._refresh_top_from_book_ticker()

    def apply_depth(self, event: DepthEvent) -> DepthSyncResult:
        self.depth_mode_active = True
        self.last_event_time = event.event_time
        self.last_raw = event.raw

        if not self.is_synced:
            self._pending_depth_events.append(event)
            return DepthSyncResult(needs_snapshot=True, reason="depth_snapshot_required")

        if self.last_update_id is None:
            self._mark_unsynced("missing_last_update")
            self._pending_depth_events.append(event)
            return DepthSyncResult(needs_snapshot=True, reason="depth_snapshot_required")

        if event.final_update_id <= self.last_update_id:
            return DepthSyncResult(ignored=True, reason="stale_depth_event")

        if event.first_update_id > self.last_update_id + 1:
            self._mark_unsynced("depth_gap")
            self._pending_depth_events.clear()
            self._pending_depth_events.append(event)
            return DepthSyncResult(needs_snapshot=True, reason="depth_gap")

        self._apply_diff(event)
        self.last_update_id = event.final_update_id
        if not self._validate_depth_book():
            self._mark_invalid("depth_invalid")
            return DepthSyncResult(needs_snapshot=True, reason="depth_invalid")
        return DepthSyncResult(applied=True)

    def apply_depth_snapshot(self, event: DepthSnapshotEvent) -> DepthSyncResult:
        self.depth_mode_active = True
        self.sync_state = "synced"
        self.last_sync_reason = "snapshot_applied"
        self.last_update_id = event.last_update_id
        self.last_event_time = event.event_time
        self.last_raw = event.raw
        self._depth_bids = {float(price): float(qty) for price, qty in event.bids if float(qty) > 0.0}
        self._depth_asks = {float(price): float(qty) for price, qty in event.asks if float(qty) > 0.0}
        self._refresh_top_from_depth()
        if not self._validate_depth_book():
            self._mark_invalid("snapshot_invalid")
            return DepthSyncResult(needs_snapshot=True, reason="snapshot_invalid")
        return self._replay_buffered_depth()

    @property
    def is_synced(self) -> bool:
        return self.sync_state == "synced"

    @property
    def is_tradeable(self) -> bool:
        if self.depth_mode_active:
            return self.is_synced and self.bid_price is not None and self.ask_price is not None
        if self.require_depth_sync:
            return False
        return self.bid_price is not None and self.ask_price is not None

    @property
    def mid_price(self) -> float | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread_bps(self) -> float | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        if self.bid_price <= 0:
            return None
        return ((self.ask_price - self.bid_price) / self.bid_price) * 10_000.0

    @property
    def microprice(self) -> float | None:
        if self.bid_price is None or self.ask_price is None:
            return self.mid_price
        if self.bid_qty is None or self.ask_qty is None:
            return self.mid_price
        qty_sum = self.bid_qty + self.ask_qty
        if qty_sum <= 0:
            return self.mid_price
        return (self.bid_price * self.ask_qty + self.ask_price * self.bid_qty) / qty_sum

    @property
    def queue_imbalance(self) -> float:
        if self.bid_qty is None or self.ask_qty is None:
            return 0.0
        qty_sum = self.bid_qty + self.ask_qty
        if qty_sum <= 0:
            return 0.0
        return (self.bid_qty - self.ask_qty) / qty_sum

    def market_levels(self, side: int, *, max_depth_bps: float | None = None) -> list[tuple[float, float]]:
        if side > 0:
            levels = sorted(self._depth_asks.items(), key=lambda item: item[0])
            if not levels and self.ask_price is not None and self.ask_qty is not None:
                return [(self.ask_price, self.ask_qty)]
            if max_depth_bps is None or self.ask_price is None:
                return levels
            cap_price = self.ask_price * (1.0 + max_depth_bps / 10_000.0)
            return [(price, qty) for price, qty in levels if price <= cap_price]

        levels = sorted(self._depth_bids.items(), key=lambda item: item[0], reverse=True)
        if not levels and self.bid_price is not None and self.bid_qty is not None:
            return [(self.bid_price, self.bid_qty)]
        if max_depth_bps is None or self.bid_price is None:
            return levels
        floor_price = self.bid_price * (1.0 - max_depth_bps / 10_000.0)
        return [(price, qty) for price, qty in levels if price >= floor_price]

    def _replay_buffered_depth(self) -> DepthSyncResult:
        while self._pending_depth_events:
            event = self._pending_depth_events.popleft()
            if self.last_update_id is None:
                self._mark_unsynced("missing_last_update")
                self._pending_depth_events.appendleft(event)
                return DepthSyncResult(needs_snapshot=True, reason="depth_snapshot_required")
            if event.final_update_id <= self.last_update_id:
                continue
            if event.first_update_id > self.last_update_id + 1:
                self._mark_unsynced("depth_gap_after_snapshot")
                self._pending_depth_events.appendleft(event)
                return DepthSyncResult(needs_snapshot=True, reason="depth_gap_after_snapshot")
            self._apply_diff(event)
            self.last_update_id = event.final_update_id
            self.last_event_time = event.event_time
            self.last_raw = event.raw
            if not self._validate_depth_book():
                self._mark_invalid("depth_invalid_after_snapshot")
                return DepthSyncResult(needs_snapshot=True, reason="depth_invalid_after_snapshot")
        return DepthSyncResult(applied=True)

    def _apply_diff(self, event: DepthEvent) -> None:
        for price, qty in event.bids:
            self._set_level(self._depth_bids, price, qty)
        for price, qty in event.asks:
            self._set_level(self._depth_asks, price, qty)
        self._refresh_top_from_depth()

    def _set_level(self, levels: dict[float, float], price: float, qty: float) -> None:
        if qty <= EPSILON:
            levels.pop(price, None)
            return
        levels[price] = qty

    def _refresh_top_from_depth(self) -> None:
        best_bid = max(self._depth_bids.items(), default=(None, None), key=lambda item: item[0] if item[0] is not None else float("-inf"))
        best_ask = min(self._depth_asks.items(), default=(None, None), key=lambda item: item[0] if item[0] is not None else float("inf"))
        self.bid_price = best_bid[0]
        self.bid_qty = best_bid[1]
        self.ask_price = best_ask[0]
        self.ask_qty = best_ask[1]

    def _refresh_top_from_book_ticker(self) -> None:
        self.bid_price = self._book_ticker_bid_price
        self.ask_price = self._book_ticker_ask_price
        self.bid_qty = self._book_ticker_bid_qty
        self.ask_qty = self._book_ticker_ask_qty

    def _validate_depth_book(self) -> bool:
        if self.bid_price is None or self.ask_price is None:
            return False
        if self.bid_price <= 0 or self.ask_price <= 0:
            return False
        if self.bid_price >= self.ask_price:
            return False
        if self._book_ticker_bid_price is not None:
            bid_div = abs(self.bid_price - self._book_ticker_bid_price) / max(self._book_ticker_bid_price, EPSILON) * 10_000.0
            if bid_div > self.book_ticker_divergence_bps:
                return False
        if self._book_ticker_ask_price is not None:
            ask_div = abs(self.ask_price - self._book_ticker_ask_price) / max(self._book_ticker_ask_price, EPSILON) * 10_000.0
            if ask_div > self.book_ticker_divergence_bps:
                return False
        return True

    def _mark_unsynced(self, reason: str) -> None:
        self.sync_state = "unsynced"
        self.last_sync_reason = reason
        self.last_update_id = None
        self._depth_bids.clear()
        self._depth_asks.clear()
        self._refresh_top_from_book_ticker()

    def _mark_invalid(self, reason: str) -> None:
        self.sync_state = "invalid"
        self.last_sync_reason = reason
        self.last_update_id = None
        self._depth_bids.clear()
        self._depth_asks.clear()
        self._refresh_top_from_book_ticker()
