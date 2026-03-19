from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from ofbot.config import ExecutionConfig
from ofbot.execution.costs import simulate_market_order_fill
from ofbot.execution.fills import FillEvent, PaperOrder
from ofbot.market.orderbook import OrderBookState
from ofbot.strategy.base import StrategyDecision


EPSILON = 1e-12


@dataclass(slots=True)
class OrderLifecycleEvent:
    event_time: datetime
    order_id: str
    symbol: str
    stage: str
    reason: str
    order_type: str
    side: int
    requested_qty: float
    remaining_qty: float
    age_s: float


class PaperBroker:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self._pending: dict[str, list[PaperOrder]] = {}
        self._intent_timestamps: dict[tuple[str, int, str, float], datetime] = {}
        self._lifecycle_stage_counts: Counter[str] = Counter()
        self._lifecycle_reason_counts: Counter[str] = Counter()
        self._recent_lifecycle: deque[OrderLifecycleEvent] = deque(maxlen=64)
        self._last_lifecycle_state: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _intent_key(symbol: str, side: int, order_type: str, target_qty: float) -> tuple[str, int, str, float]:
        return (symbol, side, order_type, round(float(target_qty), 10))

    def clear(self) -> None:
        self._pending.clear()
        self._intent_timestamps.clear()
        self._lifecycle_stage_counts.clear()
        self._lifecycle_reason_counts.clear()
        self._recent_lifecycle.clear()
        self._last_lifecycle_state.clear()

    def queue_from_decision(
        self,
        decision: StrategyDecision,
        symbol: str,
        current_qty: float,
        reference_price: float,
        event_time: datetime,
        *,
        strategy_id: str,
    ) -> list[PaperOrder]:
        desired_qty = decision.target_qty
        if decision.side > 0:
            desired_qty = abs(desired_qty)
        elif decision.side < 0:
            desired_qty = -abs(desired_qty)
        else:
            desired_qty = 0.0

        if not self.config.allow_short and desired_qty < 0:
            return []

        delta_qty = desired_qty - current_qty
        if abs(delta_qty) < EPSILON:
            return []

        side = 1 if delta_qty > 0 else -1
        decision_order_type = decision.order_type.lower()
        requested_order_type = (
            decision_order_type
            if decision_order_type in {"market", "limit"}
            else "market"
        )
        if requested_order_type == "limit" and not self.config.allow_limit_orders:
            requested_order_type = "market"
        order_type = requested_order_type

        reference = max(reference_price, EPSILON)
        if order_type == "limit":
            offset = self.config.limit_order_book_depth_bps / 10_000.0
            if side > 0:
                limit_price = reference * (1.0 - offset)
            else:
                limit_price = reference * (1.0 + offset)
        else:
            limit_price = None

        self._expire_intents(symbol=symbol, now=event_time)

        intent_key = self._intent_key(symbol, side, order_type, desired_qty)
        if intent_key in self._intent_timestamps or self.has_equivalent_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            target_qty=desired_qty,
        ):
            return []

        order = PaperOrder(
            order_id=str(uuid4()),
            symbol=symbol,
            event_time=event_time,
            side=side,
            order_type=order_type,
            requested_qty=abs(delta_qty),
            remaining_qty=abs(delta_qty),
            reference_price=reference,
            target_qty=desired_qty,
            stop_loss_bps=decision.stop_loss_bps,
            take_profit_bps=decision.take_profit_bps,
            trailing_stop_bps=decision.trailing_stop_bps,
            max_holding_time_s=decision.max_holding_time_s,
            strategy_id=strategy_id,
            context=decision.entry_context,
            expire_time=event_time + timedelta(seconds=max(1, self.config.max_order_lifetime_s)),
            activation_time=event_time + timedelta(milliseconds=max(0, self.config.latency_ms)),
        )
        if order_type == "limit":
            order.reference_price = limit_price if limit_price is not None else reference

        self._intent_timestamps[intent_key] = event_time
        return [order]

    def _expire_intents(self, symbol: str, now: datetime) -> None:
        cutoff = now - timedelta(seconds=max(1, self.config.max_order_lifetime_s))
        self._intent_timestamps = {
            key: created_at
            for key, created_at in self._intent_timestamps.items()
            if key[0] != symbol or created_at >= cutoff
        }

    def submit(self, order: PaperOrder) -> None:
        self._intent_timestamps.pop(
            self._intent_key(order.symbol, order.side, order.order_type, order.target_qty),
            None,
        )
        self._pending.setdefault(order.symbol, []).append(order)
        self._record_lifecycle(order, stage="submitted", reason="submit", now=order.event_time)

    def has_equivalent_order(
        self,
        symbol: str,
        side: int,
        order_type: str,
        target_qty: float,
    ) -> bool:
        for order in self._pending.get(symbol, []):
            if (
                order.side == side
                and order.order_type == order_type
                and abs(order.target_qty - target_qty) < EPSILON
            ):
                return True
        return False

    @property
    def pending_order_count(self) -> int:
        return sum(len(orders) for orders in self._pending.values())

    @property
    def pending_orders_by_symbol(self) -> dict[str, int]:
        return {
            symbol: len(orders)
            for symbol, orders in self._pending.items()
            if orders
        }

    @property
    def lifecycle_stage_counts(self) -> dict[str, int]:
        return dict(self._lifecycle_stage_counts)

    @property
    def lifecycle_reason_counts(self) -> dict[str, int]:
        return dict(self._lifecycle_reason_counts)

    @property
    def last_lifecycle_event(self) -> dict[str, float | int | str] | None:
        if not self._recent_lifecycle:
            return None
        event = self._recent_lifecycle[-1]
        return {
            "event_time": event.event_time.isoformat(),
            "order_id": event.order_id,
            "symbol": event.symbol,
            "stage": event.stage,
            "reason": event.reason,
            "order_type": event.order_type,
            "side": event.side,
            "requested_qty": round(event.requested_qty, 8),
            "remaining_qty": round(event.remaining_qty, 8),
            "age_s": round(event.age_s, 3),
        }

    def process(
        self,
        *,
        symbol: str,
        now: datetime,
        book: OrderBookState | None,
    ) -> list[FillEvent]:
        if symbol not in self._pending:
            return []

        next_batch: list[PaperOrder] = []
        fills: list[FillEvent] = []

        for order in self._pending[symbol]:
            if order.expire_time is not None and now > order.expire_time:
                self._record_lifecycle(order, stage="dropped", reason="expired", now=now)
                continue
            if order.activation_time is not None and now < order.activation_time:
                self._record_lifecycle(order, stage="pending", reason="activation_wait", now=now)
                next_batch.append(order)
                continue

            fill, reason = self._simulate_fill(order=order, now=now, book=book)
            if fill is None:
                self._record_lifecycle(order, stage="pending", reason=reason, now=now)
                next_batch.append(order)
                continue

            fill_reason = "partial_fill" if fill.fill_qty + EPSILON < order.remaining_qty else "full_fill"
            self._record_lifecycle(order, stage="filled", reason=fill_reason, now=now, force=True)
            fills.append(fill)
            if fill.fill_qty + EPSILON < order.remaining_qty:
                order.remaining_qty -= fill.fill_qty
                self._record_lifecycle(order, stage="pending", reason="partial_fill_remaining", now=now, force=True)
                next_batch.append(order)

        self._pending[symbol] = next_batch
        return fills

    def _simulate_fill(
        self,
        *,
        order: PaperOrder,
        now: datetime,
        book: OrderBookState | None,
    ) -> tuple[FillEvent | None, str]:
        if book is None:
            return None, "no_book"
        bid = book.bid_price
        ask = book.ask_price
        bid_qty = book.bid_qty
        ask_qty = book.ask_qty
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None, "no_top_of_book"

        spread = book.spread_bps if book.spread_bps is not None else self.config.base_spread_bps
        is_maker = order.order_type == "limit"
        if is_maker and not self._limit_is_touched(order, bid=bid, ask=ask):
            return None, "limit_not_touched"

        if is_maker:
            max_fill_qty = max(
                0.0,
                (ask_qty if order.side > 0 else bid_qty) * self.config.market_participation_cap,
            )
            fill_qty = min(order.remaining_qty, max_fill_qty)
            if fill_qty <= 0.0:
                return None, "maker_no_queue_liquidity"
            price = order.reference_price
            fee_bps = self.config.maker_fee_bps
            slippage_bps = self.config.fallback_half_spread_bps
        else:
            market_fill = simulate_market_order_fill(
                self.config,
                book=book,
                side=order.side,
                requested_qty=order.remaining_qty,
            )
            fill_qty = market_fill.requested_qty
            if fill_qty <= 0.0 or market_fill.fill_price is None:
                return None, "market_unpriced"
            price = market_fill.fill_price
            fee_bps = self.config.taker_fee_bps
            slippage_bps = market_fill.slippage_bps

        fee = fill_qty * price * fee_bps / 10_000.0
        return (
            FillEvent(
                symbol=order.symbol,
                event_time=now,
                side=order.side,
                fill_qty=fill_qty,
                fill_price=price,
                fee=fee,
                slippage_bps=slippage_bps,
                is_maker=is_maker,
                notional=fill_qty * price,
                strategy_id=order.strategy_id,
                context=dict(order.context),
                order_id=order.order_id,
                stop_loss_bps=order.stop_loss_bps,
                take_profit_bps=order.take_profit_bps,
                trailing_stop_bps=order.trailing_stop_bps,
                max_holding_time_s=order.max_holding_time_s,
                spread_bps=spread,
            ),
            "filled",
        )

    def _limit_is_touched(self, order: PaperOrder, *, bid: float, ask: float) -> bool:
        if order.is_buy:
            return ask <= order.reference_price
        return bid >= order.reference_price

    def _record_lifecycle(
        self,
        order: PaperOrder,
        *,
        stage: str,
        reason: str,
        now: datetime,
        force: bool = False,
    ) -> None:
        state = (stage, reason)
        if not force and self._last_lifecycle_state.get(order.order_id) == state:
            return
        self._last_lifecycle_state[order.order_id] = state
        age_s = max(0.0, (now - order.event_time).total_seconds())
        event = OrderLifecycleEvent(
            event_time=now,
            order_id=order.order_id,
            symbol=order.symbol,
            stage=stage,
            reason=reason,
            order_type=order.order_type,
            side=order.side,
            requested_qty=order.requested_qty,
            remaining_qty=order.remaining_qty,
            age_s=age_s,
        )
        self._lifecycle_stage_counts[stage] += 1
        self._lifecycle_reason_counts[reason] += 1
        self._recent_lifecycle.append(event)
