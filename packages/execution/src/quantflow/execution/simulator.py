from __future__ import annotations

from dataclasses import dataclass, replace

from quantflow.core.config import ExecutionConfig
from quantflow.data_model.events import BarEvent, FillEvent, OrderEvent, OrderSide, OrderType

EPSILON = 1e-12


@dataclass(slots=True)
class PendingOrder:
    symbol: str
    activation_index: int
    expiration_index: int
    order: OrderEvent


class ExecutionSimulator:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self.pending_orders: list[PendingOrder] = []

    def reset(self) -> None:
        self.pending_orders.clear()

    def submit_orders(self, orders: list[OrderEvent], current_index: int) -> None:
        activation_index = current_index + max(1, self.config.latency_bars)
        expiration_index = activation_index + max(1, self.config.max_order_lifetime_bars) - 1
        for order in orders:
            self.pending_orders.append(
                PendingOrder(
                    symbol=order.symbol,
                    activation_index=activation_index,
                    expiration_index=expiration_index,
                    order=order,
                )
            )

    def has_pending_equivalent_order(self, order: OrderEvent) -> bool:
        for pending in self.pending_orders:
            candidate = pending.order
            if (
                candidate.symbol == order.symbol
                and candidate.side == order.side
                and candidate.order_type == order.order_type
                and candidate.target_position == order.target_position
            ):
                return True
        return False

    def process_bar(self, bar: BarEvent, current_index: int) -> list[FillEvent]:
        fills: list[FillEvent] = []
        still_pending: list[PendingOrder] = []

        for pending in self.pending_orders:
            if pending.symbol != bar.symbol:
                still_pending.append(pending)
                continue
            if pending.activation_index > current_index:
                still_pending.append(pending)
                continue
            if current_index > pending.expiration_index:
                continue

            fill, remaining_order = self._simulate_fill(pending.order, bar)
            if fill is not None:
                fills.append(fill)
            if remaining_order is not None:
                if current_index >= pending.expiration_index:
                    continue
                still_pending.append(
                    PendingOrder(
                        symbol=remaining_order.symbol,
                        activation_index=current_index + 1,
                        expiration_index=pending.expiration_index,
                        order=remaining_order,
                    )
                )

        self.pending_orders = still_pending
        return fills

    def _simulate_fill(
        self, order: OrderEvent, bar: BarEvent
    ) -> tuple[FillEvent | None, OrderEvent | None]:
        if order.order_type == OrderType.MARKET:
            fill_quantity = min(
                order.quantity,
                self._max_fill_quantity(bar.volume, self.config.market_participation_cap),
            )
            if fill_quantity <= 0.0:
                return None, order
            participation_ratio = fill_quantity / max(bar.volume, EPSILON)
            price = self._market_fill_price(bar.open_price, order.side, participation_ratio)
            fee = price * fill_quantity * self.config.taker_fee_bps / 10_000.0
            fill = FillEvent(
                symbol=order.symbol,
                event_time=bar.start_time,
                side=order.side,
                order_type=order.order_type,
                quantity=fill_quantity,
                price=price,
                fee=fee,
                slippage_bps=self._market_fill_bps(participation_ratio),
                is_maker=False,
                target_position=order.target_position,
                signal_reason=order.signal_reason,
                context=order.context,
            )
            if fill_quantity >= order.quantity:
                return fill, None
            return fill, replace(order, quantity=order.quantity - fill_quantity)

        if order.limit_price is None:
            raise ValueError("Limit order requires limit_price")

        touched = (order.side == OrderSide.BUY and bar.low_price <= order.limit_price) or (
            order.side == OrderSide.SELL and bar.high_price >= order.limit_price
        )
        if not touched:
            return None, order

        max_fill_quantity = self._max_fill_quantity(bar.volume, self.config.limit_participation_cap)
        fill_quantity = min(order.quantity, max_fill_quantity)
        if fill_quantity <= 0.0:
            return None, order

        if order.side == OrderSide.BUY:
            price = min(order.limit_price, bar.open_price)
        else:
            price = max(order.limit_price, bar.open_price)

        fee = price * fill_quantity * self.config.maker_fee_bps / 10_000.0
        fill = FillEvent(
            symbol=order.symbol,
            event_time=bar.start_time,
            side=order.side,
            order_type=order.order_type,
            quantity=fill_quantity,
            price=price,
            fee=fee,
            slippage_bps=0.0,
            is_maker=True,
            target_position=order.target_position,
            signal_reason=order.signal_reason,
            context=order.context,
        )

        if fill_quantity >= order.quantity:
            return fill, None

        return fill, replace(order, quantity=order.quantity - fill_quantity)

    def _market_fill_bps(self, participation_ratio: float) -> float:
        impact_bps = self.config.impact_bps_per_unit_participation * participation_ratio
        return self.config.slippage_bps + self.config.fallback_half_spread_bps + impact_bps

    def _market_fill_price(
        self,
        reference_price: float,
        side: OrderSide,
        participation_ratio: float,
    ) -> float:
        total_bps = self._market_fill_bps(participation_ratio)
        multiplier = 1.0 + (total_bps / 10_000.0)
        if side == OrderSide.BUY:
            return reference_price * multiplier
        return reference_price / multiplier

    def build_market_fill(
        self,
        *,
        symbol: str,
        event_time,
        side: OrderSide,
        quantity: float,
        reference_price: float,
        target_position: int,
        signal_reason: str | None,
        context: dict[str, float | int | str | None],
        participation_ratio: float = 1.0,
    ) -> FillEvent:
        price = self._market_fill_price(reference_price, side, participation_ratio)
        fee = price * quantity * self.config.taker_fee_bps / 10_000.0
        return FillEvent(
            symbol=symbol,
            event_time=event_time,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            fee=fee,
            slippage_bps=self._market_fill_bps(participation_ratio),
            is_maker=False,
            target_position=target_position,
            signal_reason=signal_reason,
            context=context,
        )

    def _max_fill_quantity(self, bar_volume: float, participation_cap: float) -> float:
        return max(bar_volume * max(participation_cap, 0.0), 0.0)
