from __future__ import annotations

from quantflow.data_model.events import (
    OrderEvent,
    OrderSide,
    OrderType,
    PortfolioState,
    SignalEvent,
)


class SignalToOrderTranslator:
    def __init__(
        self,
        position_size: float,
        allow_short: bool,
        risk_per_trade_fraction: float = 0.0,
        use_limit_orders: bool = False,
        limit_offset_bps: float = 0.0,
    ) -> None:
        self.position_size = position_size
        self.allow_short = allow_short
        self.risk_per_trade_fraction = risk_per_trade_fraction
        self.use_limit_orders = use_limit_orders
        self.limit_offset_bps = limit_offset_bps

    def from_signal(
        self,
        signal: SignalEvent,
        state: PortfolioState,
        reference_price: float,
    ) -> list[OrderEvent]:
        desired_position = signal.target_position
        if not self.allow_short:
            desired_position = max(desired_position, 0)

        current_qty = (
            state.positions.get(signal.symbol).quantity if signal.symbol in state.positions else 0.0
        )
        target_qty = desired_position * self._position_size(signal, state)
        delta_qty = target_qty - current_qty

        if abs(delta_qty) < 1e-12:
            return []

        side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
        order_type = OrderType.LIMIT if self.use_limit_orders else OrderType.MARKET
        limit_price = None
        if order_type == OrderType.LIMIT:
            offset_multiplier = self.limit_offset_bps / 10_000.0
            if side == OrderSide.BUY:
                limit_price = reference_price * (1.0 - offset_multiplier)
            else:
                limit_price = reference_price * (1.0 + offset_multiplier)
        return [
            OrderEvent(
                symbol=signal.symbol,
                created_time=signal.event_time,
                activation_time=signal.event_time,
                side=side,
                order_type=order_type,
                quantity=abs(delta_qty),
                target_position=desired_position,
                limit_price=limit_price,
                signal_reason=signal.reason,
                context=signal.context,
            )
        ]

    def _position_size(self, signal: SignalEvent, state: PortfolioState) -> float:
        stop_distance_raw = signal.context.get("stop_distance")
        if self.risk_per_trade_fraction <= 0.0 or stop_distance_raw is None:
            return self.position_size

        if not isinstance(stop_distance_raw, (int, float)):
            return self.position_size

        stop_distance = float(stop_distance_raw)
        if stop_distance <= 0.0:
            return self.position_size

        risk_amount = state.equity * self.risk_per_trade_fraction
        signal.context["risk_amount"] = risk_amount
        return risk_amount / stop_distance
