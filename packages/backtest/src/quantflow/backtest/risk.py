from __future__ import annotations

from dataclasses import dataclass

from quantflow.core.config import RiskConfig
from quantflow.data_model.events import BarEvent, PortfolioState, SignalEvent, SignalSide

EPSILON = 1e-12


@dataclass(slots=True)
class _TrackedPosition:
    entry_bar_index: int
    sign: int
    best_mark_price: float


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._tracked_positions: dict[str, _TrackedPosition] = {}
        self._cooldown_until_index: dict[str, int] = {}

    def reset(self) -> None:
        self._tracked_positions.clear()
        self._cooldown_until_index.clear()

    def evaluate(
        self,
        event: BarEvent,
        state: PortfolioState,
        current_index: int,
    ) -> SignalEvent | None:
        self._prune_expired_cooldowns(current_index)
        self._sync_positions(state, current_index)
        position = state.positions.get(event.symbol)
        if position is None or abs(position.quantity) < EPSILON:
            return None

        tracked = self._tracked_positions.get(event.symbol)
        if tracked is None:
            return None

        tracked.best_mark_price = self._updated_best_mark_price(tracked, event.close_price)
        pnl_bps = self._position_pnl_bps(position.quantity, position.avg_price, event.close_price)
        best_pnl_bps = self._position_pnl_bps(
            position.quantity, position.avg_price, tracked.best_mark_price
        )
        drawdown_from_best_bps = max(best_pnl_bps - pnl_bps, 0.0)
        bars_held = current_index - tracked.entry_bar_index + 1

        if self.config.stop_loss_bps > 0.0 and pnl_bps <= -self.config.stop_loss_bps:
            return self._risk_exit_signal(
                event,
                "risk_stop_loss",
                bars_held,
                pnl_bps,
                best_pnl_bps=best_pnl_bps,
                drawdown_from_best_bps=drawdown_from_best_bps,
                current_index=current_index,
            )
        if self.config.take_profit_bps > 0.0 and pnl_bps >= self.config.take_profit_bps:
            return self._risk_exit_signal(
                event,
                "risk_take_profit",
                bars_held,
                pnl_bps,
                best_pnl_bps=best_pnl_bps,
                drawdown_from_best_bps=drawdown_from_best_bps,
            )
        if (
            self.config.trailing_stop_bps > 0.0
            and best_pnl_bps >= self.config.trailing_stop_bps
            and drawdown_from_best_bps >= self.config.trailing_stop_bps
        ):
            return self._risk_exit_signal(
                event,
                "risk_trailing_stop",
                bars_held,
                pnl_bps,
                best_pnl_bps=best_pnl_bps,
                drawdown_from_best_bps=drawdown_from_best_bps,
                current_index=current_index,
            )
        if self.config.max_holding_bars > 0 and bars_held >= self.config.max_holding_bars:
            return self._risk_exit_signal(
                event,
                "risk_max_holding_bars",
                bars_held,
                pnl_bps,
                best_pnl_bps=best_pnl_bps,
                drawdown_from_best_bps=drawdown_from_best_bps,
            )
        return None

    def filter_signal(
        self,
        signal: SignalEvent | None,
        state: PortfolioState,
        current_index: int,
    ) -> SignalEvent | None:
        if signal is None:
            return None

        self._prune_expired_cooldowns(current_index)
        if signal.target_position == 0 or not self.is_in_cooldown(signal.symbol, current_index):
            return signal

        current_position = state.positions.get(signal.symbol)
        current_qty = current_position.quantity if current_position is not None else 0.0
        if abs(current_qty) < EPSILON:
            return None

        current_target = 1 if current_qty > 0.0 else -1
        if signal.target_position == current_target:
            return signal
        return None

    def is_in_cooldown(self, symbol: str, current_index: int) -> bool:
        cooldown_until = self._cooldown_until_index.get(symbol)
        if cooldown_until is None:
            return False
        return current_index <= cooldown_until

    def _sync_positions(self, state: PortfolioState, current_index: int) -> None:
        active_symbols: set[str] = set()
        for symbol, position in state.positions.items():
            if abs(position.quantity) < EPSILON:
                continue
            active_symbols.add(symbol)
            sign = 1 if position.quantity > 0.0 else -1
            tracked = self._tracked_positions.get(symbol)
            if tracked is None or tracked.sign != sign:
                self._tracked_positions[symbol] = _TrackedPosition(
                    entry_bar_index=current_index,
                    sign=sign,
                    best_mark_price=position.avg_price,
                )

        stale_symbols = set(self._tracked_positions) - active_symbols
        for symbol in stale_symbols:
            self._tracked_positions.pop(symbol, None)

    def _prune_expired_cooldowns(self, current_index: int) -> None:
        expired = [
            symbol
            for symbol, cooldown_until in self._cooldown_until_index.items()
            if current_index > cooldown_until
        ]
        for symbol in expired:
            self._cooldown_until_index.pop(symbol, None)

    def _updated_best_mark_price(self, tracked: _TrackedPosition, mark_price: float) -> float:
        if tracked.sign > 0:
            return max(tracked.best_mark_price, mark_price)
        return min(tracked.best_mark_price, mark_price)

    def _activate_cooldown(self, symbol: str, current_index: int) -> None:
        if self.config.cooldown_bars_after_stop <= 0:
            return
        self._cooldown_until_index[symbol] = current_index + self.config.cooldown_bars_after_stop

    def _position_pnl_bps(self, quantity: float, avg_price: float, mark_price: float) -> float:
        if quantity > 0.0:
            return ((mark_price / avg_price) - 1.0) * 10_000.0
        return ((avg_price / mark_price) - 1.0) * 10_000.0

    def _risk_exit_signal(
        self,
        event: BarEvent,
        reason: str,
        bars_held: int,
        pnl_bps: float,
        *,
        best_pnl_bps: float,
        drawdown_from_best_bps: float,
        current_index: int | None = None,
    ) -> SignalEvent:
        if current_index is not None and reason in {"risk_stop_loss", "risk_trailing_stop"}:
            self._activate_cooldown(event.symbol, current_index)
        return SignalEvent(
            symbol=event.symbol,
            event_time=event.end_time,
            side=SignalSide.FLAT,
            target_position=0,
            reason=reason,
            context={
                "bars_held": bars_held,
                "pnl_bps": pnl_bps,
                "best_pnl_bps": best_pnl_bps,
                "drawdown_from_best_bps": drawdown_from_best_bps,
                "session": event.features.get("session"),
                "hour_of_day": event.features.get("hour_of_day"),
                "vol_regime": event.features.get("vol_regime"),
            },
        )
