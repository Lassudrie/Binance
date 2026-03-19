from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ofbot.execution.portfolio import PositionSnapshot
from ofbot.market.features import FeatureSnapshot
from ofbot.strategy.base import StrategyDecision

EPSILON = 1e-12


class RiskManager:
    def __init__(
        self,
        *,
        stale_data_s: float,
        broken_ws_s: float,
        abnormal_spread_bps: float,
        abnormal_volatility_bps: float,
        max_position_size: float,
        max_notional: float,
        max_concurrent_exposure: int,
        cooldown_after_loss_s: int,
        daily_loss_limit: float,
        max_holding_time_s: int,
        catastrophic_stop_loss_bps: float,
        min_hold_before_soft_exit_s: int,
        stop_loss_bps: float,
        take_profit_bps: float,
        trailing_stop_bps: float,
        max_consecutive_losses: int | None = None,
    ) -> None:
        self.stale_data_s = stale_data_s
        self.broken_ws_s = broken_ws_s
        self.abnormal_spread_bps = abnormal_spread_bps
        self.abnormal_volatility_bps = abnormal_volatility_bps
        self.max_position_size = max_position_size
        self.max_notional = max_notional
        self.max_concurrent_exposure = max_concurrent_exposure
        self.cooldown_after_loss_s = cooldown_after_loss_s
        self.daily_loss_limit = daily_loss_limit
        self.max_holding_time_s = max_holding_time_s
        self.catastrophic_stop_loss_bps = catastrophic_stop_loss_bps
        self.min_hold_before_soft_exit_s = min_hold_before_soft_exit_s
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        self.trailing_stop_bps = trailing_stop_bps
        self.max_consecutive_losses = max_consecutive_losses or 99_999
        self._last_loss_cooldown_until: dict[str, datetime] = {}
        self._daily_pnl: dict[str, float] = {}
        self._consecutive_losses: dict[str, int] = {}
        self._realized_pnl_by_day: dict[datetime.date, float] = {}
        self._position_highest_bps: dict[str, float] = {}
        self._position_lowest_bps: dict[str, float] = {}

    def reset(self, symbols: set[str]) -> None:
        for symbol in symbols:
            self._position_highest_bps.pop(symbol, None)
            self._position_lowest_bps.pop(symbol, None)
        self._last_loss_cooldown_until = {}

    def pre_entry_gate(
        self,
        *,
        symbol: str,
        snapshot: FeatureSnapshot,
        portfolio_snapshot: PositionSnapshot,
        now: datetime,
        event_timestamp: datetime,
        websocket_healthy: bool,
        open_positions: int,
        total_equity: float,
        projected_position_qty: float,
        projected_notional: float,
    ) -> tuple[bool, str | None]:
        if self._is_in_cooldown(symbol, now):
            return False, "risk_cooldown"
        if not websocket_healthy:
            return False, "risk_broken_websocket"
        if event_timestamp < now - timedelta(seconds=self.stale_data_s):
            return False, "risk_stale_market_data"
        if snapshot.spread_bps is not None and snapshot.spread_bps > self.abnormal_spread_bps:
            return False, "risk_abnormal_spread"
        if snapshot.features.get("realized_volatility") is not None:
            vol = float(snapshot.features["realized_volatility"])
            if vol > self.abnormal_volatility_bps:
                return False, "risk_abnormal_volatility"
        if abs(projected_position_qty) > self.max_position_size:
            return False, "risk_position_size"
        if projected_notional > self.max_notional:
            return False, "risk_notional_limit"
        if (
            projected_notional > EPSILON
            and open_positions >= self.max_concurrent_exposure
            and abs(portfolio_snapshot.net_position) <= EPSILON
        ):
            return False, "risk_exposure_limit"

        today = now.date()
        realized = self._realized_pnl_by_day.get(today, 0.0)
        if realized < 0 and realized < -abs(total_equity) * self.daily_loss_limit:
            return False, "risk_daily_loss_limit"
        if self._consecutive_losses.get(symbol, 0) >= self.max_consecutive_losses:
            return False, "risk_max_consecutive_losses"
        return True, None

    def evaluate_position_exit(
        self,
        symbol: str,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        now: datetime,
        *,
        estimated_exit_cost_bps: float = 0.0,
        realized_pnl_delta: float = 0.0,
    ) -> StrategyDecision | None:
        if abs(state.net_position) < EPSILON:
            return None

        gross_pnl_bps = _pnl_bps(state)
        net_pnl_bps_est = gross_pnl_bps - max(0.0, estimated_exit_cost_bps)

        if symbol not in self._position_highest_bps:
            self._position_highest_bps[symbol] = net_pnl_bps_est
            self._position_lowest_bps[symbol] = net_pnl_bps_est
        self._position_highest_bps[symbol] = max(self._position_highest_bps[symbol], net_pnl_bps_est)
        self._position_lowest_bps[symbol] = min(self._position_lowest_bps[symbol], net_pnl_bps_est)

        context = state.risk_context
        stop_loss = context.get("stop_loss_bps", self.stop_loss_bps)
        take_profit = context.get("take_profit_bps", self.take_profit_bps)
        trailing = context.get("trailing_stop_bps", self.trailing_stop_bps)
        max_time = context.get("max_holding_time_s", self.max_holding_time_s)
        held = 0.0
        if state.position_open_time is not None:
            held = (now - state.position_open_time).total_seconds()

        if state.position_open_time is not None and max_time is not None:
            if held >= int(max_time):
                return _flat_decision(
                    symbol=symbol,
                    reason="risk_max_holding",
                    detail=f"held_{int(held)}s",
                )

        if gross_pnl_bps <= -float(self.catastrophic_stop_loss_bps):
            return _flat_decision(symbol, "risk_catastrophic_stop", f"gross_pnl_{gross_pnl_bps:.2f}")

        if held < float(self.min_hold_before_soft_exit_s):
            return None

        if isinstance(stop_loss, (int, float)) and net_pnl_bps_est <= -float(stop_loss):
            return _flat_decision(symbol, "risk_stop_loss", f"net_pnl_{net_pnl_bps_est:.2f}")
        if isinstance(take_profit, (int, float)) and net_pnl_bps_est >= float(take_profit):
            return _flat_decision(symbol, "risk_take_profit", f"net_pnl_{net_pnl_bps_est:.2f}")
        if isinstance(trailing, (int, float)) and trailing > 0:
            peak = self._position_highest_bps.get(symbol, net_pnl_bps_est)
            drawdown = peak - net_pnl_bps_est
            if net_pnl_bps_est >= 0 and drawdown >= float(trailing):
                return _flat_decision(symbol, "risk_trailing_stop", f"drawdown_{drawdown:.2f}")
        return None

    def clear_position_extremes(self, symbol: str) -> None:
        self._position_highest_bps.pop(symbol, None)
        self._position_lowest_bps.pop(symbol, None)

    def register_realized_pnl(self, symbol: str, trade_realized_pnl: float, at_time: datetime) -> None:
        if trade_realized_pnl < 0 and self.cooldown_after_loss_s > 0:
            self._last_loss_cooldown_until[symbol] = at_time + timedelta(
                seconds=self.cooldown_after_loss_s
            )
            self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
        else:
            self._consecutive_losses[symbol] = 0
        today = at_time.date()
        self._realized_pnl_by_day[today] = self._realized_pnl_by_day.get(today, 0.0) + trade_realized_pnl

    def _is_in_cooldown(self, symbol: str, now: datetime) -> bool:
        until = self._last_loss_cooldown_until.get(symbol)
        return until is not None and now <= until


def _pnl_bps(state: PositionSnapshot) -> float:
    if abs(state.avg_entry_price) <= EPSILON:
        return 0.0
    if state.net_position > 0:
        if state.last_price is None:
            return 0.0
        return (state.last_price / state.avg_entry_price - 1.0) * 10_000.0
    return (state.avg_entry_price / max(state.last_price or state.avg_entry_price, EPSILON) - 1.0) * 10_000.0


def _flat_decision(symbol: str, reason: str, detail: str) -> StrategyDecision:
    return StrategyDecision(
        symbol=symbol,
        side=0,
        target_qty=0.0,
        reason=reason,
        confidence=1.0,
        entry_context={"exit_reason": detail},
    )
