from __future__ import annotations

from datetime import datetime
from typing import Any

from ofbot.market.features import FeatureSnapshot
from ofbot.market.regimes import RegimeLabel
from ofbot.strategy.base import Strategy, StrategyDecision
from ofbot.execution.portfolio import PositionSnapshot


def _feature(snapshot: FeatureSnapshot, name: str, *, default: float = 0.0) -> float:
    value = snapshot.features.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _extract_regime(snapshot: FeatureSnapshot) -> RegimeLabel:
    return snapshot.regime


class ContinuationStrategy(Strategy):
    """Continuation strategy on aligned order flow + queue imbalance + microprice drift."""

    def __init__(
        self,
        *,
        trend_alignment_threshold: float = 0.08,
        queue_imbalance_threshold: float = 0.04,
        microprice_drift_threshold_bps: float = 2.5,
        min_expected_net_edge_bps: float = 3.0,
        edge_fast_horizon_s: int = 15,
        edge_slow_horizon_s: int = 30,
        edge_fast_weight: float = 0.6,
        edge_slow_weight: float = 0.4,
        edge_cvd_bonus_bps: float = 1.5,
        edge_queue_bonus_bps: float = 4.0,
        edge_bonus_cap_bps: float = 12.0,
        min_momentum_5s_bps: float = 0.8,
        spread_bps_max: float = 180.0,
        volatility_bps_max: float = 700.0,
        notrade_z: float = 0.15,
        no_trade_z: float | None = None,
        max_holding_time_s: int = 420,
        min_hold_before_discretionary_exit_s: int = 20,
        reverse_exit_cvd_z: float = 1.0,
        reverse_exit_queue_imbalance: float = 0.85,
        reverse_exit_microprice_drift_bps: float = 0.1,
        stop_loss_bps: float = 70.0,
        take_profit_bps: float = 150.0,
        trailing_stop_bps: float = 35.0,
    ) -> None:
        super().__init__("continuation")
        self.trend_alignment_threshold = trend_alignment_threshold
        self.queue_imbalance_threshold = queue_imbalance_threshold
        self.microprice_drift_threshold_bps = microprice_drift_threshold_bps
        self.min_expected_net_edge_bps = min_expected_net_edge_bps
        self.edge_fast_horizon_s = max(1, int(edge_fast_horizon_s))
        self.edge_slow_horizon_s = max(self.edge_fast_horizon_s, int(edge_slow_horizon_s))
        weight_sum = max(float(edge_fast_weight) + float(edge_slow_weight), 1e-12)
        self.edge_fast_weight = float(edge_fast_weight) / weight_sum
        self.edge_slow_weight = float(edge_slow_weight) / weight_sum
        self.edge_cvd_bonus_bps = max(0.0, float(edge_cvd_bonus_bps))
        self.edge_queue_bonus_bps = max(0.0, float(edge_queue_bonus_bps))
        self.edge_bonus_cap_bps = max(0.0, float(edge_bonus_cap_bps))
        self.min_momentum_5s_bps = min_momentum_5s_bps
        self.spread_bps_max = spread_bps_max
        self.volatility_bps_max = volatility_bps_max
        self.no_trade_z = notrade_z if no_trade_z is None else no_trade_z
        self.max_holding_time_s = max_holding_time_s
        self.min_hold_before_discretionary_exit_s = min_hold_before_discretionary_exit_s
        self.reverse_exit_cvd_z = reverse_exit_cvd_z
        self.reverse_exit_queue_imbalance = reverse_exit_queue_imbalance
        self.reverse_exit_microprice_drift_bps = reverse_exit_microprice_drift_bps
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        self.trailing_stop_bps = trailing_stop_bps

    def decide(
        self,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        *,
        long_only: bool,
    ) -> StrategyDecision | None:
        current = (
            1
            if state.net_position > 0
            else -1
            if state.net_position < 0
            else 0
        )

        spread_bps = snapshot.spread_bps
        volatility = _feature(snapshot, "realized_volatility")
        if spread_bps is not None and spread_bps > self.spread_bps_max:
            return self._exit_if_needed(current, snapshot, "risk_spread_gate", "spread_blocked")
        if volatility is not None and volatility > self.volatility_bps_max:
            return self._exit_if_needed(current, snapshot, "risk_volatility_gate", "volatility_blocked")

        cvd = _feature(snapshot, "cvd_base_1s", default=0.0)
        cvd_z = _feature(snapshot, "cvd_base_1s_z")
        queue_imb = _feature(snapshot, "queue_imbalance")
        micro_drift = _feature(
            snapshot,
            "microprice_drift_bps_1s",
            default=_feature(snapshot, "microprice_drift_bps"),
        )
        momentum = _feature(snapshot, "momentum_bps_1s", default=_feature(snapshot, "short_return_bps_1s"))
        momentum_5s = _feature(
            snapshot,
            "momentum_bps_5s",
            default=_feature(snapshot, "short_return_bps_5s"),
        )

        aligned = (
            cvd > self.trend_alignment_threshold
            and cvd_z >= self.no_trade_z
            and queue_imb > self.queue_imbalance_threshold
            and micro_drift > self.microprice_drift_threshold_bps
            and momentum > self.trend_alignment_threshold
            and momentum_5s >= self.min_momentum_5s_bps
        )
        reverse = (
            cvd_z <= -self.reverse_exit_cvd_z
            and queue_imb <= -self.reverse_exit_queue_imbalance
            and micro_drift <= -self.reverse_exit_microprice_drift_bps
        )
        regime = _extract_regime(snapshot)
        context: dict[str, float | int | str | None] = {
            "strategy": self.name,
            "cvd_base_1s": cvd,
            "cvd_base_1s_z": cvd_z,
            "queue_imbalance": queue_imb,
            "microprice_drift_bps_1s": micro_drift,
            "momentum_bps_1s": momentum,
            "momentum_bps_5s": momentum_5s,
            "min_expected_net_edge_bps": self.min_expected_net_edge_bps,
            "regime": str(regime.as_dict()),
        }

        if long_only:
            if current > 0 and self._held_long_enough(snapshot.event_time, state.position_open_time) and reverse:
                return self._flat_decision(
                    snapshot=snapshot,
                    reason="continuation_reverse_exit",
                    detail="reverse_confirmed",
                )
            if aligned and current <= 0:
                return StrategyDecision(
                    symbol=snapshot.symbol,
                    side=1,
                    target_qty=1.0,
                    reason="continuation_long",
                    confidence=min(1.0, abs(cvd_z) / 3.0),
                    order_type="market",
                    max_holding_time_s=self.max_holding_time_s,
                    stop_loss_bps=self.stop_loss_bps,
                    take_profit_bps=self.take_profit_bps,
                    trailing_stop_bps=self.trailing_stop_bps,
                    entry_context=context,
                )
            return None

        if current != 0:
            direction = 1 if current > 0 else -1
            directional_reverse = reverse if direction > 0 else aligned
            if self._held_long_enough(snapshot.event_time, state.position_open_time) and directional_reverse:
                return self._flat_decision(
                    snapshot=snapshot,
                    reason="continuation_reverse_exit",
                    detail="reverse_confirmed",
                )
            return None

        if aligned:
            if current <= 0:
                return StrategyDecision(
                    symbol=snapshot.symbol,
                    side=1,
                    target_qty=1.0,
                    reason="continuation_flow",
                    confidence=min(1.0, abs(cvd_z) / 3.0),
                    order_type="market",
                    max_holding_time_s=self.max_holding_time_s,
                    stop_loss_bps=self.stop_loss_bps,
                    take_profit_bps=self.take_profit_bps,
                    trailing_stop_bps=self.trailing_stop_bps,
                    entry_context=context,
                )
            return None
        if reverse:
            return StrategyDecision(
                symbol=snapshot.symbol,
                side=-1,
                target_qty=1.0,
                reason="continuation_flow",
                confidence=min(1.0, abs(cvd_z) / 3.0),
                order_type="market",
                max_holding_time_s=self.max_holding_time_s,
                stop_loss_bps=self.stop_loss_bps,
                take_profit_bps=self.take_profit_bps,
                trailing_stop_bps=self.trailing_stop_bps,
                entry_context=context,
            )
        return None

    def _exit_if_needed(
        self,
        current: int,
        snapshot: FeatureSnapshot,
        reason: str,
        detail: str,
    ) -> StrategyDecision | None:
        if current == 0:
            return None
        return self._flat_decision(snapshot=snapshot, reason=reason, detail=detail)

    def _flat_decision(
        self,
        *,
        snapshot: FeatureSnapshot,
        reason: str,
        detail: str,
    ) -> StrategyDecision:
        return StrategyDecision(
            symbol=snapshot.symbol,
            side=0,
            target_qty=0.0,
            reason=reason,
            confidence=0.2,
            entry_context={"exit_reason": detail, "strategy": self.name},
        )

    def _held_long_enough(
        self,
        now: datetime,
        opened_at: datetime | None,
    ) -> bool:
        if opened_at is None:
            return False
        return (now - opened_at).total_seconds() >= self.min_hold_before_discretionary_exit_s

    def reset(self) -> None:
        return

    def get_parameters(self) -> dict[str, float | int | str | bool]:
        return {
            "trend_alignment_threshold": self.trend_alignment_threshold,
            "queue_imbalance_threshold": self.queue_imbalance_threshold,
            "microprice_drift_threshold_bps": self.microprice_drift_threshold_bps,
            "min_expected_net_edge_bps": self.min_expected_net_edge_bps,
            "edge_fast_horizon_s": self.edge_fast_horizon_s,
            "edge_slow_horizon_s": self.edge_slow_horizon_s,
            "edge_fast_weight": self.edge_fast_weight,
            "edge_slow_weight": self.edge_slow_weight,
            "edge_cvd_bonus_bps": self.edge_cvd_bonus_bps,
            "edge_queue_bonus_bps": self.edge_queue_bonus_bps,
            "edge_bonus_cap_bps": self.edge_bonus_cap_bps,
            "min_momentum_5s_bps": self.min_momentum_5s_bps,
            "spread_bps_max": self.spread_bps_max,
            "volatility_bps_max": self.volatility_bps_max,
            "no_trade_z": self.no_trade_z,
            "max_holding_time_s": self.max_holding_time_s,
            "min_hold_before_discretionary_exit_s": self.min_hold_before_discretionary_exit_s,
            "reverse_exit_cvd_z": self.reverse_exit_cvd_z,
            "reverse_exit_queue_imbalance": self.reverse_exit_queue_imbalance,
            "reverse_exit_microprice_drift_bps": self.reverse_exit_microprice_drift_bps,
            "stop_loss_bps": self.stop_loss_bps,
            "take_profit_bps": self.take_profit_bps,
            "trailing_stop_bps": self.trailing_stop_bps,
        }
