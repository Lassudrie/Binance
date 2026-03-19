from __future__ import annotations

from ofbot.market.features import FeatureSnapshot
from ofbot.market.regimes import RegimeLabel
from ofbot.strategy.base import Strategy, StrategyDecision
from ofbot.strategy.continuation import ContinuationStrategy
from ofbot.strategy.exhaustion import ExhaustionFadeStrategy
from ofbot.execution.portfolio import PositionSnapshot


def _feature(snapshot: FeatureSnapshot, name: str, default: float = 0.0) -> float:
    value = snapshot.features.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    return default


class RegimeAwareHybridStrategy(Strategy):
    """Hybrid selector between continuation and exhaustion with regime conditioning."""

    def __init__(
        self,
        *,
        regime_confidence_threshold: float = 0.55,
        continuation_label: str = "continuation",
        exhaustion_label: str = "exhaustion",
        time_exit_s: int = 600,
        spread_bps_max: float = 200.0,
        volatility_bps_max: float = 750.0,
        stop_loss_bps: float = 65.0,
        take_profit_bps: float = 120.0,
        trailing_stop_bps: float = 28.0,
    ) -> None:
        super().__init__("hybrid")
        self.regime_confidence_threshold = regime_confidence_threshold
        self.continuation_label = continuation_label
        self.exhaustion_label = exhaustion_label
        self.time_exit_s = time_exit_s
        self.spread_bps_max = spread_bps_max
        self.volatility_bps_max = volatility_bps_max
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        self.trailing_stop_bps = trailing_stop_bps
        self.continuation = ContinuationStrategy(
            spread_bps_max=spread_bps_max,
            volatility_bps_max=volatility_bps_max,
            max_holding_time_s=time_exit_s,
            stop_loss_bps=stop_loss_bps,
            take_profit_bps=take_profit_bps,
            trailing_stop_bps=trailing_stop_bps,
        )
        self.exhaustion = ExhaustionFadeStrategy(
            spread_bps_max=spread_bps_max,
            volatility_bps_max=volatility_bps_max,
            max_holding_time_s=time_exit_s,
            stop_loss_bps=stop_loss_bps,
            take_profit_bps=take_profit_bps,
            trailing_stop_bps=trailing_stop_bps,
        )

    def decide(
        self,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        *,
        long_only: bool,
    ) -> StrategyDecision | None:
        spread = snapshot.spread_bps
        volatility = _feature(snapshot, "realized_volatility")
        regime = snapshot.regime
        if spread is not None and spread > self.spread_bps_max:
            return self._exit_if_needed("risk_spread", snapshot, state.net_position, "spread_block")
        if volatility is not None and volatility > self.volatility_bps_max:
            return self._exit_if_needed("risk_vol", snapshot, state.net_position, "volatility_block")

        flow_confidence = abs(_feature(snapshot, "cvd_base_1s_z"))
        burst_score = abs(_feature(snapshot, "burst_intensity_5s_z"))
        trend_bias = regime.trend

        if trend_bias.startswith("trend") and flow_confidence >= self.regime_confidence_threshold:
            return self.continuation.decide(snapshot, state, long_only=long_only)

        if (
            burst_score >= self.regime_confidence_threshold
            and regime.volatility in {"high", "neutral"}
        ):
            return self.exhaustion.decide(snapshot, state, long_only=long_only)

        if state.net_position != 0:
            # no-trade regime in flat market: hold with tighter time exit
            return self._time_exit_if_needed(snapshot, state, long_only=long_only)

        return None

    def _time_exit_if_needed(
        self,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        *,
        long_only: bool,
    ) -> StrategyDecision | None:
        if not state.position_open_time:
            return None
        held_sec = (snapshot.event_time - state.position_open_time).total_seconds()
        if held_sec >= self.time_exit_s:
            return StrategyDecision(
                symbol=snapshot.symbol,
                side=0,
                target_qty=0.0,
                reason="hybrid_time_exit",
                confidence=0.2,
                entry_context={"strategy": self.name, "exit_reason": "time_exit"},
            )
        return None

    def _exit_if_needed(
        self,
        reason: str,
        snapshot: FeatureSnapshot,
        current: int,
        detail: str,
    ) -> StrategyDecision | None:
        if current == 0:
            return None
        return StrategyDecision(
            symbol=snapshot.symbol,
            side=0,
            target_qty=0.0,
            reason=reason,
            confidence=0.1,
            entry_context={"strategy": self.name, "exit_reason": detail},
        )

    def reset(self) -> None:
        self.continuation.reset()
        self.exhaustion.reset()

    def get_parameters(self) -> dict[str, float | int | str | bool]:
        return {
            "regime_confidence_threshold": self.regime_confidence_threshold,
            "continuation_label": self.continuation_label,
            "exhaustion_label": self.exhaustion_label,
            "time_exit_s": self.time_exit_s,
            "spread_bps_max": self.spread_bps_max,
            "volatility_bps_max": self.volatility_bps_max,
            "stop_loss_bps": self.stop_loss_bps,
            "take_profit_bps": self.take_profit_bps,
            "trailing_stop_bps": self.trailing_stop_bps,
        }
