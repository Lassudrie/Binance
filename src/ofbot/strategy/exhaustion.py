from __future__ import annotations

from ofbot.market.features import FeatureSnapshot
from ofbot.market.regimes import RegimeLabel
from ofbot.strategy.base import Strategy, StrategyDecision
from ofbot.execution.portfolio import PositionSnapshot


def _feature(snapshot: FeatureSnapshot, name: str, default: float = 0.0) -> float:
    value = snapshot.features.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _regime(snapshot: FeatureSnapshot) -> RegimeLabel:
    return snapshot.regime


class ExhaustionFadeStrategy(Strategy):
    """Fade variant for bursty, extreme flow with weak follow-through."""

    def __init__(
        self,
        *,
        burst_threshold: float = 2.0,
        extreme_flow_threshold: float = 1.8,
        decay_momentum_threshold: float = 0.35,
        spread_bps_max: float = 220.0,
        volatility_bps_max: float = 900.0,
        max_holding_time_s: int = 360,
        stop_loss_bps: float = 55.0,
        take_profit_bps: float = 100.0,
        trailing_stop_bps: float = 30.0,
        no_trade_z: float = 0.6,
        min_burst_z: float = 1.0,
    ) -> None:
        super().__init__("exhaustion")
        self.burst_threshold = burst_threshold
        self.extreme_flow_threshold = extreme_flow_threshold
        self.decay_momentum_threshold = decay_momentum_threshold
        self.spread_bps_max = spread_bps_max
        self.volatility_bps_max = volatility_bps_max
        self.max_holding_time_s = max_holding_time_s
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        self.trailing_stop_bps = trailing_stop_bps
        self.no_trade_z = no_trade_z
        self.min_burst_z = min_burst_z

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
            else -1 if state.net_position < 0 else 0
        )
        spread_bps = snapshot.spread_bps
        volatility = _feature(snapshot, "realized_volatility", default=0.0)
        if spread_bps is not None and spread_bps > self.spread_bps_max:
            return self._exit_if_needed(current, snapshot, "exhaustion_spread_gate", "spread_block")
        if volatility > self.volatility_bps_max:
            return self._exit_if_needed(current, snapshot, "exhaustion_vol_gate", "volatility_block")

        cvd = _feature(snapshot, "cvd_base_5s")
        cvd_z = _feature(snapshot, "cvd_base_5s_z")
        burst = _feature(snapshot, "burst_intensity_5s")
        burst_z = _feature(snapshot, "burst_intensity_5s_z")
        momentum = _feature(snapshot, "short_return_bps_5s", default=_feature(snapshot, "momentum_bps_1s"))
        # strict condition: strong burst but momentum weakens
        extreme_flow = abs(cvd_z) >= self.extreme_flow_threshold and cvd_z ** 2 > 0
        burst_condition = burst >= self.burst_threshold and burst_z >= self.min_burst_z
        decay_condition = abs(momentum) < self.decay_momentum_threshold

        context = {
            "strategy": self.name,
            "cvd_base_5s": cvd,
            "cvd_base_5s_z": cvd_z,
            "burst_intensity_5s": burst,
            "burst_intensity_5s_z": burst_z,
            "momentum_bps_5s": momentum,
            "regime": str(_regime(snapshot).as_dict()),
        }

        regime = _regime(snapshot)
        if regime.volatility == "high" and abs(momentum) < self.decay_momentum_threshold:
            return self._exit_if_needed(current, snapshot, "exhaustion_regime_filter", "high_volatility")

        if long_only:
            if burst_condition and extreme_flow and decay_condition and cvd_z > self.no_trade_z:
                # in long-only, only fade upward burst with opposite pressure
                return StrategyDecision(
                    symbol=snapshot.symbol,
                    side=0,
                    target_qty=0.0,
                    reason="exhaustion_fade_long_only",
                    confidence=min(1.0, burst_z / 3.0),
                    order_type="market",
                    max_holding_time_s=self.max_holding_time_s,
                    stop_loss_bps=self.stop_loss_bps,
                    take_profit_bps=self.take_profit_bps,
                    trailing_stop_bps=self.trailing_stop_bps,
                    entry_context=context,
                )
            if abs(cvd_z) < self.no_trade_z and current != 0:
                return self._exit_if_needed(current, snapshot, "exhaustion_flat", "flow_normalized")
            return None

        if burst_condition and extreme_flow and decay_condition:
            side = -1 if cvd_z > 0 else 1
            if current != side and (side != 0):
                return StrategyDecision(
                    symbol=snapshot.symbol,
                    side=side,
                    target_qty=1.0,
                    reason="exhaustion_fade",
                    confidence=min(1.0, burst_z / 3.0),
                    order_type="market",
                    max_holding_time_s=self.max_holding_time_s,
                    stop_loss_bps=self.stop_loss_bps,
                    take_profit_bps=self.take_profit_bps,
                    trailing_stop_bps=self.trailing_stop_bps,
                    entry_context=context,
                )
        if abs(cvd_z) < self.no_trade_z and current != 0:
            return self._exit_if_needed(current, snapshot, "exhaustion_exit", "flow_decay")

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
        return StrategyDecision(
            symbol=snapshot.symbol,
            side=0,
            target_qty=0.0,
            reason=reason,
            confidence=0.0,
            entry_context={"strategy": self.name, "exit_reason": detail},
        )

    def reset(self) -> None:
        return

    def get_parameters(self) -> dict[str, float | int | str | bool]:
        return {
            "burst_threshold": self.burst_threshold,
            "extreme_flow_threshold": self.extreme_flow_threshold,
            "decay_momentum_threshold": self.decay_momentum_threshold,
            "spread_bps_max": self.spread_bps_max,
            "volatility_bps_max": self.volatility_bps_max,
            "max_holding_time_s": self.max_holding_time_s,
            "stop_loss_bps": self.stop_loss_bps,
            "take_profit_bps": self.take_profit_bps,
            "trailing_stop_bps": self.trailing_stop_bps,
            "no_trade_z": self.no_trade_z,
            "min_burst_z": self.min_burst_z,
        }
