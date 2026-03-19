from __future__ import annotations

from quantflow.core.config import StrategyConfig
from quantflow.strategy.base import Strategy
from quantflow.strategy.baselines import NoTradeBaselineStrategy, RandomBaselineStrategy
from quantflow.strategy.orderflow import (
    BurstBreakoutConfirmationStrategy,
    CumDeltaReversionV1Strategy,
    CumulativeDeltaBreakoutStrategy,
    CumulativeDeltaMeanReversionStrategy,
    DeltaContinuationStrategy,
    DeltaImpulseContinuationV1Strategy,
    ExhaustionMeanReversionStrategy,
    ImbalanceBurstExhaustionV1Strategy,
    Spot10mRegimePullbackStrategy,
    TradeImbalanceMomentumStrategy,
)


def build_strategy(strategy_config: StrategyConfig) -> Strategy:
    name = strategy_config.name
    params = strategy_config.params

    if name == "cumulative_delta_breakout":
        return CumulativeDeltaBreakoutStrategy(**params)
    if name == "cumdelta_reversion_v1":
        return CumDeltaReversionV1Strategy(**params)
    if name == "delta_impulse_continuation_v1":
        return DeltaImpulseContinuationV1Strategy(**params)
    if name == "imbalance_burst_exhaustion_v1":
        return ImbalanceBurstExhaustionV1Strategy(**params)
    if name == "spot_10m_regime_pullback":
        return Spot10mRegimePullbackStrategy(**params)
    if name == "cumulative_delta_mean_reversion":
        return CumulativeDeltaMeanReversionStrategy(**params)
    if name == "delta_continuation":
        return DeltaContinuationStrategy(**params)
    if name == "trade_imbalance_momentum":
        return TradeImbalanceMomentumStrategy(**params)
    if name == "burst_breakout_confirmation":
        return BurstBreakoutConfirmationStrategy(**params)
    if name == "exhaustion_mean_reversion":
        return ExhaustionMeanReversionStrategy(**params)
    if name == "no_trade_baseline":
        return NoTradeBaselineStrategy()
    if name == "random_baseline":
        return RandomBaselineStrategy(**params)
    raise ValueError(f"Unsupported strategy: {name}")
