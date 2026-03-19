from __future__ import annotations

from ofbot.config import StrategyModeConfig
from ofbot.strategy.base import Strategy
from ofbot.strategy.continuation import ContinuationStrategy
from ofbot.strategy.exhaustion import ExhaustionFadeStrategy
from ofbot.strategy.hybrid import RegimeAwareHybridStrategy


def build_strategy(config: StrategyModeConfig, long_only: bool) -> Strategy:
    if config.name == "continuation":
        return ContinuationStrategy(**config.params)
    if config.name == "exhaustion":
        return ExhaustionFadeStrategy(**config.params)
    if config.name == "hybrid":
        return RegimeAwareHybridStrategy(**config.params)
    return ContinuationStrategy()
