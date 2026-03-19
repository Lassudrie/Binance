from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from statistics import mean
from typing import Any

from ofbot.memory.store import MemoryStore


@dataclass(slots=True)
class StrategyStats:
    strategy: str
    regime: str
    sample_count: int
    mean: float
    variance: float


class ThompsonPolicySelector:
    def __init__(
        self,
        *,
        store: MemoryStore,
        run_id: str,
        symbol: str,
        min_samples_total: int,
        min_samples_per_regime: int,
        min_improvement_bps: float,
        rolling_window: int,
        confidence: float,
        max_drawdown_bps: float,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.symbol = symbol
        self.min_samples_total = min_samples_total
        self.min_samples_per_regime = min_samples_per_regime
        self.min_improvement_bps = min_improvement_bps
        self.rolling_window = rolling_window
        self.confidence = confidence
        self.max_drawdown_bps = max_drawdown_bps
        self._history: dict[str, deque[float]] = {}

    def select(
        self,
        *,
        default_strategy: str,
        regime: str,
        candidate_strategies: list[str],
        incumbent_by_regime: dict[str, str],
        allow_promotion: bool,
    ) -> tuple[str, str]:
        incumbent = incumbent_by_regime.get(regime, default_strategy)
        incumbent_stats = self._compute_strategy_stats(incumbent, regime)
        if incumbent_stats.sample_count < self.min_samples_total:
            return incumbent, f"kept_{incumbent}"
        candidate_scores: dict[str, float] = {}
        for strategy in candidate_strategies:
            stats = self._compute_strategy_stats(strategy, regime)
            candidate_scores[strategy] = self._posterior_sample(stats, symbol=self.symbol)

        if not allow_promotion or not candidate_scores:
            return incumbent, f"kept_{incumbent}"

        best_strategy = max(candidate_scores, key=candidate_scores.get)
        best_score = candidate_scores[best_strategy]
        incumbent_score = candidate_scores.get(incumbent, incumbent_stats.mean)
        if best_strategy != incumbent and self._promotion_gate(incumbent_stats, self._compute_strategy_stats(best_strategy, regime)):
            self.store.upsert_policy_state(
                run_id=self.run_id,
                symbol=self.symbol,
                regime=regime,
                incumbent=best_strategy,
                challenger=incumbent,
                incumbent_mean=self._compute_strategy_stats(best_strategy, regime).mean,
                challenger_mean=incumbent_stats.mean,
                reason=f"promoted_{best_strategy}_over_{incumbent}_score_{best_score:.3f}",
            )
            return best_strategy, f"promoted_{best_strategy}"

        if best_strategy != incumbent and best_score > incumbent_score:
            return best_strategy, "sampled"
        return incumbent, "kept"

    def _promotion_gate(self, incumbent: StrategyStats, challenger: StrategyStats) -> bool:
        if challenger.sample_count < self.min_samples_per_regime:
            return False
        if incumbent.sample_count < self.min_samples_total:
            return False
        if challenger.mean < incumbent.mean - self.max_drawdown_bps / 10_000.0:
            return False
        if challenger.mean - incumbent.mean < self.min_improvement_bps / 10_000.0:
            return False
        if challenger.sample_count < 2:
            return False
        if challenger.variance < 0:
            return False
        return True

    def _compute_strategy_stats(self, strategy: str, regime: str) -> StrategyStats:
        rows = self.store.read_performance(
            run_id=self.run_id,
            strategy=strategy,
            symbol=self.symbol,
            regime=regime,
        )
        realized = [float(row[3]) for row in rows]
        if not realized:
            return StrategyStats(strategy=strategy, regime=regime, sample_count=0, mean=0.0, variance=0.0)
        sample = realized[-self.rolling_window :]
        mu = mean(sample)
        variance = (
            sum((value - mu) ** 2 for value in sample) / max(len(sample), 1)
        )
        stats = StrategyStats(
            strategy=strategy,
            regime=regime,
            sample_count=len(sample),
            mean=mu,
            variance=variance / max(len(sample), 1),
        )
        self.store.update_policy_stats(
            run_id=self.run_id,
            strategy=strategy,
            symbol=self.symbol,
            regime=regime,
            sample_count=stats.sample_count,
            mean=stats.mean,
            variance=stats.variance,
        )
        return stats

    def _posterior_sample(self, stats: StrategyStats, symbol: str) -> float:
        if stats.sample_count < self.min_samples_total:
            return random.gauss(0.0, 1.0)
        sigma = stats.variance**0.5 / max(1, stats.sample_count**0.5)
        return random.gauss(stats.mean, max(sigma, 1e-9))

    def write_policy_state_summary(self) -> None:
        # no-op placeholder for compatibility with reporting pipeline
        pass
