from __future__ import annotations

from typing import Any

from ofbot.config import PolicyLibraryConfig
from ofbot.memory.bandit import ThompsonPolicySelector, StrategyStats
from ofbot.memory.store import MemoryStore
from ofbot.strategy.base import Strategy
from ofbot.strategy import build_strategy


class StrategySelector:
    def __init__(
        self,
        *,
        run_id: str,
        symbol: str,
        store: MemoryStore,
        library: PolicyLibraryConfig,
        min_samples_total: int,
        min_samples_per_regime: int,
        min_improvement_bps: float,
        confidence: float,
        rolling_window: int,
        max_drawdown_bps: float,
    ) -> None:
        self.run_id = run_id
        self.symbol = symbol
        self.store = store
        self.library = library
        self.instances: dict[str, Strategy] = {}
        self.candidates: list[str] = []
        for name in [library.continuation.name, library.exhaustion.name, library.hybrid.name]:
            cfg = getattr(library, name)
            if cfg.enabled:
                self.instances[name] = build_strategy(cfg, long_only=True)
                self.candidates.append(name)
        if not self.candidates:
            self.candidates = [library.default]
            self.instances[self.candidates[0]] = build_strategy(
                getattr(library, library.default),
                long_only=True,
            )
        self.default = library.default
        self.bandit = ThompsonPolicySelector(
            store=store,
            run_id=run_id,
            symbol=symbol,
            min_samples_total=min_samples_total,
            min_samples_per_regime=min_samples_per_regime,
            min_improvement_bps=min_improvement_bps,
            rolling_window=rolling_window,
            confidence=confidence,
            max_drawdown_bps=max_drawdown_bps,
        )
        self.incumbent_by_regime = {regime: self.default for regime in []}

    def choose(self, *, regime_key: str) -> tuple[str, Strategy]:
        selected_name, _ = self.bandit.select(
            default_strategy=self.default,
            regime=regime_key,
            candidate_strategies=self.candidates,
            incumbent_by_regime=self.incumbent_by_regime,
            allow_promotion=True,
        )
        self.incumbent_by_regime.setdefault(regime_key, selected_name)
        return selected_name, self.instances[selected_name]

    def sync_from_store(self, run_id: str, symbol: str) -> None:
        rows = self.store.get_policy_state(run_id, symbol)
        for regime, incumbent, _challenger, _mean in rows.items():
            self.incumbent_by_regime[regime] = incumbent
