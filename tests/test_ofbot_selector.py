from __future__ import annotations

from datetime import UTC, datetime

from ofbot.memory.bandit import ThompsonPolicySelector
from ofbot.memory.store import MemoryStore
from tests.ofbot_helpers import build_test_config


def _trade_context_time(offset: int) -> datetime:
    return datetime(2026, 1, 1, 0, 0, offset, tzinfo=UTC)


def test_selector_promotes_challenger_with_positive_evidence(tmp_path, monkeypatch) -> None:
    config = build_test_config(tmp_path)
    config.learning.min_samples_total = 2
    config.learning.min_samples_per_regime = 1
    config.learning.enable_promotion = True
    config.learning.min_improvement_bps = 1.0
    config.learning.max_drawdown_bps = 1_000.0

    selector = ThompsonPolicySelector(
        store=MemoryStore(config.learning.duckdb_path),
        run_id="test_run",
        symbol="BTCUSDT",
        min_samples_total=config.learning.min_samples_total,
        min_samples_per_regime=config.learning.min_samples_per_regime,
        min_improvement_bps=config.learning.min_improvement_bps,
        rolling_window=config.learning.rolling_window,
        confidence=config.learning.confidence,
        max_drawdown_bps=config.learning.max_drawdown_bps,
    )
    store = selector.store

    for idx, pnl in enumerate([1.0, 2.0]):
        store.insert_trade_context(
            run_id="test_run",
            symbol="BTCUSDT",
            strategy="continuation",
            regime="low|low|range|low",
            context={},
            entry_time=_trade_context_time(idx),
            exit_time=_trade_context_time(idx + 1),
            direction=1,
            qty=1.0,
            entry_price=100.0,
            exit_price=100.0 + pnl / 10_000,
            realized_pnl=pnl / 10_000,
            gross_pnl=pnl / 10_000,
            fees=0.0,
            mae_bps=0.0,
            mfe_bps=0.0,
            spread_entry_bps=1.0,
            spread_exit_bps=1.0,
            holding_seconds=1.0,
        )

    for idx, pnl in enumerate([5.0, 6.0, 7.0], start=10):
        store.insert_trade_context(
            run_id="test_run",
            symbol="BTCUSDT",
            strategy="exhaustion",
            regime="low|low|range|low",
            context={},
            entry_time=_trade_context_time(idx),
            exit_time=_trade_context_time(idx + 1),
            direction=1,
            qty=1.0,
            entry_price=100.0,
            exit_price=100.0 + pnl / 10_000,
            realized_pnl=pnl / 10_000,
            gross_pnl=pnl / 10_000,
            fees=0.0,
            mae_bps=0.0,
            mfe_bps=0.0,
            spread_entry_bps=1.0,
            spread_exit_bps=1.0,
            holding_seconds=1.0,
        )

    # deterministic posterior draws make behavior reproducible for this test
    monkeypatch.setattr(
        "ofbot.memory.bandit.random.gauss",
        lambda mean, sigma: mean,
    )

    selected, reason = selector.select(
        default_strategy="continuation",
        regime="low|low|range|low",
        candidate_strategies=["continuation", "exhaustion"],
        incumbent_by_regime={},
        allow_promotion=True,
    )

    assert selected == "exhaustion"
    assert reason == "promoted_exhaustion"


def test_selector_keeps_incumbent_without_enough_samples(tmp_path) -> None:
    config = build_test_config(tmp_path)
    config.learning.min_samples_total = 5
    selector = ThompsonPolicySelector(
        store=MemoryStore(config.learning.duckdb_path),
        run_id="test_run",
        symbol="BTCUSDT",
        min_samples_total=config.learning.min_samples_total,
        min_samples_per_regime=config.learning.min_samples_per_regime,
        min_improvement_bps=config.learning.min_improvement_bps,
        rolling_window=config.learning.rolling_window,
        confidence=config.learning.confidence,
        max_drawdown_bps=config.learning.max_drawdown_bps,
    )

    # no sufficient observations for promotion or Bayesian confidence
    selected, reason = selector.select(
        default_strategy="continuation",
        regime="low|low|range|low",
        candidate_strategies=["continuation", "exhaustion"],
        incumbent_by_regime={},
        allow_promotion=True,
    )

    assert selected == "continuation"
    assert reason in {"kept_continuation", "kept"}
