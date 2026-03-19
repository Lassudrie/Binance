from __future__ import annotations

import json
from datetime import UTC, datetime

from ofbot.market.features import FeatureSnapshot
from ofbot.market.regimes import RegimeLabel
from ofbot.memory.journal import JournalWriter
from ofbot.memory.store import MemoryStore
from tests.ofbot_helpers import build_test_config


def test_journal_compacts_feature_snapshot_for_signals(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    store = MemoryStore(config.learning.duckdb_path)
    try:
        journal = JournalWriter(store=store, run_id="run_compact")
        snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=datetime(2026, 1, 1, tzinfo=UTC),
            features={
                "queue_imbalance": 0.99,
                "cvd_base_1s": 4.2,
                "cvd_base_1s_z": 1.7,
                "microprice_drift_bps_1s": 3.4,
                "momentum_bps_1s": 3.0,
                "momentum_bps_5s": 4.1,
                "realized_volatility": 12.0,
                "regime_volatility": "high",
                "regime_spread": "low",
                "regime_trend": "trend_up",
                "regime_flow": "high",
                "very_large_unused_blob": "x" * 2048,
            },
            regime=RegimeLabel(volatility="high", spread="low", trend="trend_up", flow="high"),
            spread_bps=1.2,
            mid_price=100.0,
            microprice=100.1,
        )

        journal.append_signal(
            snapshot=snapshot,
            strategy_id="continuation",
            order_intent="submit",
            signal_strength=0.8,
            rationale="test",
            params={"expected_net_edge_bps": 3.2},
        )

        stored = store._conn.execute(
            "SELECT feature_snapshot FROM event_journal WHERE run_id='run_compact'"
        ).fetchone()[0]
    finally:
        store.close()

    payload = json.loads(stored)
    assert payload["feature_count"] == 12
    assert payload["spread_bps"] == 1.2
    assert payload["mid_price"] == 100.0
    assert payload["queue_imbalance"] == 0.99
    assert "very_large_unused_blob" not in payload
