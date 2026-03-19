from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ofbot.cli._main import _run_note_trade
from ofbot.memory.learning import advance_learning_cycle, get_source_candidates, validate_pending_candidates
from ofbot.memory.store import MemoryStore
from tests.ofbot_helpers import build_test_config


def _seed_negative_run(store: MemoryStore, *, run_id: str, now: datetime) -> int:
    store.insert_event_journal(
        run_id=run_id,
        event_time=now,
        symbol="BTCUSDT",
        action="signal",
        strategy="continuation",
        params={"expected_net_edge_bps": 3.0, "estimated_entry_cost_bps": 1.5},
        regime="low|tight|trend_up|high",
        order_intent="submit",
        reason="continuation_long",
    )
    store.insert_event_journal(
        run_id=run_id,
        event_time=now + timedelta(seconds=12),
        symbol="BTCUSDT",
        action="signal",
        strategy="risk",
        params={"expected_net_edge_bps": -1.0},
        regime="low|tight|trend_up|high",
        order_intent="submit",
        reason="risk_stop_loss",
    )
    store.insert_trade_context(
        run_id=run_id,
        symbol="BTCUSDT",
        strategy="continuation",
        regime="low|tight|trend_up|high",
        context={"source": "test"},
        entry_time=now,
        exit_time=now + timedelta(seconds=12),
        direction=1,
        qty=1.0,
        entry_price=100.0,
        exit_price=99.8,
        realized_pnl=-1.5,
        gross_pnl=-1.2,
        fees=0.3,
        mae_bps=-25.0,
        mfe_bps=12.0,
        spread_entry_bps=1.0,
        spread_exit_bps=1.0,
        holding_seconds=12.0,
        entry_fees=0.15,
        exit_fees=0.15,
        entry_fill_count=1,
        exit_fill_count=1,
        entry_notional=100.0,
        exit_notional=99.8,
        closed_qty=1.0,
    )
    return 1


def test_advance_learning_cycle_generates_overview_and_candidate(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    store = MemoryStore(config.learning.duckdb_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        _seed_negative_run(store, run_id="run_a", now=now)
        overview_path = advance_learning_cycle(
            store=store,
            config=config,
            run_id="run_a",
            raw_input_path=None,
            report_dir=config.paths.reports_dir / "run_a" / "report_run_a",
            validate_pending=False,
            generate_candidates=True,
        )
    finally:
        store.close()

    assert overview_path.exists()
    payload = json.loads((overview_path.parent / "overview.json").read_text())
    assert payload["run_summary"]["trade_count"] == 1
    assert any(item["lesson_type"] == "stricter_entry" for item in payload["lessons"])
    assert len(payload["candidate_queue"]) == 1
    candidate = payload["candidate_queue"][0]
    assert candidate["strategy"] == "continuation"
    assert Path(candidate["overlay_yaml_path"]).exists()
    assert (overview_path.parent / "candidates" / f"{candidate['candidate_id']}.yaml").exists()


def test_validate_pending_candidates_promotes_after_three_oos_runs(tmp_path, monkeypatch) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    store = MemoryStore(config.learning.duckdb_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    raw_input_path = tmp_path / "validation.parquet"
    raw_input_path.write_text("placeholder")
    try:
        _seed_negative_run(store, run_id="run_source", now=now)
        advance_learning_cycle(
            store=store,
            config=config,
            run_id="run_source",
            raw_input_path=None,
            report_dir=config.paths.reports_dir / "run_source" / "report_run_source",
            validate_pending=False,
            generate_candidates=True,
        )
        candidate_id = get_source_candidates(store, "run_source")[0]["candidate_id"]

        monkeypatch.setattr("ofbot.replay.player.iter_replay_events", lambda path: [])

        def _fake_run_replay_metrics(*, config, events):
            params = config.strategy_library.continuation.params
            if params:
                return {
                    "trade_count": 8,
                    "net_pnl": 2.5,
                    "mean_holding_seconds": 10.0,
                    "max_drawdown": 0.4,
                }
            return {
                "trade_count": 8,
                "net_pnl": 1.0,
                "mean_holding_seconds": 10.0,
                "max_drawdown": 0.6,
            }

        monkeypatch.setattr("ofbot.memory.learning.run_replay_metrics", _fake_run_replay_metrics)

        for index in range(3):
            updates = validate_pending_candidates(
                store=store,
                validation_run_id=f"run_val_{index}",
                raw_input_path=raw_input_path,
            )
            assert len(updates) == 1

        refreshed = get_source_candidates(store, "run_source")[0]
    finally:
        store.close()

    assert refreshed["candidate_id"] == candidate_id
    assert refreshed["status"] == "promotable"
    assert refreshed["validation_aggregate"]["validation_count"] == 3


def test_validate_pending_candidates_stays_pending_when_improved_but_under_sample_gate(tmp_path, monkeypatch) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    store = MemoryStore(config.learning.duckdb_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    raw_input_path = tmp_path / "validation.parquet"
    raw_input_path.write_text("placeholder")
    try:
        _seed_negative_run(store, run_id="run_source", now=now)
        advance_learning_cycle(
            store=store,
            config=config,
            run_id="run_source",
            raw_input_path=None,
            report_dir=config.paths.reports_dir / "run_source" / "report_run_source",
            validate_pending=False,
            generate_candidates=True,
        )
        candidate_id = get_source_candidates(store, "run_source")[0]["candidate_id"]

        monkeypatch.setattr("ofbot.replay.player.iter_replay_events", lambda path: [])

        def _fake_run_replay_metrics(*, config, events):
            params = config.strategy_library.continuation.params
            if params:
                return {
                    "trade_count": 1,
                    "net_pnl": -0.5,
                    "mean_holding_seconds": 10.0,
                    "max_drawdown": 0.4,
                }
            return {
                "trade_count": 1,
                "net_pnl": -1.0,
                "mean_holding_seconds": 10.0,
                "max_drawdown": 0.6,
            }

        monkeypatch.setattr("ofbot.memory.learning.run_replay_metrics", _fake_run_replay_metrics)

        for index in range(3):
            updates = validate_pending_candidates(
                store=store,
                validation_run_id=f"run_val_{index}",
                raw_input_path=raw_input_path,
            )
            assert len(updates) == 1

        refreshed = get_source_candidates(store, "run_source")[0]
    finally:
        store.close()

    assert refreshed["candidate_id"] == candidate_id
    assert refreshed["status"] == "pending"
    assert refreshed["validation_aggregate"]["validation_count"] == 3


def test_validate_pending_candidates_rejects_after_three_worse_oos_runs(tmp_path, monkeypatch) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    store = MemoryStore(config.learning.duckdb_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    raw_input_path = tmp_path / "validation.parquet"
    raw_input_path.write_text("placeholder")
    try:
        _seed_negative_run(store, run_id="run_source", now=now)
        advance_learning_cycle(
            store=store,
            config=config,
            run_id="run_source",
            raw_input_path=None,
            report_dir=config.paths.reports_dir / "run_source" / "report_run_source",
            validate_pending=False,
            generate_candidates=True,
        )

        monkeypatch.setattr("ofbot.replay.player.iter_replay_events", lambda path: [])

        def _fake_run_replay_metrics(*, config, events):
            params = config.strategy_library.continuation.params
            if params:
                return {
                    "trade_count": 1,
                    "net_pnl": -1.5,
                    "mean_holding_seconds": 10.0,
                    "max_drawdown": 0.7,
                }
            return {
                "trade_count": 1,
                "net_pnl": -1.0,
                "mean_holding_seconds": 10.0,
                "max_drawdown": 0.6,
            }

        monkeypatch.setattr("ofbot.memory.learning.run_replay_metrics", _fake_run_replay_metrics)

        for index in range(3):
            updates = validate_pending_candidates(
                store=store,
                validation_run_id=f"run_worse_{index}",
                raw_input_path=raw_input_path,
            )
            assert len(updates) == 1

        refreshed = get_source_candidates(store, "run_source")[0]
    finally:
        store.close()

    assert refreshed["status"] == "rejected"
    assert refreshed["validation_aggregate"]["validation_count"] == 3


def test_run_note_trade_appends_manual_note_and_refreshes_overview(tmp_path, monkeypatch) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    store = MemoryStore(config.learning.duckdb_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        trade_id = _seed_negative_run(store, run_id="run_note", now=now)
        advance_learning_cycle(
            store=store,
            config=config,
            run_id="run_note",
            raw_input_path=None,
            report_dir=config.paths.reports_dir / "run_note" / "report_run_note",
            validate_pending=False,
            generate_candidates=True,
        )
    finally:
        store.close()

    monkeypatch.setattr("ofbot.cli._main.load_config", lambda path: config)

    _run_note_trade(
        "ignored.yaml",
        run_id="run_note",
        trade_id=trade_id,
        tag="postmortem",
        note="exit trop tardive",
    )

    store = MemoryStore(config.learning.duckdb_path)
    try:
        notes = store.run_query_df(
            "SELECT * FROM manual_notes WHERE run_id=? ORDER BY id ASC",
            ["run_note"],
        )
    finally:
        store.close()

    overview_path = config.paths.reports_dir / "run_note" / "report_run_note" / "learning" / "overview.md"
    assert len(notes.index) == 1
    assert overview_path.exists()
    assert "exit trop tardive" in overview_path.read_text()
