from __future__ import annotations

from datetime import datetime

from ofbot.maintenance import cleanup_runtime_artifacts, write_live_learning_status
from ofbot.memory.store import MemoryStore
from tests.ofbot_helpers import build_test_config


def test_cleanup_runtime_artifacts_keeps_recent_and_pending_candidate_sources(tmp_path) -> None:
    config = build_test_config(tmp_path)
    config.housekeeping.keep_recent_run_raw = 1
    config.housekeeping.keep_recent_report_csv = 1

    store = MemoryStore(config.learning.duckdb_path)
    try:
        for run_id in ("run_old", "run_pending", "run_recent"):
            raw_path = config.paths.raw_dir / f"raw_{run_id}.parquet"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(b"x" * 32)

            report_dir = config.paths.reports_dir / run_id / f"report_{run_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "events.csv").write_text("event\n")
            (report_dir / "trades.csv").write_text("trade\n")

            store.upsert_run_review(
                run_id=run_id,
                verdict="flat",
                config_snapshot_path=None,
                config_snapshot_hash=None,
                raw_input_path=str(raw_path),
                report_dir=str(report_dir),
                summary={"run_id": run_id, "trade_count": 0, "net_pnl": 0.0},
                notes={},
            )

        store._conn.execute(
            "UPDATE run_reviews SET updated_at=? WHERE run_id='run_old'",
            [datetime(2026, 3, 19, 10, 0, 0)],
        )
        store._conn.execute(
            "UPDATE run_reviews SET updated_at=? WHERE run_id='run_pending'",
            [datetime(2026, 3, 19, 11, 0, 0)],
        )
        store._conn.execute(
            "UPDATE run_reviews SET updated_at=? WHERE run_id='run_recent'",
            [datetime(2026, 3, 19, 12, 0, 0)],
        )

        overlay_path = config.paths.memory_dir / "learning_candidates" / "pending.yaml"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text("strategy: continuation\n")
        store.upsert_learning_candidate(
            candidate_id="candidate_pending",
            source_run_id="run_pending",
            strategy="continuation",
            archetype="stricter_entry",
            status="pending",
            base_config_hash="hash",
            overlay_yaml_path=str(overlay_path),
            params={"min_expected_net_edge_bps": 3.0},
            rationale={},
        )

        tmp_pytest = config.paths.root / ".tmp_pytest"
        tmp_pytest.mkdir(parents=True, exist_ok=True)
        (tmp_pytest / "tmp.txt").write_text("tmp\n")

        summary = cleanup_runtime_artifacts(config=config, store=store)
    finally:
        store.close()

    assert summary["freed_bytes"] > 0
    assert not (config.paths.raw_dir / "raw_run_old.parquet").exists()
    assert (config.paths.raw_dir / "raw_run_recent.parquet").exists()
    assert (config.paths.raw_dir / "raw_run_pending.parquet").exists()
    assert not (config.paths.reports_dir / "run_old" / "report_run_old" / "events.csv").exists()
    assert not (config.paths.reports_dir / "run_old" / "report_run_old" / "trades.csv").exists()
    assert (config.paths.reports_dir / "run_recent" / "report_run_recent" / "events.csv").exists()
    assert (config.paths.reports_dir / "run_pending" / "report_run_pending" / "events.csv").exists()
    assert not (config.paths.root / ".tmp_pytest").exists()


def test_write_live_learning_status_writes_markdown_snapshot(tmp_path) -> None:
    config = build_test_config(tmp_path)
    output_path = tmp_path / "docs" / "status.md"

    store = MemoryStore(config.learning.duckdb_path)
    try:
        report_dir = config.paths.reports_dir / "run_source" / "report_run_source"
        report_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = config.paths.memory_dir / "learning_candidates" / "candidate.yaml"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text("strategy: continuation\n")

        store.upsert_run_review(
            run_id="run_source",
            verdict="negative",
            config_snapshot_path=None,
            config_snapshot_hash="hash",
            raw_input_path=None,
            report_dir=str(report_dir),
            summary={
                "run_id": "run_source",
                "trade_count": 2,
                "net_pnl": -4.5,
                "dominant_strategy": "continuation",
                "expected_edge_blocks": 12,
            },
            notes={},
        )
        store.upsert_learning_candidate(
            candidate_id="candidate_alpha",
            source_run_id="run_source",
            strategy="continuation",
            archetype="stricter_entry",
            status="pending",
            base_config_hash="hash",
            overlay_yaml_path=str(overlay_path),
            params={"min_expected_net_edge_bps": 3.0},
            rationale={},
        )
        store.upsert_candidate_validation(
            candidate_id="candidate_alpha",
            validation_run_id="run_validation",
            baseline_net_pnl=-3.0,
            candidate_net_pnl=-1.0,
            baseline_trade_count=2,
            candidate_trade_count=1,
            baseline_max_drawdown=0.0,
            candidate_max_drawdown=0.0,
            verdict="improved",
            summary={},
        )

        written_path = write_live_learning_status(
            config=config,
            store=store,
            output_path=output_path,
        )
    finally:
        store.close()

    assert written_path == output_path
    text = output_path.read_text()
    assert "# Live Learning Status" in text
    assert "run_source" in text
    assert "candidate_alpha" in text
    assert "Alpha status" in text
