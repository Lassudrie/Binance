from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from ofbot import alpha_loop as alpha_loop_mod
from ofbot.config import dump_config_yaml
from ofbot.memory.store import MemoryStore
from tests.ofbot_helpers import build_test_config


def test_live_canary_promotes_and_deploys_after_positive_live_window(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config_path = tmp_path / "config" / "local.paper.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_config_yaml(config))
    store = MemoryStore(config.learning.duckdb_path)
    try:
        store.upsert_learning_candidate(
            candidate_id="cand_a",
            source_run_id="run_source",
            strategy="continuation",
            archetype="stricter_entry",
            status="promotable",
            base_config_hash="hash",
            overlay_yaml_path=str(tmp_path / "overlay.yaml"),
            params={"no_trade_z": 1.35},
            rationale={"type": "candidate"},
        )

        active = alpha_loop_mod.ensure_live_canary_candidate(
            store=store,
            base_config_path=config_path,
        )
        assert active is not None
        assert active["status"] == "live_canary"
        active_config_path = Path(active["active_config_path"])
        assert active_config_path.exists()
        assert alpha_loop_mod.active_paper_config_path(config).exists()

        store.upsert_run_review(
            run_id="run_live",
            verdict="positive",
            config_snapshot_path=None,
            config_snapshot_hash=None,
            raw_input_path=None,
            report_dir=None,
            summary={"trade_count": 20, "net_pnl": 5.0},
            notes={},
        )

        status = alpha_loop_mod.record_live_canary_result(
            store=store,
            base_config_path=config_path,
            candidate_id="cand_a",
            run_id="run_live",
            active_config_path=active_config_path,
        )
        refreshed = alpha_loop_mod.runtime_candidate_rows(store)[0]
    finally:
        store.close()

    assert status == "deployed_local"
    assert refreshed["status"] == "deployed_local"
    assert alpha_loop_mod.active_paper_config_path(config).exists()


def test_live_canary_rejects_candidate_on_critical_risk(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config_path = tmp_path / "config" / "local.paper.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_config_yaml(config))
    store = MemoryStore(config.learning.duckdb_path)
    try:
        store.upsert_learning_candidate(
            candidate_id="cand_b",
            source_run_id="run_source",
            strategy="continuation",
            archetype="stricter_entry",
            status="promotable",
            base_config_hash="hash",
            overlay_yaml_path=str(tmp_path / "overlay.yaml"),
            params={"no_trade_z": 1.35},
            rationale={"type": "candidate"},
        )
        active = alpha_loop_mod.ensure_live_canary_candidate(
            store=store,
            base_config_path=config_path,
        )
        assert active is not None
        active_config_path = Path(active["active_config_path"])

        store.upsert_run_review(
            run_id="run_risk",
            verdict="negative",
            config_snapshot_path=None,
            config_snapshot_hash=None,
            raw_input_path=None,
            report_dir=None,
            summary={"trade_count": 1, "net_pnl": -1.0},
            notes={},
        )
        store.insert_event_journal(
            run_id="run_risk",
            event_time=datetime.now(UTC),
            symbol="BTCUSDT",
            action="signal",
            strategy="risk",
            order_intent="risk_daily_loss_limit",
            reason="risk_gate",
            params={},
        )

        status = alpha_loop_mod.record_live_canary_result(
            store=store,
            base_config_path=config_path,
            candidate_id="cand_b",
            run_id="run_risk",
            active_config_path=active_config_path,
        )
        refreshed = alpha_loop_mod.runtime_candidate_rows(store)[0]
    finally:
        store.close()

    assert status == "rejected_live"
    assert refreshed["status"] == "rejected_live"
    assert not alpha_loop_mod.active_paper_config_path(config).exists()


def test_run_offline_research_cycle_persists_validated_edge(tmp_path, monkeypatch) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    store = MemoryStore(config.learning.duckdb_path)
    repo_root = Path(__file__).resolve().parents[1]
    suite_config_path = repo_root / "configs" / "order_flow_suite_spot_hunt.toml"
    deep_config_path = repo_root / "configs" / "candidate_order_flow_hunt.toml"
    report_root = tmp_path / "offline_report"
    report_root.mkdir(parents=True, exist_ok=True)
    try:
        monkeypatch.setattr(
            "ofbot.alpha_loop._load_order_flow_suite_features",
            lambda config, symbols, dataset_type: {},
        )
        monkeypatch.setattr(
            "ofbot.alpha_loop.run_order_flow_suite",
            lambda feature_frames, config: SimpleNamespace(
                ranked_candidates=[
                    {
                        "strategy_name": "cumdelta_reversion_v1",
                        "bar_size": "15s",
                        "execution_mode": "passive",
                        "params": {"entry_z": 1.0, "exit_z": 0.1, "vol_regime_min": 0.0},
                        "walkforward_mean_test_net_pnl": 1.2,
                        "positive_test_window_ratio": 0.7,
                        "execution_fragile": False,
                    }
                ]
            ),
        )
        monkeypatch.setattr(
            "ofbot.alpha_loop._load_candidate_deep_dive_features",
            lambda config, symbols, dataset_type: None,
        )
        monkeypatch.setattr(
            "ofbot.alpha_loop.run_candidate_deep_dive",
            lambda features, config: SimpleNamespace(
                aggregate={
                    "final_verdict": "validated_edge",
                    "best_walkforward_mean_test_net_pnl": 1.4,
                },
                best_scenario={"scenario_name": "baseline"},
            ),
        )
        monkeypatch.setattr(
            "ofbot.alpha_loop.write_candidate_deep_dive_report",
            lambda result, reports_dir, run_name: report_root / run_name / "report.md",
        )

        result = alpha_loop_mod.run_offline_research_cycle(
            store=store,
            suite_config_path=suite_config_path,
            deep_dive_config_path=deep_config_path,
        )
        persisted = store.list_offline_research_candidates(limit=4)
    finally:
        store.close()

    assert result["status"] == "validated_edge"
    assert any(row["status"] == "validated_edge" for row in persisted)
