from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import polars as pl

from ofbot.config import dump_config_yaml, load_config
from ofbot.research.backtest import ReplayBacktestResult
from ofbot.research.campaign import run_paper_research_campaign
from ofbot.research.config import load_research_config
from ofbot.research.promotion import promote_candidate_to_paper, rollback_active_paper_candidate
from ofbot.research.registry import ResearchRegistry
from ofbot.research.runner import run_research_cycle
from tests.ofbot_helpers import build_test_config


def _write_replay_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 3, 21, 0, 0, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(120):
        event_time = start + timedelta(seconds=index)
        rows.append(
            {
                "sequence": index + 1,
                "run_id": "test_run",
                "event_type": "bookTicker",
                "symbol": "BTCUSDT",
                "event_time": event_time,
                "received_at": event_time,
                "raw_json": "{}",
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _write_research_config(path: Path, raw_input_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "strategy_name: continuation",
                "paths:",
                "  registry_dir: data/research/registry",
                "  artifacts_dir: data/research/runs",
                "  deployment_dir: data/research/deployments",
                "  active_paper_config_path: data/research/deployments/active.paper.yaml",
                "dataset:",
                f"  raw_input_path: {raw_input_path}",
                "  train_ratio: 0.5",
                "  validation_ratio: 0.25",
                "  test_ratio: 0.25",
                "  stability_subperiods: 3",
                "  walk_forward_windows: 2",
                "  walk_forward_train_ratio: 0.6",
                "  min_unique_timestamps: 20",
                "generation:",
                "  method: grid",
                "  max_variants: 4",
                "  top_k_train: 2",
                "  top_k_validation: 1",
                "scoring:",
                "  min_trades: 5",
                "  pnl_scale: 4.0",
                "  drawdown_scale: 6.0",
                "  expectancy_scale: 0.25",
                "  max_drawdown_abs: 8.0",
                "  min_profit_factor: 0.9",
                "  min_win_rate: 0.45",
                "robustness:",
                "  min_subperiod_positive_ratio: 0.5",
                "  min_walk_forward_positive_ratio: 0.5",
                "  min_stressed_positive_ratio: 0.5",
                "  min_neighbor_positive_ratio: 0.5",
                "  stressed_maker_fee_multipliers: [1.0, 1.25]",
                "  stressed_taker_fee_multipliers: [1.0, 1.25]",
                "  stressed_impact_multipliers: [1.0]",
                "  stressed_latency_ms_add: [0]",
                "selection:",
                "  min_validation_score_delta: 0.01",
                "  min_test_score_delta: 0.0",
                "  max_drawdown_increase_abs: 2.0",
                "  require_test_improvement: true",
                "  paper_window_minutes: 30",
                "param_spaces:",
                "  min_expected_net_edge_bps:",
                "    values: [2.5, 3.0, 3.5]",
            ]
        ),
        encoding="utf-8",
    )


def _fake_backtest_runner(base_config, strategy_name, params, frame, split_name, execution_overrides):
    edge = float(params.get("min_expected_net_edge_bps", 3.0) or 3.0)
    stress_penalty = 0.0
    if execution_overrides:
        stress_penalty += max(0.0, float(execution_overrides.get("taker_fee_bps", base_config.execution.taker_fee_bps)) - base_config.execution.taker_fee_bps) * 0.15
        stress_penalty += max(0.0, float(execution_overrides.get("maker_fee_bps", base_config.execution.maker_fee_bps)) - base_config.execution.maker_fee_bps) * 0.10
    split_bonus = 0.0
    if "train" in split_name:
        split_bonus = 0.30
    elif "validation" in split_name:
        split_bonus = 0.20
    elif "test" in split_name:
        split_bonus = 0.10
    net_pnl = 4.5 - abs(edge - 3.5) * 3.5 + split_bonus - stress_penalty
    if edge <= 2.5:
        net_pnl -= 2.5
    trade_count = 10
    max_drawdown = 1.5 + abs(edge - 3.5) * 1.2 + stress_penalty * 0.2
    win_rate = 0.52 + max(0.0, edge - 3.0) * 0.08 - stress_penalty * 0.01
    profit_factor = 1.05 + max(0.0, edge - 3.0) * 0.40 - stress_penalty * 0.02
    expectancy = net_pnl / trade_count
    sharpe_like = 1.0 + net_pnl / 6.0
    return ReplayBacktestResult(
        strategy_name=strategy_name,
        params=dict(params),
        split_name=split_name,
        metrics={
            "trade_count": trade_count,
            "net_pnl": float(net_pnl),
            "gross_pnl": float(net_pnl + 0.5),
            "total_fees": 0.5,
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "sharpe_like": float(sharpe_like),
            "average_holding_seconds": 12.0,
            "expected_edge_blocks": 3,
            "signal_submit_count": 12,
        },
    )


def test_research_cycle_validates_candidate_and_records_registry(tmp_path: Path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.default = "continuation"  # type: ignore[assignment]
    config.strategy_library.continuation.enabled = True
    config.strategy_library.continuation.params = {
        "min_expected_net_edge_bps": 3.0,
        "no_trade_z": 1.35,
    }
    config_path = tmp_path / "config" / "local.paper.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_config_yaml(config))

    raw_input_path = tmp_path / "raw_input.parquet"
    _write_replay_frame(raw_input_path)
    research_config_path = tmp_path / "config" / "research.yaml"
    _write_research_config(research_config_path, raw_input_path)

    result = run_research_cycle(
        base_config_path=config_path,
        research_config_path=research_config_path,
        backtest_runner=_fake_backtest_runner,
    )

    summary = result["summary"]
    assert summary["status"] == "validated_for_paper"
    assert summary["selected_candidate"] is not None
    assert summary["selected_candidate"]["params"]["min_expected_net_edge_bps"] == 3.5
    assert result["report_path"].exists()

    research_config = load_research_config(research_config_path)
    registry = ResearchRegistry(research_config.paths.registry_dir)
    candidates = registry.list_candidates(status="validated_for_paper")

    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == summary["selected_candidate"]["candidate_id"]


def test_promote_and_rollback_candidate_write_paper_configs(tmp_path: Path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.default = "continuation"  # type: ignore[assignment]
    config.strategy_library.continuation.enabled = True
    config.strategy_library.continuation.params = {
        "min_expected_net_edge_bps": 3.0,
        "no_trade_z": 1.35,
    }
    config_path = tmp_path / "config" / "local.paper.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_config_yaml(config))

    raw_input_path = tmp_path / "raw_input.parquet"
    _write_replay_frame(raw_input_path)
    research_config_path = tmp_path / "config" / "research.yaml"
    _write_research_config(research_config_path, raw_input_path)
    research_config = load_research_config(research_config_path)
    registry = ResearchRegistry(research_config.paths.registry_dir)
    registry.record_candidate(
        {
            "candidate_id": "cand_test",
            "research_run_id": "research_20260321_000000",
            "strategy_name": "continuation",
            "params": {"min_expected_net_edge_bps": 3.5, "no_trade_z": 1.35},
            "baseline_params": {"min_expected_net_edge_bps": 3.0, "no_trade_z": 1.35},
            "status": "validated_for_paper",
        }
    )

    deployed_path = promote_candidate_to_paper(
        base_config_path=config_path,
        registry=registry,
        deployment_dir=research_config.paths.deployment_dir,
        active_config_path=research_config.paths.active_paper_config_path,
        candidate_id="cand_test",
        paper_window_minutes=research_config.selection.paper_window_minutes,
    )
    deployed = load_config(deployed_path)

    assert deployed_path.exists()
    assert deployed.strategy_library.default == "continuation"
    assert deployed.strategy_library.continuation.enabled is True
    assert deployed.strategy_library.exhaustion.enabled is False
    assert deployed.learning.enabled is False
    assert deployed.strategy_library.continuation.params["min_expected_net_edge_bps"] == 3.5

    active_path = rollback_active_paper_candidate(
        base_config_path=config_path,
        registry=registry,
        active_config_path=research_config.paths.active_paper_config_path,
        candidate_id="cand_test",
        reason="paper_underperformance",
    )
    rolled_back = load_config(active_path)
    latest = registry.get_candidate("cand_test")

    assert active_path.exists()
    assert rolled_back.strategy_library.continuation.params["min_expected_net_edge_bps"] == 3.0
    assert rolled_back.learning.enabled is False
    assert latest is not None
    assert latest["status"] == "rolled_back"


def test_campaign_runs_sessions_and_promotes_candidate(tmp_path: Path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.default = "continuation"  # type: ignore[assignment]
    config.strategy_library.continuation.enabled = True
    config.strategy_library.continuation.params = {
        "min_expected_net_edge_bps": 3.0,
        "no_trade_z": 1.35,
    }
    config_path = tmp_path / "config" / "local.paper.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_config_yaml(config))

    raw_input_path = tmp_path / "raw_input.parquet"
    _write_replay_frame(raw_input_path)
    research_config_path = tmp_path / "config" / "research.yaml"
    _write_research_config(research_config_path, raw_input_path)
    research_config = load_research_config(research_config_path)
    registry = ResearchRegistry(research_config.paths.registry_dir)

    session_runs: list[str] = []
    research_call_count = 0

    async def fake_live_runner(
        config_path_value: str,
        max_events: int,
        max_seconds: int,
        symbols: list[str] | None,
    ) -> Path:
        del config_path_value, max_events, max_seconds, symbols
        run_id = f"run_20260322_00000{len(session_runs) + 1}"
        session_runs.append(run_id)
        report_dir = tmp_path / "reports" / run_id / f"report_{run_id}"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.md").write_text("# report\n", encoding="utf-8")
        (report_dir / "events.csv").write_text(
            "\n".join(
                [
                    "id,run_id,event_time,symbol,action,strategy,params,regime,order_intent,feature_snapshot,fill_qty,fill_price,spread_bps,slippage_bps,realized_pnl,mae_bps,mfe_bps,holding_seconds,vol_bucket,error_tags,reason,fee,notional,is_maker,order_id,fill_seq,fill_role",
                    f"1,{run_id},2026-03-22 00:00:00,BTCUSDT,submit,continuation,,,submit,,,,,,,,,,, ,,,,order_1,,",
                    f"2,{run_id},2026-03-22 00:00:01,BTCUSDT,fill,continuation,,,submit,,,,,,,,,,, ,,,,order_1,1,entry",
                    f"3,{run_id},2026-03-22 00:00:02,BTCUSDT,signal,continuation,,,risk_expected_net_edge,,,,,,,,,,, ,risk_gate,,,,,,",
                ]
            ),
            encoding="utf-8",
        )
        (report_dir / "trades.csv").write_text(
            "\n".join(
                [
                    "id,run_id,symbol,strategy,regime,context,entry_time,exit_time,direction,qty,entry_price,exit_price,realized_pnl,gross_pnl,fees,mae_bps,mfe_bps,spread_entry_bps,spread_exit_bps,holding_seconds,error_tags,entry_fees,exit_fees,entry_fill_count,exit_fill_count,entry_notional,exit_notional,closed_qty",
                    f"1,{run_id},BTCUSDT,continuation,unlabeled,{{}},2026-03-22 00:00:00,2026-03-22 00:00:03,1,0.01,100,101,0.5,0.6,0.1,0,0,0,0,3.0,,0.05,0.05,1,1,1,1,0.01",
                ]
            ),
            encoding="utf-8",
        )
        _write_replay_frame(config.paths.raw_dir / f"raw_{run_id}.parquet")
        return report_dir / "report.md"

    def fake_research_cycle_runner(*, base_config_path: Path, research_config_path: Path, raw_input_path: Path) -> dict[str, object]:
        del base_config_path, research_config_path, raw_input_path
        nonlocal research_call_count
        research_call_count += 1
        report_path = tmp_path / f"research_report_{research_call_count}.md"
        report_path.write_text("# research\n", encoding="utf-8")
        if research_call_count == 1:
            registry.record_candidate(
                {
                    "candidate_id": "cand_session_1",
                    "research_run_id": "research_20260322_000001",
                    "strategy_name": "continuation",
                    "params": {"min_expected_net_edge_bps": 3.5, "no_trade_z": 1.35},
                    "baseline_params": {"min_expected_net_edge_bps": 3.0, "no_trade_z": 1.35},
                    "status": "validated_for_paper",
                }
            )
            return {
                "summary": {
                    "status": "validated_for_paper",
                    "selected_candidate": {"candidate_id": "cand_session_1"},
                },
                "report_path": report_path,
            }
        return {
            "summary": {
                "status": "no_candidate",
                "selected_candidate": None,
            },
            "report_path": report_path,
        }

    result = asyncio.run(
        run_paper_research_campaign(
            base_config_path=config_path,
            research_config_path=research_config_path,
            sessions=2,
            session_seconds=5,
            live_runner=fake_live_runner,
            research_cycle_runner=fake_research_cycle_runner,
        )
    )

    sessions_path = Path(result["campaign_dir"]) / "sessions.jsonl"
    rows = [json.loads(line) for line in sessions_path.read_text(encoding="utf-8").splitlines()]
    active_config = load_config(research_config.paths.active_paper_config_path)

    assert result["status"] == "completed"
    assert result["sessions_completed"] == 2
    assert [row["deployment_action"] for row in rows] == ["promoted_candidate", "kept_current"]
    assert rows[0]["session_metrics"]["submits"] == 1
    assert rows[0]["session_metrics"]["fills"] == 1
    assert rows[0]["session_metrics"]["closed_trades"] == 1
    assert active_config.strategy_library.continuation.params["min_expected_net_edge_bps"] == 3.5
