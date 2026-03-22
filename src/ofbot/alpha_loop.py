from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ofbot.config import AppConfig, dump_config_yaml, load_config
from ofbot.memory.learning import get_run_review
from ofbot.memory.store import MemoryStore
from quantflow.cli.main import _load_candidate_deep_dive_features, _load_order_flow_suite_features
from quantflow.core import load_config as load_qflow_config
from quantflow.core.config import StrategyConfig
from quantflow.reporting import write_candidate_deep_dive_report
from quantflow.validation import run_candidate_deep_dive, run_order_flow_suite


CRITICAL_CANARY_REASONS = frozenset(
    {
        "risk_daily_loss_limit",
        "risk_max_consecutive_losses",
        "risk_catastrophic_stop",
    }
)
INFRASTRUCTURE_CANARY_REASONS = frozenset(
    {
        "risk_broken_websocket",
        "risk_stale_market_data",
    }
)
ACTIVE_RUNTIME_STATUSES = frozenset({"live_canary", "deployed_local"})
RUNTIME_STATUS_PRIORITY = {
    "deployed_local": 0,
    "live_canary": 1,
    "promotable": 2,
    "pending": 3,
    "confirmed_live": 4,
    "rejected_live": 5,
    "rejected": 6,
}


def deployment_root(config: AppConfig) -> Path:
    return config.paths.memory_dir / "deployments"


def active_paper_config_path(config: AppConfig) -> Path:
    return deployment_root(config) / "active.paper.yaml"


def current_runtime_config_path(*, base_config_path: Path) -> Path:
    config = load_config(base_config_path)
    active_path = active_paper_config_path(config)
    return active_path if active_path.exists() else base_config_path


def runtime_candidate_rows(store: MemoryStore) -> list[dict[str, Any]]:
    frame = store.run_query_df(
        """
        SELECT
            lc.candidate_id,
            lc.source_run_id,
            lc.strategy,
            lc.archetype,
            lc.status,
            lc.params,
            lc.overlay_yaml_path,
            COUNT(cv.validation_run_id) AS validation_count,
            COALESCE(SUM(cv.baseline_net_pnl), 0.0) AS baseline_net_pnl,
            COALESCE(SUM(cv.candidate_net_pnl), 0.0) AS candidate_net_pnl,
            COALESCE(SUM(cv.candidate_trade_count), 0) AS candidate_trade_count,
            MAX(lc.updated_at) AS updated_at
        FROM learning_candidates lc
        LEFT JOIN candidate_validations cv ON cv.candidate_id = lc.candidate_id
        GROUP BY
            lc.candidate_id,
            lc.source_run_id,
            lc.strategy,
            lc.archetype,
            lc.status,
            lc.params,
            lc.overlay_yaml_path
        ORDER BY updated_at DESC, lc.candidate_id ASC
        """
    )
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        params = _decode_json(row.get("params"))
        baseline_net = float(row.get("baseline_net_pnl", 0.0) or 0.0)
        candidate_net = float(row.get("candidate_net_pnl", 0.0) or 0.0)
        status = str(row.get("status") or "pending")
        rows.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "source_run_id": str(row["source_run_id"]),
                "strategy": str(row["strategy"]),
                "archetype": str(row["archetype"]),
                "status": status,
                "params": params,
                "overlay_yaml_path": str(row["overlay_yaml_path"]),
                "validation_count": int(row.get("validation_count", 0) or 0),
                "baseline_net_pnl": baseline_net,
                "candidate_net_pnl": candidate_net,
                "candidate_trade_count": int(row.get("candidate_trade_count", 0) or 0),
                "delta_net_pnl": candidate_net - baseline_net,
                "updated_at": row.get("updated_at"),
                "status_priority": RUNTIME_STATUS_PRIORITY.get(status, 99),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["status_priority"],
            -float(row["delta_net_pnl"]),
            -int(row["validation_count"]),
            row["candidate_id"],
        ),
    )


def current_active_runtime_candidate(store: MemoryStore) -> dict[str, Any] | None:
    for row in runtime_candidate_rows(store):
        if row["status"] in ACTIVE_RUNTIME_STATUSES:
            return row
    return None


def best_promotable_runtime_candidate(store: MemoryStore) -> dict[str, Any] | None:
    promotable = [row for row in runtime_candidate_rows(store) if row["status"] == "promotable"]
    if not promotable:
        return None
    return sorted(
        promotable,
        key=lambda row: (-float(row["delta_net_pnl"]), -int(row["validation_count"]), row["candidate_id"]),
    )[0]


def best_offline_candidate(store: MemoryStore) -> dict[str, Any] | None:
    candidates = store.list_offline_research_candidates(limit=20)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            0 if row.get("status") == "validated_edge" else 1,
            -float(
                _decode_json(row.get("summary"))
                .get("deep_dive_aggregate", {})
                .get("best_walkforward_mean_test_net_pnl", 0.0)
                or 0.0
            ),
            str(row.get("candidate_id") or ""),
        ),
    )[0]


def live_candidate_config(candidate: dict[str, Any], *, base_config_path: Path) -> AppConfig:
    base_config = load_config(base_config_path)
    strategy_name = str(candidate["strategy"])
    params = dict(candidate.get("params") or {})
    for variant_name in ("continuation", "exhaustion", "hybrid"):
        variant = getattr(base_config.strategy_library, variant_name)
        variant.enabled = variant_name == strategy_name
        if variant_name == strategy_name:
            variant.params.update(params)
    base_config.strategy_library.default = strategy_name  # type: ignore[assignment]
    return base_config


def ensure_live_canary_candidate(
    *,
    store: MemoryStore,
    base_config_path: Path,
) -> dict[str, Any] | None:
    active = current_active_runtime_candidate(store)
    if active is not None:
        return active

    candidate = best_promotable_runtime_candidate(store)
    if candidate is None:
        return None

    config = live_candidate_config(candidate, base_config_path=base_config_path)
    root = deployment_root(config)
    root.mkdir(parents=True, exist_ok=True)
    deployment_path = root / f"{candidate['candidate_id']}.paper.yaml"
    deployment_path.write_text(dump_config_yaml(config))
    active_paper_config_path(config).write_text(dump_config_yaml(config))
    store.update_learning_candidate_status(candidate["candidate_id"], "live_canary")
    candidate["status"] = "live_canary"
    candidate["active_config_path"] = str(deployment_path)
    return candidate


def reset_active_runtime_config(*, base_config_path: Path) -> None:
    config = load_config(base_config_path)
    active_path = active_paper_config_path(config)
    if active_path.exists():
        active_path.unlink()


def candidate_rollout_summary(
    *,
    store: MemoryStore,
    run_id: str,
) -> dict[str, Any]:
    review = get_run_review(store, run_id) or {}
    summary = dict(review.get("summary") or {})
    event_rows = store.run_query_df(
        """
        SELECT order_intent, reason
        FROM event_journal
        WHERE run_id=?
        """,
        [run_id],
    ).to_dict(orient="records")
    critical_risk_count = 0
    infrastructure_risk_count = 0
    for row in event_rows:
        for value in (row.get("order_intent"), row.get("reason")):
            if value in CRITICAL_CANARY_REASONS:
                critical_risk_count += 1
            if value in INFRASTRUCTURE_CANARY_REASONS:
                infrastructure_risk_count += 1
    trade_count = int(summary.get("trade_count", 0) or 0)
    net_pnl = float(summary.get("net_pnl", 0.0) or 0.0)
    is_infrastructure_only = (
        infrastructure_risk_count > 0 and critical_risk_count == 0 and trade_count == 0 and net_pnl == 0.0
    )
    return {
        "run_id": run_id,
        "trade_count": trade_count,
        "net_pnl": net_pnl,
        "critical_risk_count": critical_risk_count,
        "infrastructure_risk_count": infrastructure_risk_count,
        "is_infrastructure_only": is_infrastructure_only,
        "summary": summary,
    }


def record_live_canary_result(
    *,
    store: MemoryStore,
    base_config_path: Path,
    candidate_id: str,
    run_id: str,
    active_config_path: Path,
) -> str:
    rollout = candidate_rollout_summary(store=store, run_id=run_id)
    store.upsert_candidate_rollout(
        candidate_id=candidate_id,
        run_id=run_id,
        rollout_stage="live_canary",
        net_pnl=float(rollout["net_pnl"]),
        trade_count=int(rollout["trade_count"]),
        critical_risk_count=int(rollout["critical_risk_count"]),
        infrastructure_risk_count=int(rollout["infrastructure_risk_count"]),
        active_config_path=str(active_config_path),
        summary={
            "is_infrastructure_only": bool(rollout["is_infrastructure_only"]),
            "run_summary": rollout["summary"],
        },
    )

    rollouts = store.get_candidate_rollouts(candidate_id)
    counted_rollouts = [
        row
        for row in rollouts
        if not bool(_decode_json(row.get("summary")).get("is_infrastructure_only", False))
    ]
    critical_total = sum(int(row.get("critical_risk_count", 0) or 0) for row in counted_rollouts)
    total_trades = sum(int(row.get("trade_count", 0) or 0) for row in counted_rollouts)
    total_net_pnl = sum(float(row.get("net_pnl", 0.0) or 0.0) for row in counted_rollouts)

    if critical_total > 0:
        store.update_learning_candidate_status(candidate_id, "rejected_live")
        reset_active_runtime_config(base_config_path=base_config_path)
        return "rejected_live"

    if total_trades >= 20:
        if total_net_pnl > 0.0:
            store.update_learning_candidate_status(candidate_id, "confirmed_live")
            store.update_learning_candidate_status(candidate_id, "deployed_local")
            return "deployed_local"
        store.update_learning_candidate_status(candidate_id, "rejected_live")
        reset_active_runtime_config(base_config_path=base_config_path)
        return "rejected_live"

    return "live_canary"


def offline_candidate_id(
    *,
    strategy_name: str,
    bar_size: str,
    execution_mode: str,
    params: dict[str, Any],
    dataset_window: str,
    config_hash: str,
) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(
        f"{strategy_name}|{bar_size}|{execution_mode}|{dataset_window}|{config_hash}|{payload}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{strategy_name}_{bar_size}_{execution_mode}_{digest}"


def run_offline_research_cycle(
    *,
    store: MemoryStore,
    suite_config_path: Path,
    deep_dive_config_path: Path,
) -> dict[str, Any]:
    suite_config = load_qflow_config(suite_config_path)
    feature_frames = _load_order_flow_suite_features(
        suite_config,
        list(suite_config.dataset.symbols),
        suite_config.dataset.dataset_type,
    )
    suite_result = run_order_flow_suite(feature_frames=feature_frames, config=suite_config)
    suite_hash = _file_hash(suite_config_path)
    dataset_window = (
        f"{suite_config.dataset.start_date.isoformat()}::{suite_config.dataset.end_date.isoformat()}"
    )
    for row in suite_result.ranked_candidates[:5]:
        candidate_id = offline_candidate_id(
            strategy_name=str(row["strategy_name"]),
            bar_size=str(row["bar_size"]),
            execution_mode=str(row["execution_mode"]),
            params=dict(row["params"]),
            dataset_window=dataset_window,
            config_hash=suite_hash,
        )
        store.upsert_offline_research_candidate(
            candidate_id=candidate_id,
            strategy_name=str(row["strategy_name"]),
            bar_size=str(row["bar_size"]),
            execution_mode=str(row["execution_mode"]),
            status="scouted",
            params=dict(row["params"]),
            dataset_window=dataset_window,
            config_hash=suite_hash,
            report_dir=None,
            summary={"suite_candidate": dict(row)},
        )

    qualified = next(
        (
            row
            for row in suite_result.ranked_candidates
            if str(row["execution_mode"]) == "passive"
            and float(row["walkforward_mean_test_net_pnl"]) > 0.0
            and float(row["positive_test_window_ratio"]) >= 0.55
            and not bool(row["execution_fragile"])
        ),
        None,
    )
    if qualified is None:
        return {
            "suite_result": suite_result,
            "selected_candidate": None,
            "deep_dive_result": None,
            "status": "no_candidate",
        }

    qualified_id = offline_candidate_id(
        strategy_name=str(qualified["strategy_name"]),
        bar_size=str(qualified["bar_size"]),
        execution_mode=str(qualified["execution_mode"]),
        params=dict(qualified["params"]),
        dataset_window=dataset_window,
        config_hash=suite_hash,
    )
    store.upsert_offline_research_candidate(
        candidate_id=qualified_id,
        strategy_name=str(qualified["strategy_name"]),
        bar_size=str(qualified["bar_size"]),
        execution_mode=str(qualified["execution_mode"]),
        status="deep_dive_pending",
        params=dict(qualified["params"]),
        dataset_window=dataset_window,
        config_hash=suite_hash,
        report_dir=None,
        summary={"suite_candidate": dict(qualified)},
    )

    deep_config = load_qflow_config(deep_dive_config_path)
    deep_config = _configured_deep_dive_config(
        deep_config=deep_config,
        suite_candidate=qualified,
    )
    deep_features = _load_candidate_deep_dive_features(
        deep_config,
        list(deep_config.dataset.symbols),
        deep_config.dataset.dataset_type,
    )
    deep_result = run_candidate_deep_dive(features=deep_features, config=deep_config)
    run_name = (
        f"{deep_config.reporting.run_name}_{qualified['strategy_name']}_{qualified['bar_size']}_candidate_deep_dive"
    )
    report_path = write_candidate_deep_dive_report(
        deep_result,
        deep_config.paths.reports_dir,
        run_name,
    )
    final_status = str(deep_result.aggregate["final_verdict"])
    store.upsert_offline_research_candidate(
        candidate_id=qualified_id,
        strategy_name=str(qualified["strategy_name"]),
        bar_size=str(qualified["bar_size"]),
        execution_mode=str(qualified["execution_mode"]),
        status=final_status,
        params=dict(qualified["params"]),
        dataset_window=dataset_window,
        config_hash=_file_hash(deep_dive_config_path),
        report_dir=str(report_path.parent),
        summary={
            "suite_candidate": dict(qualified),
            "deep_dive_aggregate": dict(deep_result.aggregate),
            "best_scenario": dict(deep_result.best_scenario),
        },
    )
    return {
        "suite_result": suite_result,
        "selected_candidate": dict(qualified),
        "deep_dive_result": deep_result,
        "status": final_status,
        "report_path": report_path,
    }


def write_alpha_hunt_status(
    *,
    store: MemoryStore,
    base_config_path: Path,
    output_path: Path | None = None,
) -> Path:
    config = load_config(base_config_path)
    destination = output_path or (config.paths.root / "docs" / "alpha_hunt_status.md")
    destination.parent.mkdir(parents=True, exist_ok=True)

    runtime_candidates = runtime_candidate_rows(store)
    active_runtime = current_active_runtime_candidate(store)
    best_runtime = runtime_candidates[0] if runtime_candidates else None
    best_offline = best_offline_candidate(store)
    active_config = current_runtime_config_path(base_config_path=base_config_path)

    lines = [
        "# Alpha Hunt Status",
        "",
        f"Generated: `{datetime.now(UTC).replace(microsecond=0).isoformat()}`",
        "",
        "## Runtime",
        f"- Base config: `{base_config_path}`",
        f"- Active config: `{active_config}`",
        f"- Active candidate: `{active_runtime['candidate_id']}`" if active_runtime else "- Active candidate: none",
    ]
    if best_runtime is not None:
        lines.extend(
            [
                "",
                "## Best Runtime Candidate",
                f"- candidate: `{best_runtime['candidate_id']}`",
                f"- status: `{best_runtime['status']}`",
                f"- strategy: `{best_runtime['strategy']}`/{best_runtime['archetype']}",
                f"- delta vs baseline: `{best_runtime['delta_net_pnl']:+.6f}`",
                f"- validations: `{best_runtime['validation_count']}`",
                f"- candidate trades: `{best_runtime['candidate_trade_count']}`",
            ]
        )
    else:
        lines.extend(["", "## Best Runtime Candidate", "- none"])

    if best_offline is not None:
        summary = _decode_json(best_offline.get("summary"))
        deep_dive = summary.get("deep_dive_aggregate", {})
        lines.extend(
            [
                "",
                "## Best Offline Candidate",
                f"- candidate: `{best_offline['candidate_id']}`",
                f"- status: `{best_offline['status']}`",
                f"- strategy: `{best_offline['strategy_name']}`",
                f"- bar size: `{best_offline['bar_size']}`",
                f"- execution: `{best_offline['execution_mode']}`",
                f"- best OOS mean: `{deep_dive.get('best_walkforward_mean_test_net_pnl', 'n/a')}`",
                f"- report dir: `{best_offline.get('report_dir') or 'n/a'}`",
            ]
        )
    else:
        lines.extend(["", "## Best Offline Candidate", "- none"])

    if active_runtime is not None:
        rollouts = store.get_candidate_rollouts(str(active_runtime["candidate_id"]))
        counted_rollouts = [
            row
            for row in rollouts
            if not bool(_decode_json(row.get("summary")).get("is_infrastructure_only", False))
        ]
        lines.extend(
            [
                "",
                "## Live Canary",
                f"- counted runs: `{len(counted_rollouts)}`",
                f"- counted trades: `{sum(int(row.get('trade_count', 0) or 0) for row in counted_rollouts)}`",
                f"- counted net pnl: `{sum(float(row.get('net_pnl', 0.0) or 0.0) for row in counted_rollouts):+.6f}`",
                f"- critical risk hits: `{sum(int(row.get('critical_risk_count', 0) or 0) for row in counted_rollouts)}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Next Action",
            _next_alpha_action(best_runtime=best_runtime, active_runtime=active_runtime, best_offline=best_offline),
        ]
    )
    destination.write_text("\n".join(lines) + "\n")
    return destination


def next_offline_refresh(*, now: datetime, hours: float) -> datetime:
    return now + timedelta(seconds=max(300.0, hours * 3600.0))


def _configured_deep_dive_config(
    *,
    deep_config: Any,
    suite_candidate: dict[str, Any],
) -> Any:
    feature_profile = "order_flow_suite_5s_v1" if suite_candidate["bar_size"] == "5s" else "order_flow_suite_15s_v1"
    return deep_config.__class__(
        paths=deep_config.paths,
        binance=deep_config.binance,
        dataset=deep_config.dataset,
        features=deep_config.features.__class__(
            bar_size=str(suite_candidate["bar_size"]),
            feature_profile=feature_profile,
            rolling_window_bars=deep_config.features.rolling_window_bars,
            volatility_window_bars=deep_config.features.volatility_window_bars,
            range_window_bars=deep_config.features.range_window_bars,
            normalization_window_bars=deep_config.features.normalization_window_bars,
        ),
        strategy=StrategyConfig(
            name=str(suite_candidate["strategy_name"]),
            position_size=deep_config.strategy.position_size,
            params=dict(suite_candidate["params"]),
        ),
        execution=deep_config.execution,
        portfolio=replace(deep_config.portfolio, allow_short=False),
        risk=deep_config.risk,
        validation=deep_config.validation,
        reporting=deep_config.reporting,
    )


def _next_alpha_action(
    *,
    best_runtime: dict[str, Any] | None,
    active_runtime: dict[str, Any] | None,
    best_offline: dict[str, Any] | None,
) -> str:
    if active_runtime is not None:
        return f"- Keep running paper_local canary for `{active_runtime['candidate_id']}`."
    if best_runtime is not None and best_runtime["status"] == "promotable":
        return f"- Activate `{best_runtime['candidate_id']}` as the next live canary."
    if best_offline is not None and best_offline["status"] == "validated_edge":
        return "- Offline research found a validated edge, but runtime rollout still needs its own native candidate."
    return "- Keep collecting replay/live evidence until a promotable runtime candidate appears."


def _decode_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
