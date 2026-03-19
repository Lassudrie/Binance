from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ofbot.config import AppConfig, dump_config_yaml, load_config
from ofbot.memory.store import MemoryStore

MAX_CANDIDATES_PER_RUN = 5
MAX_PARAMS_PER_CANDIDATE = 2
PROMOTABLE_MIN_VALIDATIONS = 3
PROMOTABLE_MIN_TRADES = 20


def default_run_report_dir(config: AppConfig, run_id: str) -> Path:
    return config.paths.reports_dir / run_id / f"report_{run_id}"


def advance_learning_cycle(
    *,
    store: MemoryStore,
    config: AppConfig,
    run_id: str,
    raw_input_path: Path | None,
    report_dir: Path | None = None,
    validate_pending: bool = True,
    generate_candidates: bool = True,
) -> Path:
    resolved_report_dir = report_dir or default_run_report_dir(config, run_id)
    resolved_report_dir.mkdir(parents=True, exist_ok=True)

    summary, trades, events = build_run_summary(store=store, run_id=run_id)
    config_snapshot_path, config_snapshot_hash = write_config_snapshot(
        config=config,
        run_id=run_id,
        report_dir=resolved_report_dir,
    )
    review_verdict = derive_review_verdict(summary)
    raw_path_value = str(raw_input_path) if raw_input_path is not None and raw_input_path.exists() else None
    store.upsert_run_review(
        run_id=run_id,
        verdict=review_verdict,
        config_snapshot_path=str(config_snapshot_path),
        config_snapshot_hash=config_snapshot_hash,
        raw_input_path=raw_path_value,
        report_dir=str(resolved_report_dir),
        summary=summary,
        notes={"learning_artifacts_dir": str(resolved_report_dir / "learning")},
    )

    store.delete_lessons_for_run(run_id)
    lessons = extract_lessons(summary=summary, trades=trades, events=events, config=config)
    for lesson in lessons:
        store.insert_lesson_learned(
            run_id=run_id,
            lesson_key=str(lesson["lesson_key"]),
            scope=str(lesson["scope"]),
            lesson_type=str(lesson["lesson_type"]),
            strategy=_optional_str(lesson.get("strategy")),
            regime=_optional_str(lesson.get("regime")),
            evidence=_as_dict(lesson.get("evidence")),
            recommended_action=_as_dict(lesson.get("recommended_action")),
        )

    affected_runs = {run_id}
    if validate_pending and raw_input_path is not None and raw_input_path.exists():
        for update in validate_pending_candidates(
            store=store,
            validation_run_id=run_id,
            raw_input_path=raw_input_path,
        ):
            affected_runs.add(str(update["source_run_id"]))

    if generate_candidates:
        for candidate in materialize_candidates(
            store=store,
            config=config,
            run_id=run_id,
            lessons=lessons,
            config_snapshot_hash=config_snapshot_hash,
        ):
            affected_runs.add(str(candidate["source_run_id"]))

    current_overview = refresh_learning_artifacts(
        store=store,
        config=config,
        run_id=run_id,
        report_dir=resolved_report_dir,
    )
    for affected_run in sorted(affected_runs):
        if affected_run == run_id:
            continue
        refresh_learning_artifacts(store=store, config=config, run_id=affected_run)
    return current_overview


def refresh_learning_artifacts(
    *,
    store: MemoryStore,
    config: AppConfig,
    run_id: str,
    report_dir: Path | None = None,
) -> Path:
    review = get_run_review(store, run_id)
    resolved_report_dir = report_dir or report_dir_for_run(store, config, run_id, review)
    learning_dir = resolved_report_dir / "learning"
    candidates_dir = learning_dir / "candidates"
    validations_dir = learning_dir / "validations"
    learning_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    validations_dir.mkdir(parents=True, exist_ok=True)

    summary = _as_dict(review.get("summary")) if review else {}
    lessons = get_lessons(store, run_id)
    notes = get_manual_notes(store, run_id)
    candidate_queue = get_source_candidates(store, run_id)
    validation_status = get_validation_updates(store, run_id)

    snapshot_path = Path(str(review["config_snapshot_path"])) if review and review.get("config_snapshot_path") else None
    if snapshot_path is not None and snapshot_path.exists():
        shutil.copyfile(snapshot_path, learning_dir / "config_snapshot.yaml")

    for candidate in candidate_queue:
        source_path = Path(str(candidate["overlay_yaml_path"]))
        if source_path.exists():
            shutil.copyfile(source_path, candidates_dir / f"{candidate['candidate_id']}.yaml")
        payload = {
            "candidate": candidate,
            "aggregate": candidate["validation_aggregate"],
            "validations": get_candidate_validations(store, str(candidate["candidate_id"])),
        }
        (validations_dir / f"{candidate['candidate_id']}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
        )

    overview = {
        "run_summary": summary,
        "lessons": lessons,
        "manual_notes": notes,
        "candidate_queue": candidate_queue,
        "validation_status": validation_status,
        "next_actions": build_next_actions(summary, candidate_queue, validation_status, review),
    }
    (learning_dir / "overview.json").write_text(json.dumps(overview, indent=2, sort_keys=True))
    (learning_dir / "overview.md").write_text(render_learning_markdown(run_id=run_id, overview=overview))
    return learning_dir / "overview.md"


def build_run_summary(
    *,
    store: MemoryStore,
    run_id: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    trades = store.run_query_df(
        "SELECT * FROM trade_contexts WHERE run_id=? ORDER BY exit_time ASC",
        [run_id],
    )
    events = store.run_query_df(
        "SELECT * FROM event_journal WHERE run_id=? ORDER BY event_time ASC, id ASC",
        [run_id],
    )
    trade_count = int(len(trades.index))
    net_pnl = _safe_sum(trades, "realized_pnl")
    gross_pnl = _safe_sum(trades, "gross_pnl")
    fees = _fee_sum(trades)
    mean_holding = _safe_mean(trades, "holding_seconds")
    winners = trades[trades["realized_pnl"] > 0.0] if "realized_pnl" in trades.columns else trades.iloc[0:0]
    losers = trades[trades["realized_pnl"] <= 0.0] if "realized_pnl" in trades.columns else trades.iloc[0:0]
    dominant_strategy = dominant_strategy_name(trades=trades, events=events)
    dominant_regime = dominant_regime_name(trades=trades, events=events)
    event_metrics = summarize_event_metrics(events)
    strategy_rows = summarize_strategies(trades)

    winner_capture_ratio = 0.0
    if not winners.empty:
        entry_notional = pd.to_numeric(winners.get("entry_notional", pd.Series(dtype=float)), errors="coerce")
        if entry_notional.empty:
            entry_notional = pd.to_numeric(winners["entry_price"], errors="coerce") * pd.to_numeric(winners["qty"], errors="coerce")
        realized_bps = (
            pd.to_numeric(winners["realized_pnl"], errors="coerce")
            / entry_notional.replace(0.0, pd.NA)
            * 10_000.0
        )
        mfe = pd.to_numeric(winners["mfe_bps"], errors="coerce").replace(0.0, pd.NA)
        ratios = (realized_bps / mfe).replace([pd.NA, math.inf, -math.inf], pd.NA).dropna()
        if not ratios.empty:
            winner_capture_ratio = float(ratios.clip(lower=0.0, upper=2.0).mean())

    summary = {
        "run_id": run_id,
        "trade_count": trade_count,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "mean_holding_seconds": mean_holding,
        "dominant_strategy": dominant_strategy,
        "dominant_regime": dominant_regime,
        "strategy_rows": strategy_rows,
        "winner_trade_count": int(len(winners.index)),
        "loser_trade_count": int(len(losers.index)),
        "winner_mean_holding_seconds": _safe_mean(winners, "holding_seconds"),
        "loser_mean_holding_seconds": _safe_mean(losers, "holding_seconds"),
        "winner_mean_mfe_bps": _safe_mean(winners, "mfe_bps"),
        "loser_mean_mfe_bps": _safe_mean(losers, "mfe_bps"),
        "winner_capture_ratio": winner_capture_ratio,
        "expected_edge_blocks": int(event_metrics["expected_edge_blocks"]),
        "stop_exit_rate": float(event_metrics["stop_exit_rate"]),
        "exit_reasons": event_metrics["exit_reasons"],
        "signal_submit_count": int(event_metrics["signal_submit_count"]),
    }
    return summary, trades, events


def derive_review_verdict(summary: dict[str, Any]) -> str:
    trade_count = int(summary.get("trade_count", 0) or 0)
    net_pnl = float(summary.get("net_pnl", 0.0) or 0.0)
    if trade_count <= 0:
        return "no_trades"
    if net_pnl > 0.0:
        return "positive"
    if net_pnl < 0.0:
        return "negative"
    return "flat"


def write_config_snapshot(
    *,
    config: AppConfig,
    run_id: str,
    report_dir: Path,
) -> tuple[Path, str]:
    stable_path = config.paths.memory_dir / "snapshots" / f"{run_id}.yaml"
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    raw_yaml = dump_config_yaml(config)
    stable_path.write_text(raw_yaml)
    learning_dir = report_dir / "learning"
    learning_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stable_path, learning_dir / "config_snapshot.yaml")
    return stable_path, hashlib.sha256(raw_yaml.encode("utf-8")).hexdigest()


def extract_lessons(
    *,
    summary: dict[str, Any],
    trades: pd.DataFrame,
    events: pd.DataFrame,
    config: AppConfig,
) -> list[dict[str, Any]]:
    lessons: list[dict[str, Any]] = []
    dominant_strategy = _optional_str(summary.get("dominant_strategy")) or config.strategy_library.default
    dominant_regime = _optional_str(summary.get("dominant_regime"))
    trade_count = int(summary.get("trade_count", 0) or 0)
    signal_submit_count = int(summary.get("signal_submit_count", 0) or 0)
    stop_exit_rate = float(summary.get("stop_exit_rate", 0.0) or 0.0)
    net_pnl = float(summary.get("net_pnl", 0.0) or 0.0)
    winner_hold = float(summary.get("winner_mean_holding_seconds", 0.0) or 0.0)
    loser_hold = float(summary.get("loser_mean_holding_seconds", 0.0) or 0.0)
    loser_mfe = float(summary.get("loser_mean_mfe_bps", 0.0) or 0.0)
    winner_capture = float(summary.get("winner_capture_ratio", 0.0) or 0.0)
    winner_mfe = float(summary.get("winner_mean_mfe_bps", 0.0) or 0.0)

    if trade_count <= 0:
        lessons.append(
            build_lesson(
                lesson_key=f"{dominant_strategy}_no_trades",
                scope="run",
                lesson_type="no_trades",
                strategy=dominant_strategy,
                regime=dominant_regime,
                evidence={
                    "trade_count": trade_count,
                    "signal_submit_count": signal_submit_count,
                    "expected_edge_blocks": int(summary.get("expected_edge_blocks", 0) or 0),
                },
                recommended_action={"type": "observe", "message": "No trades to learn from in this run."},
            )
        )
        return lessons

    if net_pnl < 0.0 or stop_exit_rate >= 0.40:
        lessons.append(
            build_lesson(
                lesson_key=f"{dominant_strategy}_stricter_entry",
                scope="run",
                lesson_type="stricter_entry",
                strategy=dominant_strategy,
                regime=dominant_regime,
                evidence={
                    "net_pnl": net_pnl,
                    "stop_exit_rate": stop_exit_rate,
                    "exit_reasons": summary.get("exit_reasons", {}),
                },
                recommended_action={
                    "type": "candidate",
                    "archetype": "stricter_entry",
                    "strategy": dominant_strategy,
                },
            )
        )

    if loser_hold > 0.0 and winner_hold > 0.0 and loser_hold > winner_hold * 1.2 and loser_mfe >= 5.0:
        lessons.append(
            build_lesson(
                lesson_key=f"{dominant_strategy}_faster_exit",
                scope="run",
                lesson_type="faster_exit",
                strategy=dominant_strategy,
                regime=dominant_regime,
                evidence={
                    "winner_mean_holding_seconds": winner_hold,
                    "loser_mean_holding_seconds": loser_hold,
                    "loser_mean_mfe_bps": loser_mfe,
                },
                recommended_action={
                    "type": "candidate",
                    "archetype": "faster_exit",
                    "strategy": dominant_strategy,
                },
            )
        )

    if winner_capture > 0.0 and winner_capture < 0.35 and winner_mfe >= 10.0:
        lessons.append(
            build_lesson(
                lesson_key=f"{dominant_strategy}_longer_hold",
                scope="run",
                lesson_type="longer_hold",
                strategy=dominant_strategy,
                regime=dominant_regime,
                evidence={
                    "winner_capture_ratio": winner_capture,
                    "winner_mean_mfe_bps": winner_mfe,
                },
                recommended_action={
                    "type": "candidate",
                    "archetype": "longer_hold",
                    "strategy": dominant_strategy,
                },
            )
        )

    if int(summary.get("expected_edge_blocks", 0) or 0) > max(3, signal_submit_count // 2):
        lessons.append(
            build_lesson(
                lesson_key=f"{dominant_strategy}_expected_edge_blocks",
                scope="run",
                lesson_type="expected_edge_blocks",
                strategy=dominant_strategy,
                regime=dominant_regime,
                evidence={
                    "expected_edge_blocks": int(summary.get("expected_edge_blocks", 0) or 0),
                    "signal_submit_count": signal_submit_count,
                },
                recommended_action={
                    "type": "observe",
                    "message": "Expected-edge gate blocked a large share of entries.",
                },
            )
        )

    return lessons


def build_lesson(
    *,
    lesson_key: str,
    scope: str,
    lesson_type: str,
    strategy: str | None,
    regime: str | None,
    evidence: dict[str, Any],
    recommended_action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lesson_key": lesson_key,
        "scope": scope,
        "lesson_type": lesson_type,
        "strategy": strategy,
        "regime": regime,
        "evidence": evidence,
        "recommended_action": recommended_action,
    }


def materialize_candidates(
    *,
    store: MemoryStore,
    config: AppConfig,
    run_id: str,
    lessons: list[dict[str, Any]],
    config_snapshot_hash: str,
) -> list[dict[str, Any]]:
    created: list[dict[str, Any]] = []
    chosen_actions: set[tuple[str, str]] = set()

    for lesson in lessons:
        if len(created) >= MAX_CANDIDATES_PER_RUN:
            break
        action = _as_dict(lesson.get("recommended_action"))
        if action.get("type") != "candidate":
            continue
        strategy = _optional_str(action.get("strategy"))
        archetype = _optional_str(action.get("archetype"))
        if strategy is None or archetype is None:
            continue
        pair = (strategy, archetype)
        if pair in chosen_actions:
            continue
        candidate_config = build_candidate_config(config=config, strategy=strategy, archetype=archetype)
        if candidate_config is None:
            continue
        changed_params = extract_changed_params(
            base_params=strategy_params(config=config, strategy=strategy),
            candidate_params=strategy_params(config=candidate_config, strategy=strategy),
        )
        if not changed_params or len(changed_params) > MAX_PARAMS_PER_CANDIDATE:
            continue
        candidate_id = build_candidate_id(
            source_run_id=run_id,
            strategy=strategy,
            archetype=archetype,
            params=changed_params,
        )
        overlay_path = config.paths.memory_dir / "learning_candidates" / f"{candidate_id}.yaml"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text(dump_config_yaml(candidate_config))
        current_status = current_candidate_status(store, candidate_id)
        status = current_status or "pending"
        rationale = {
            "lesson_key": lesson["lesson_key"],
            "evidence": lesson["evidence"],
            "recommended_action": action,
        }
        store.upsert_learning_candidate(
            candidate_id=candidate_id,
            source_run_id=run_id,
            strategy=strategy,
            archetype=archetype,
            status=status,
            base_config_hash=config_snapshot_hash,
            overlay_yaml_path=str(overlay_path),
            params=changed_params,
            rationale=rationale,
        )
        created.append(
            {
                "candidate_id": candidate_id,
                "source_run_id": run_id,
                "strategy": strategy,
                "archetype": archetype,
                "params": changed_params,
                "overlay_yaml_path": str(overlay_path),
            }
        )
        chosen_actions.add(pair)
    return created


def validate_pending_candidates(
    *,
    store: MemoryStore,
    validation_run_id: str,
    raw_input_path: Path,
) -> list[dict[str, Any]]:
    if not raw_input_path.exists():
        return []

    candidates = store.run_query_df(
        """
        SELECT *
        FROM learning_candidates
        WHERE status='pending' AND source_run_id <> ?
        ORDER BY created_at ASC, candidate_id ASC
        """,
        [validation_run_id],
    )
    if candidates.empty:
        return []

    from ofbot.replay.player import iter_replay_events

    events = iter_replay_events(raw_input_path)
    baseline_cache: dict[tuple[str, str], dict[str, Any]] = {}
    updates: list[dict[str, Any]] = []

    for row in candidates.to_dict(orient="records"):
        candidate = normalize_record(row)
        source_run_id = str(candidate["source_run_id"])
        strategy = str(candidate["strategy"])
        review = get_run_review(store, source_run_id)
        snapshot_path_value = _optional_str(review.get("config_snapshot_path")) if review else None
        candidate_config_path = Path(str(candidate["overlay_yaml_path"]))
        if snapshot_path_value is None or not Path(snapshot_path_value).exists() or not candidate_config_path.exists():
            continue

        cache_key = (snapshot_path_value, strategy)
        if cache_key not in baseline_cache:
            baseline_cache[cache_key] = run_replay_metrics(
                config=build_baseline_validation_config(
                    snapshot_path=Path(snapshot_path_value),
                    strategy=strategy,
                ),
                events=events,
            )
        candidate_metrics = run_replay_metrics(
            config=load_config(candidate_config_path),
            events=events,
        )
        baseline_metrics = baseline_cache[cache_key]
        verdict = validation_verdict(baseline_metrics=baseline_metrics, candidate_metrics=candidate_metrics)
        summary = {
            "raw_input_path": str(raw_input_path),
            "source_run_id": source_run_id,
            "validation_run_id": validation_run_id,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        }
        store.upsert_candidate_validation(
            candidate_id=str(candidate["candidate_id"]),
            validation_run_id=validation_run_id,
            baseline_net_pnl=float(baseline_metrics["net_pnl"]),
            candidate_net_pnl=float(candidate_metrics["net_pnl"]),
            baseline_trade_count=int(baseline_metrics["trade_count"]),
            candidate_trade_count=int(candidate_metrics["trade_count"]),
            baseline_max_drawdown=float(baseline_metrics["max_drawdown"]),
            candidate_max_drawdown=float(candidate_metrics["max_drawdown"]),
            verdict=verdict,
            summary=summary,
        )
        status = recompute_candidate_status(store, str(candidate["candidate_id"]))
        updates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_run_id": source_run_id,
                "validation_run_id": validation_run_id,
                "verdict": verdict,
                "status": status,
                "baseline_net_pnl": baseline_metrics["net_pnl"],
                "candidate_net_pnl": candidate_metrics["net_pnl"],
            }
        )
    return updates


def build_baseline_validation_config(*, snapshot_path: Path, strategy: str) -> AppConfig:
    config = load_config(snapshot_path)
    return force_single_strategy(config=config, strategy=strategy)


def build_candidate_config(*, config: AppConfig, strategy: str, archetype: str) -> AppConfig | None:
    candidate = force_single_strategy(config=config.model_copy(deep=True), strategy=strategy)
    params = strategy_params(config=candidate, strategy=strategy)
    changed_params = candidate_param_updates(strategy=strategy, archetype=archetype, params=params)
    if not changed_params:
        return None
    strategy_cfg = getattr(candidate.strategy_library, strategy)
    strategy_cfg.params.update(changed_params)
    return candidate


def force_single_strategy(*, config: AppConfig, strategy: str) -> AppConfig:
    forced = config.model_copy(deep=True)
    forced.learning.enabled = False
    forced.paper_local_record_raw = False
    forced.strategy_library.default = strategy  # type: ignore[assignment]
    for strategy_name in ("continuation", "exhaustion", "hybrid"):
        variant = getattr(forced.strategy_library, strategy_name)
        variant.enabled = strategy_name == strategy
    return forced


def strategy_params(*, config: AppConfig, strategy: str) -> dict[str, Any]:
    params = getattr(config.strategy_library, strategy).params
    return dict(params)


def extract_changed_params(
    *,
    base_params: dict[str, Any],
    candidate_params: dict[str, Any],
) -> dict[str, Any]:
    changed: dict[str, Any] = {}
    for key, value in candidate_params.items():
        if base_params.get(key) != value:
            changed[key] = value
    return changed


def candidate_param_updates(
    *,
    strategy: str,
    archetype: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    if strategy == "continuation":
        return continuation_candidate_params(archetype=archetype, params=params)
    if strategy == "exhaustion":
        return exhaustion_candidate_params(archetype=archetype, params=params)
    if strategy == "hybrid":
        return hybrid_candidate_params(archetype=archetype, params=params)
    return {}


def continuation_candidate_params(*, archetype: str, params: dict[str, Any]) -> dict[str, Any]:
    if archetype == "stricter_entry":
        return {
            "min_expected_net_edge_bps": round(float(params.get("min_expected_net_edge_bps", 2.0) or 2.0) + 1.0, 3),
            "no_trade_z": round(float(params.get("no_trade_z", params.get("notrade_z", 0.15)) or 0.15) + 0.15, 3),
        }
    if archetype == "faster_exit":
        return {
            "max_holding_time_s": max(30, int(float(params.get("max_holding_time_s", 420) or 420) * 0.75)),
            "trailing_stop_bps": round(max(2.0, float(params.get("trailing_stop_bps", 35.0) or 35.0) * 0.8), 3),
        }
    if archetype == "longer_hold":
        return {
            "max_holding_time_s": max(30, int(float(params.get("max_holding_time_s", 420) or 420) * 1.25)),
            "take_profit_bps": round(max(1.0, float(params.get("take_profit_bps", 150.0) or 150.0) * 1.2), 3),
        }
    return {}


def exhaustion_candidate_params(*, archetype: str, params: dict[str, Any]) -> dict[str, Any]:
    if archetype == "stricter_entry":
        return {
            "extreme_flow_threshold": round(float(params.get("extreme_flow_threshold", 1.8) or 1.8) + 0.2, 3),
            "min_burst_z": round(float(params.get("min_burst_z", 1.0) or 1.0) + 0.2, 3),
        }
    if archetype == "faster_exit":
        return {
            "max_holding_time_s": max(30, int(float(params.get("max_holding_time_s", 360) or 360) * 0.75)),
            "trailing_stop_bps": round(max(2.0, float(params.get("trailing_stop_bps", 30.0) or 30.0) * 0.8), 3),
        }
    if archetype == "longer_hold":
        return {
            "max_holding_time_s": max(30, int(float(params.get("max_holding_time_s", 360) or 360) * 1.25)),
            "take_profit_bps": round(max(1.0, float(params.get("take_profit_bps", 100.0) or 100.0) * 1.15), 3),
        }
    return {}


def hybrid_candidate_params(*, archetype: str, params: dict[str, Any]) -> dict[str, Any]:
    if archetype == "stricter_entry":
        return {
            "regime_confidence_threshold": round(min(0.99, float(params.get("regime_confidence_threshold", 0.55) or 0.55) + 0.05), 3),
            "spread_bps_max": round(max(1.0, float(params.get("spread_bps_max", 200.0) or 200.0) * 0.9), 3),
        }
    if archetype == "faster_exit":
        return {
            "time_exit_s": max(30, int(float(params.get("time_exit_s", 600) or 600) * 0.75)),
            "trailing_stop_bps": round(max(2.0, float(params.get("trailing_stop_bps", 28.0) or 28.0) * 0.8), 3),
        }
    if archetype == "longer_hold":
        return {
            "time_exit_s": max(30, int(float(params.get("time_exit_s", 600) or 600) * 1.25)),
            "take_profit_bps": round(max(1.0, float(params.get("take_profit_bps", 120.0) or 120.0) * 1.15), 3),
        }
    return {}


def build_candidate_id(
    *,
    source_run_id: str,
    strategy: str,
    archetype: str,
    params: dict[str, Any],
) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(f"{source_run_id}|{strategy}|{archetype}|{payload}".encode("utf-8")).hexdigest()[:8]
    return f"{source_run_id}_{strategy}_{archetype}_{digest}"


def current_candidate_status(store: MemoryStore, candidate_id: str) -> str | None:
    row = store._conn.execute(
        "SELECT status FROM learning_candidates WHERE candidate_id=?",
        [candidate_id],
    ).fetchone()
    return str(row[0]) if row is not None else None


def recompute_candidate_status(store: MemoryStore, candidate_id: str) -> str:
    validations = get_candidate_validations(store, candidate_id)
    validation_count = len(validations)
    candidate_trade_count = sum(int(item.get("candidate_trade_count", 0) or 0) for item in validations)
    candidate_net_pnl = sum(float(item.get("candidate_net_pnl", 0.0) or 0.0) for item in validations)
    baseline_net_pnl = sum(float(item.get("baseline_net_pnl", 0.0) or 0.0) for item in validations)
    if (
        validation_count >= PROMOTABLE_MIN_VALIDATIONS
        and candidate_trade_count >= PROMOTABLE_MIN_TRADES
        and candidate_net_pnl > 0.0
        and candidate_net_pnl > baseline_net_pnl
    ):
        status = "promotable"
    elif validation_count >= PROMOTABLE_MIN_VALIDATIONS and candidate_net_pnl <= baseline_net_pnl:
        status = "rejected"
    else:
        status = "pending"
    store.update_learning_candidate_status(candidate_id, status)
    return status


def validation_verdict(
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> str:
    baseline_net = float(baseline_metrics.get("net_pnl", 0.0) or 0.0)
    candidate_net = float(candidate_metrics.get("net_pnl", 0.0) or 0.0)
    if candidate_net > baseline_net:
        return "improved"
    if candidate_net < baseline_net:
        return "worse"
    return "flat"


def run_replay_metrics(*, config: AppConfig, events: list[Any]) -> dict[str, Any]:
    from ofbot.engine import PaperEngine

    with tempfile.TemporaryDirectory(prefix="ofbot_learning_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_config = config.model_copy(deep=True)
        runtime_config.paths.raw_dir = temp_dir / "raw"
        runtime_config.paths.reports_dir = temp_dir / "reports"
        runtime_config.paths.memory_dir = temp_dir / "memory"
        runtime_config.learning.duckdb_path = temp_dir / "memory" / "learning.duckdb"
        runtime_config.paper_local_record_raw = False

        engine = PaperEngine(config=runtime_config, depth_snapshot_client=None, runtime_clock="event")
        try:
            engine.process_events(events)
            trades = engine.store.run_query_df(
                """
                SELECT realized_pnl, holding_seconds, exit_time
                FROM trade_contexts
                WHERE run_id=?
                ORDER BY exit_time ASC
                """,
                [engine.run_id],
            )
            return {
                "trade_count": int(len(trades.index)),
                "net_pnl": _safe_sum(trades, "realized_pnl"),
                "mean_holding_seconds": _safe_mean(trades, "holding_seconds"),
                "max_drawdown": compute_max_drawdown(trades),
            }
        finally:
            engine.close()


def compute_max_drawdown(trades: pd.DataFrame) -> float:
    if trades.empty or "realized_pnl" not in trades.columns:
        return 0.0
    cumulative = pd.to_numeric(trades["realized_pnl"], errors="coerce").fillna(0.0).cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    return float(drawdown.max()) if not drawdown.empty else 0.0


def report_dir_for_run(
    store: MemoryStore,
    config: AppConfig,
    run_id: str,
    review: dict[str, Any] | None = None,
) -> Path:
    active_review = review or get_run_review(store, run_id)
    path_value = _optional_str(active_review.get("report_dir")) if active_review else None
    if path_value:
        return Path(path_value)
    return default_run_report_dir(config, run_id)


def get_run_review(store: MemoryStore, run_id: str) -> dict[str, Any] | None:
    frame = store.run_query_df("SELECT * FROM run_reviews WHERE run_id=?", [run_id])
    if frame.empty:
        return None
    return normalize_record(frame.to_dict(orient="records")[0])


def get_lessons(store: MemoryStore, run_id: str) -> list[dict[str, Any]]:
    frame = store.run_query_df(
        "SELECT * FROM lessons_learned WHERE run_id=? ORDER BY id ASC",
        [run_id],
    )
    return [normalize_record(row) for row in frame.to_dict(orient="records")]


def get_manual_notes(store: MemoryStore, run_id: str) -> list[dict[str, Any]]:
    frame = store.run_query_df(
        "SELECT * FROM manual_notes WHERE run_id=? ORDER BY created_at ASC, id ASC",
        [run_id],
    )
    return [normalize_record(row) for row in frame.to_dict(orient="records")]


def get_source_candidates(store: MemoryStore, run_id: str) -> list[dict[str, Any]]:
    frame = store.run_query_df(
        """
        SELECT *
        FROM learning_candidates
        WHERE source_run_id=?
        ORDER BY created_at ASC, candidate_id ASC
        """,
        [run_id],
    )
    candidates: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        normalized = normalize_record(row)
        candidate_id = str(normalized["candidate_id"])
        normalized["validation_aggregate"] = candidate_validation_aggregate(store, candidate_id)
        candidates.append(normalized)
    return candidates


def get_candidate_validations(store: MemoryStore, candidate_id: str) -> list[dict[str, Any]]:
    frame = store.run_query_df(
        """
        SELECT *
        FROM candidate_validations
        WHERE candidate_id=?
        ORDER BY created_at ASC, validation_run_id ASC
        """,
        [candidate_id],
    )
    return [normalize_record(row) for row in frame.to_dict(orient="records")]


def get_validation_updates(store: MemoryStore, validation_run_id: str) -> list[dict[str, Any]]:
    frame = store.run_query_df(
        """
        SELECT v.*, c.source_run_id, c.strategy, c.archetype, c.status
        FROM candidate_validations v
        JOIN learning_candidates c ON c.candidate_id = v.candidate_id
        WHERE v.validation_run_id=?
        ORDER BY v.created_at ASC, v.candidate_id ASC
        """,
        [validation_run_id],
    )
    return [normalize_record(row) for row in frame.to_dict(orient="records")]


def candidate_validation_aggregate(store: MemoryStore, candidate_id: str) -> dict[str, Any]:
    validations = get_candidate_validations(store, candidate_id)
    return {
        "validation_count": len(validations),
        "candidate_trade_count": sum(int(item.get("candidate_trade_count", 0) or 0) for item in validations),
        "baseline_net_pnl": sum(float(item.get("baseline_net_pnl", 0.0) or 0.0) for item in validations),
        "candidate_net_pnl": sum(float(item.get("candidate_net_pnl", 0.0) or 0.0) for item in validations),
        "improved_count": sum(1 for item in validations if item.get("verdict") == "improved"),
    }


def build_next_actions(
    summary: dict[str, Any],
    candidate_queue: list[dict[str, Any]],
    validation_status: list[dict[str, Any]],
    review: dict[str, Any] | None,
) -> list[str]:
    actions: list[str] = []
    if int(summary.get("trade_count", 0) or 0) <= 0:
        actions.append("Inspect spread gates, websocket health and expected-edge blocks before tuning parameters.")
    if review is not None and not review.get("raw_input_path"):
        actions.append("Raw parquet missing for this run; OOS candidate validation was skipped.")
    promotable = [row for row in candidate_queue if row.get("status") == "promotable"]
    pending = [row for row in candidate_queue if row.get("status") == "pending"]
    if promotable:
        actions.append("Review promotable candidates manually before any rollout to the active config.")
    elif pending:
        actions.append("Collect more future runs to reach the replay OOS promotion gates.")
    if validation_status and any(row.get("verdict") == "worse" for row in validation_status):
        actions.append("Review candidates that underperformed on this run and keep them pending or reject after enough samples.")
    if not actions:
        actions.append("No immediate action; keep accumulating paper runs and refresh the learning review.")
    return actions


def render_learning_markdown(*, run_id: str, overview: dict[str, Any]) -> str:
    summary = _as_dict(overview.get("run_summary"))
    lines = [
        "# ofbot Learning Overview",
        "",
        f"Run ID: `{run_id}`",
        "",
        "## Run Summary",
        f"- Verdict: `{derive_review_verdict(summary)}`",
        f"- Trades: `{int(summary.get('trade_count', 0) or 0)}`",
        f"- Net PnL: `{float(summary.get('net_pnl', 0.0) or 0.0):.6f}`",
        f"- Gross PnL: `{float(summary.get('gross_pnl', 0.0) or 0.0):.6f}`",
        f"- Fees: `{float(summary.get('fees', 0.0) or 0.0):.6f}`",
        f"- Dominant strategy: `{summary.get('dominant_strategy') or 'n/a'}`",
        f"- Dominant regime: `{summary.get('dominant_regime') or 'n/a'}`",
        "",
        "## Lessons Learned",
    ]
    lessons = list(overview.get("lessons") or [])
    if not lessons:
        lines.append("- none")
    else:
        for lesson in lessons:
            lines.append(
                f"- `{lesson.get('lesson_type')}` on `{lesson.get('strategy') or 'n/a'}`: {json.dumps(lesson.get('evidence', {}), sort_keys=True)}"
            )

    lines.extend(["", "## Manual Notes"])
    notes = list(overview.get("manual_notes") or [])
    if not notes:
        lines.append("- none")
    else:
        for note in notes:
            scope = note.get("note_scope") or "run"
            trade_context_id = note.get("trade_context_id")
            trade_label = f" trade_id={trade_context_id}" if trade_context_id is not None else ""
            lines.append(
                f"- `{scope}` `{note.get('tag') or 'note'}`{trade_label}: {note.get('note_text') or ''}"
            )

    lines.extend(["", "## Candidate Queue"])
    candidate_queue = list(overview.get("candidate_queue") or [])
    if not candidate_queue:
        lines.append("- none")
    else:
        for candidate in candidate_queue:
            aggregate = _as_dict(candidate.get("validation_aggregate"))
            lines.append(
                f"- `{candidate.get('candidate_id')}` `{candidate.get('status')}` `{candidate.get('strategy')}`/{candidate.get('archetype')} params={json.dumps(candidate.get('params', {}), sort_keys=True)} validations={int(aggregate.get('validation_count', 0) or 0)}"
            )

    lines.extend(["", "## Validation Status"])
    validation_status = list(overview.get("validation_status") or [])
    if not validation_status:
        lines.append("- none")
    else:
        for item in validation_status:
            lines.append(
                f"- `{item.get('candidate_id')}` from `{item.get('source_run_id')}`: `{item.get('verdict')}` baseline=`{float(item.get('baseline_net_pnl', 0.0) or 0.0):.6f}` candidate=`{float(item.get('candidate_net_pnl', 0.0) or 0.0):.6f}`"
            )

    lines.extend(["", "## Next Actions"])
    for action in overview.get("next_actions") or []:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def dominant_strategy_name(*, trades: pd.DataFrame, events: pd.DataFrame) -> str | None:
    if not trades.empty and "strategy" in trades.columns:
        counts = Counter(str(item) for item in trades["strategy"].fillna("unknown"))
        if counts:
            return counts.most_common(1)[0][0]
    if not events.empty and "strategy" in events.columns:
        counts = Counter(str(item) for item in events["strategy"].fillna("unknown"))
        if counts:
            return counts.most_common(1)[0][0]
    return None


def dominant_regime_name(*, trades: pd.DataFrame, events: pd.DataFrame) -> str | None:
    if not trades.empty and "regime" in trades.columns:
        counts = Counter(str(item) for item in trades["regime"].fillna("unknown"))
        if counts:
            return counts.most_common(1)[0][0]
    if not events.empty and "regime" in events.columns:
        counts = Counter(str(item) for item in events["regime"].fillna("unknown"))
        if counts:
            return counts.most_common(1)[0][0]
    return None


def summarize_strategies(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty or "strategy" not in trades.columns:
        return []
    rows: list[dict[str, Any]] = []
    grouped = trades.groupby("strategy", dropna=False)
    for strategy, frame in grouped:
        normalized_strategy = str(strategy)
        wins = frame[pd.to_numeric(frame["realized_pnl"], errors="coerce").fillna(0.0) > 0.0]
        rows.append(
            {
                "strategy": normalized_strategy,
                "trade_count": int(len(frame.index)),
                "net_pnl": _safe_sum(frame, "realized_pnl"),
                "mean_holding_seconds": _safe_mean(frame, "holding_seconds"),
                "win_rate": float(len(wins.index) / len(frame.index)) if len(frame.index) > 0 else 0.0,
            }
        )
    rows.sort(key=lambda item: (-int(item["trade_count"]), str(item["strategy"])))
    return rows


def summarize_event_metrics(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {
            "signal_submit_count": 0,
            "expected_edge_blocks": 0,
            "exit_reasons": {},
            "stop_exit_rate": 0.0,
        }

    params = events["params"].apply(load_json_dict) if "params" in events.columns else pd.Series([{}] * len(events.index))
    action = events["action"].fillna("") if "action" in events.columns else pd.Series([""] * len(events.index))
    order_intent = events["order_intent"].fillna("") if "order_intent" in events.columns else pd.Series([""] * len(events.index))
    reason = events["reason"].fillna("") if "reason" in events.columns else pd.Series([""] * len(events.index))
    expected_net_edge = pd.to_numeric(params.apply(lambda item: item.get("expected_net_edge_bps")), errors="coerce")
    entry_mask = action.eq("signal") & order_intent.eq("submit") & expected_net_edge.notna()
    exit_mask = action.eq("signal") & order_intent.eq("submit") & ~entry_mask
    exit_reasons = {str(key): int(value) for key, value in reason[exit_mask].value_counts().to_dict().items()}
    stop_count = sum(
        value
        for key, value in exit_reasons.items()
        if "stop" in key or "trailing" in key
    )
    exit_count = sum(exit_reasons.values())
    return {
        "signal_submit_count": int((action.eq("signal") & order_intent.eq("submit")).sum()),
        "expected_edge_blocks": int((action.eq("signal") & order_intent.eq("risk_expected_net_edge")).sum()),
        "exit_reasons": exit_reasons,
        "stop_exit_rate": float(stop_count / exit_count) if exit_count > 0 else 0.0,
    }


def load_json_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, pd.Timestamp):
            normalized[key] = value.isoformat()
        elif isinstance(value, str) and key in {"summary", "notes", "evidence", "recommended_action", "params", "rationale"}:
            normalized[key] = load_json_dict(value)
        else:
            normalized[key] = value
    return normalized


def _safe_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(series.mean()) if not series.empty else 0.0


def _fee_sum(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    if "entry_fees" in trades.columns or "exit_fees" in trades.columns:
        return _safe_sum(trades, "entry_fees") + _safe_sum(trades, "exit_fees")
    return _safe_sum(trades, "fees")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
