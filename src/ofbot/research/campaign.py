from __future__ import annotations

import csv
from collections import Counter
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

import polars as pl

from ofbot.config import load_config
from ofbot.research.backtest import run_replay_backtest
from ofbot.research.config import load_research_config
from ofbot.research.dataset import load_replay_frame
from ofbot.research.evaluation import build_score_card
from ofbot.research.promotion import promote_candidate_to_paper, rollback_active_paper_candidate
from ofbot.research.registry import ResearchRegistry
from ofbot.research.runner import generate_variants


LiveRunner = Callable[[str, int, int, list[str] | None], Awaitable[Path | None]]
ResearchCycleRunner = Callable[..., dict[str, Any]]


async def run_paper_research_campaign(
    *,
    base_config_path: Path,
    research_config_path: Path,
    sessions: int,
    session_seconds: int,
    live_runner: LiveRunner,
    symbols: list[str] | None = None,
    reset_active_paper: bool = True,
    research_cycle_runner: ResearchCycleRunner | None = None,
) -> dict[str, Any]:
    if sessions <= 0:
        raise ValueError("sessions must be strictly positive")
    if session_seconds <= 0:
        raise ValueError("session_seconds must be strictly positive")

    research_config = load_research_config(research_config_path)
    registry = ResearchRegistry(research_config.paths.registry_dir)
    campaign_id = _campaign_id()
    campaign_dir = research_config.paths.artifacts_dir.parent / "campaigns" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = campaign_dir / "sessions.jsonl"
    summary_path = campaign_dir / "summary.json"
    active_config_path = research_config.paths.active_paper_config_path
    active_candidate_id: str | None = None

    if reset_active_paper or not active_config_path.exists():
        rollback_active_paper_candidate(
            base_config_path=base_config_path,
            registry=registry,
            active_config_path=active_config_path,
            candidate_id=None,
            reason="campaign_start_baseline",
        )

    summary: dict[str, Any] = {
        "campaign_id": campaign_id,
        "status": "running",
        "base_config_path": str(base_config_path),
        "research_config_path": str(research_config_path),
        "campaign_dir": str(campaign_dir),
        "active_paper_config_path": str(active_config_path),
        "sessions_requested": int(sessions),
        "sessions_completed": 0,
        "session_seconds": int(session_seconds),
        "symbols": list(symbols or []),
        "started_at": datetime.now(UTC).isoformat(),
        "active_candidate_id": None,
    }
    _write_json(summary_path, summary)

    for session_index in range(1, sessions + 1):
        session_started_at = datetime.now(UTC)
        runtime_config_path = active_config_path if active_config_path.exists() else base_config_path
        report_path = await live_runner(
            str(runtime_config_path),
            0,
            int(session_seconds),
            symbols,
        )
        record: dict[str, Any] = {
            "campaign_id": campaign_id,
            "session_index": session_index,
            "started_at": session_started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "runtime_config_path": str(runtime_config_path),
            "report_path": str(report_path) if report_path is not None else None,
            "run_id": None,
            "raw_input_path": None,
            "session_metrics": {},
            "research_status": None,
            "research_report_path": None,
            "candidate_id": None,
            "active_candidate_id_before_session": active_candidate_id,
            "active_candidate_id_after_session": active_candidate_id,
            "deployment_action": "kept_current",
            "deployment_path": None,
            "notes": [],
        }

        if report_path is None:
            record["deployment_action"] = "halted_missing_report"
            record["notes"].append("live runner returned no report; campaign halted")
            _append_jsonl(sessions_path, record)
            summary.update(
                {
                    "status": "halted_missing_report",
                    "sessions_completed": session_index - 1,
                    "ended_at": datetime.now(UTC).isoformat(),
                    "active_candidate_id": active_candidate_id,
                }
            )
            _write_json(summary_path, summary)
            return summary

        run_id = _run_id_from_report_path(report_path)
        record["run_id"] = run_id
        record["session_metrics"] = summarize_session_artifacts(report_path)

        runtime_config = load_config(runtime_config_path)
        raw_input_path = runtime_config.paths.raw_dir / f"raw_{run_id}.parquet"
        if raw_input_path.exists():
            record["raw_input_path"] = str(raw_input_path)
        else:
            record["notes"].append(f"raw parquet missing: {raw_input_path}")

        if raw_input_path.exists():
            try:
                if research_cycle_runner is None:
                    research_result = run_campaign_tuning_cycle(
                        base_config_path=runtime_config_path,
                        research_config_path=research_config_path,
                        raw_input_path=raw_input_path,
                        tune_window_seconds=max(30, min(60, session_seconds // 10)),
                        max_variants=min(2, research_config.generation.max_variants),
                    )
                else:
                    research_result = research_cycle_runner(
                        base_config_path=runtime_config_path,
                        research_config_path=research_config_path,
                        raw_input_path=raw_input_path,
                    )
                research_summary = research_result.get("summary") or {}
                record["research_status"] = str(research_summary.get("status") or "unknown")
                record["research_report_path"] = str(research_result.get("report_path"))
                candidate = research_summary.get("selected_candidate") or {}
                candidate_id = candidate.get("candidate_id")
                if isinstance(candidate_id, str) and candidate_id:
                    record["candidate_id"] = candidate_id
                if record["research_status"] == "validated_for_paper" and record["candidate_id"] is not None:
                    deployment_path = promote_candidate_to_paper(
                        base_config_path=runtime_config_path,
                        registry=registry,
                        deployment_dir=research_config.paths.deployment_dir,
                        active_config_path=active_config_path,
                        candidate_id=record["candidate_id"],
                        paper_window_minutes=max(1, int(session_seconds // 60)),
                    )
                    active_candidate_id = record["candidate_id"]
                    record["active_candidate_id_after_session"] = active_candidate_id
                    record["deployment_action"] = "promoted_candidate"
                    record["deployment_path"] = str(deployment_path)
                else:
                    record["notes"].append("no validated paper candidate; kept current config")
            except Exception as exc:  # pragma: no cover - defensive logging path
                record["research_status"] = "research_failed"
                record["notes"].append(f"research_failed: {exc}")

        _append_jsonl(sessions_path, record)
        summary.update(
            {
                "status": "running",
                "sessions_completed": session_index,
                "last_run_id": run_id,
                "last_report_path": str(report_path),
                "active_candidate_id": active_candidate_id,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(summary_path, summary)

    summary.update(
        {
            "status": "completed",
            "sessions_completed": int(sessions),
            "ended_at": datetime.now(UTC).isoformat(),
            "active_candidate_id": active_candidate_id,
        }
    )
    _write_json(summary_path, summary)
    return summary


def summarize_session_artifacts(report_path: Path) -> dict[str, Any]:
    report_dir = report_path.parent
    events_path = report_dir / "events.csv"
    trades_path = report_dir / "trades.csv"

    action_counts: Counter[str] = Counter()
    order_intent_counts: Counter[str] = Counter()
    risk_gate_counts: Counter[str] = Counter()
    if events_path.exists():
        with events_path.open("r", encoding="utf-8", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                action = str(row.get("action") or "").strip()
                if action:
                    action_counts[action] += 1
                order_intent = str(row.get("order_intent") or "").strip()
                if order_intent and action != "fill":
                    order_intent_counts[order_intent] += 1
                if str(row.get("reason") or "").strip() == "risk_gate":
                    if order_intent:
                        risk_gate_counts[order_intent] += 1

    trade_count = 0
    net_pnl = 0.0
    gross_pnl = 0.0
    fees = 0.0
    holding_sum = 0.0
    if trades_path.exists():
        with trades_path.open("r", encoding="utf-8", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                trade_count += 1
                net_pnl += _float_value(row.get("realized_pnl"))
                gross_pnl += _float_value(row.get("gross_pnl"))
                fees += _float_value(row.get("fees"))
                holding_sum += _float_value(row.get("holding_seconds"))

    return {
        "signals": int(action_counts.get("signal", 0)),
        "submits": max(int(action_counts.get("submit", 0)), int(order_intent_counts.get("submit", 0))),
        "fills": max(int(action_counts.get("fill", 0)), int(order_intent_counts.get("fill", 0))),
        "closed_trades": int(trade_count),
        "net_pnl": float(net_pnl),
        "gross_pnl": float(gross_pnl),
        "fees": float(fees),
        "mean_holding_seconds": float(holding_sum / trade_count) if trade_count > 0 else 0.0,
        "top_risk_gates": dict(risk_gate_counts.most_common(5)),
    }


def run_campaign_tuning_cycle(
    *,
    base_config_path: Path,
    research_config_path: Path,
    raw_input_path: Path | None = None,
    tune_window_seconds: int | None = None,
    max_variants: int | None = None,
) -> dict[str, Any]:
    base_config = load_config(base_config_path)
    research_config = load_research_config(research_config_path)
    if raw_input_path is not None:
        research_config.dataset.raw_input_path = raw_input_path

    frame = load_replay_frame(research_config.dataset.raw_input_path)
    frame = _restrict_frame_for_tuning(frame, seconds=tune_window_seconds)
    registry = ResearchRegistry(research_config.paths.registry_dir)
    tune_id = datetime.now(UTC).strftime("campaign_tune_%Y%m%d_%H%M%S")
    artifacts_dir = research_config.paths.artifacts_dir / tune_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_params = dict(getattr(base_config.strategy_library, research_config.strategy_name).params)
    baseline_params.update(research_config.baseline_params)
    baseline_result = run_replay_backtest(
        base_config=base_config,
        strategy_name=research_config.strategy_name,
        params=baseline_params,
        frame=frame,
        split_name="campaign_baseline",
    )
    baseline_scorecard = build_score_card(
        metrics=baseline_result.metrics,
        config=research_config.scoring,
        subperiod_positive_ratio=1.0,
    )
    registry.record_experiment(
        {
            "research_run_id": tune_id,
            "stage": "campaign_baseline",
            "strategy_name": research_config.strategy_name,
            "params": baseline_params,
            "metrics": baseline_result.metrics,
            "score": baseline_scorecard.score,
            "decision": "baseline",
        }
    )

    variants = generate_variants(
        baseline_params=baseline_params,
        param_spaces={key: list(spec.values) for key, spec in research_config.param_spaces.items()},
        method=research_config.generation.method,
        max_variants=max_variants or research_config.generation.max_variants,
        seed=research_config.generation.seed,
    )

    ranked: list[tuple[float, int, float, dict[str, Any], dict[str, Any]]] = []
    for params in variants:
        result = run_replay_backtest(
            base_config=base_config,
            strategy_name=research_config.strategy_name,
            params=params,
            frame=frame,
            split_name="campaign_variant",
        )
        scorecard = build_score_card(
            metrics=result.metrics,
            config=research_config.scoring,
            subperiod_positive_ratio=1.0,
        )
        trade_count = int(result.metrics.get("trade_count", 0) or 0)
        accepted = (
            scorecard.accepted
            and scorecard.score >= baseline_scorecard.score + research_config.selection.min_validation_score_delta
            and trade_count >= research_config.scoring.min_trades
            and abs(float(result.metrics.get("max_drawdown", 0.0) or 0.0))
            <= abs(float(baseline_result.metrics.get("max_drawdown", 0.0) or 0.0))
            + research_config.selection.max_drawdown_increase_abs
        )
        decision = "accepted" if accepted else "rejected"
        registry.record_experiment(
            {
                "research_run_id": tune_id,
                "stage": "campaign_variant",
                "strategy_name": research_config.strategy_name,
                "params": params,
                "metrics": result.metrics,
                "score": scorecard.score,
                "decision": decision,
            }
        )
        if accepted:
            ranked.append(
                (
                    float(scorecard.score),
                    trade_count,
                    float(result.metrics.get("net_pnl", 0.0) or 0.0),
                    params,
                    result.metrics,
                )
            )

    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2]))

    summary: dict[str, Any] = {
        "research_run_id": tune_id,
        "mode": "campaign_tuning",
        "status": "no_candidate",
        "strategy_name": research_config.strategy_name,
        "baseline_params": baseline_params,
        "baseline_metrics": baseline_result.metrics,
        "baseline_score": baseline_scorecard.score,
        "tuning_window_seconds": tune_window_seconds,
        "sample_rows": frame.height,
        "variants_evaluated": len(variants),
        "selected_candidate": None,
    }

    if ranked:
        best_score, _, _, best_params, best_metrics = ranked[0]
        candidate_id = _campaign_candidate_id(
            tune_id=tune_id,
            strategy_name=research_config.strategy_name,
            params=best_params,
        )
        candidate = {
            "candidate_id": candidate_id,
            "research_run_id": tune_id,
            "strategy_name": research_config.strategy_name,
            "params": best_params,
            "baseline_params": baseline_params,
            "status": "validated_for_paper",
            "paper_window_minutes": research_config.selection.paper_window_minutes,
            "validation_metrics": best_metrics,
            "validation_score": best_score,
            "raw_input_path": str(research_config.dataset.raw_input_path),
            "artifact_path": str(artifacts_dir / "report.md"),
            "decision_reason": "campaign_tuning_delta",
        }
        registry.record_candidate(candidate)
        summary["status"] = "validated_for_paper"
        summary["selected_candidate"] = {
            "candidate_id": candidate_id,
            "strategy_name": research_config.strategy_name,
            "params": best_params,
            "validation_metrics": best_metrics,
            "validation_score": best_score,
        }

    report_path = _write_campaign_tuning_report(artifacts_dir=artifacts_dir, summary=summary)
    return {"summary": summary, "report_path": report_path}


def _campaign_id() -> str:
    return datetime.now(UTC).strftime("campaign_%Y%m%d_%H%M%S")


def _run_id_from_report_path(report_path: Path) -> str:
    if report_path.parent.parent.name.startswith("run_"):
        return report_path.parent.parent.name
    if report_path.parent.name.startswith("report_run_"):
        return report_path.parent.name.removeprefix("report_")
    raise ValueError(f"unable to infer run_id from report path: {report_path}")


def _float_value(raw: str | None) -> float:
    if raw in (None, ""):
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _restrict_frame_for_tuning(frame: pl.DataFrame, *, seconds: int | None) -> pl.DataFrame:
    if seconds is None or seconds <= 0 or frame.height == 0:
        return frame
    latest_time = frame.select(pl.col("event_time").max()).item()
    if latest_time is None:
        return frame
    clipped = frame.filter(pl.col("event_time") >= pl.lit(latest_time - timedelta(seconds=seconds)))
    if clipped.height == 0:
        return frame
    return clipped.sort(["event_time", "sequence"])


def _campaign_candidate_id(*, tune_id: str, strategy_name: str, params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(f"{strategy_name}|{payload}".encode("utf-8")).hexdigest()[:12]
    return f"{strategy_name}_{tune_id}_{digest}"


def _write_campaign_tuning_report(*, artifacts_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = artifacts_dir / "summary.json"
    report_path = artifacts_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Campaign Tuning Report",
        "",
        f"- research_run_id: `{summary['research_run_id']}`",
        f"- status: `{summary['status']}`",
        f"- strategy_name: `{summary['strategy_name']}`",
        f"- variants_evaluated: `{summary['variants_evaluated']}`",
        f"- baseline_score: `{summary['baseline_score']:.6f}`",
        f"- baseline_metrics: `{json.dumps(summary['baseline_metrics'], sort_keys=True)}`",
    ]
    candidate = summary.get("selected_candidate")
    if candidate is not None:
        lines.extend(
            [
                "",
                "## Selected Candidate",
                f"- candidate_id: `{candidate['candidate_id']}`",
                f"- params: `{json.dumps(candidate['params'], sort_keys=True)}`",
                f"- validation_score: `{candidate['validation_score']:.6f}`",
                f"- validation_metrics: `{json.dumps(candidate['validation_metrics'], sort_keys=True)}`",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
