from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable

import polars as pl

from ofbot.config import AppConfig, load_config
from ofbot.research.backtest import ReplayBacktestResult, build_single_strategy_config, run_replay_backtest
from ofbot.research.config import ResearchConfig, load_research_config
from ofbot.research.dataset import ReplayDatasetSplit, load_replay_frame, split_replay_dataset
from ofbot.research.evaluation import (
    ScoreCard,
    build_score_card,
    robustness_gate_failures,
    run_robustness_suite,
)
from ofbot.research.registry import ResearchRegistry


BacktestRunner = Callable[[AppConfig, str, dict[str, Any], pl.DataFrame, str, dict[str, Any] | None], ReplayBacktestResult]


@dataclass(slots=True)
class RankedVariant:
    variant_id: str
    params: dict[str, Any]
    train_result: ReplayBacktestResult
    train_score: float
    validation_result: ReplayBacktestResult | None = None
    validation_scorecard: ScoreCard | None = None
    robustness: dict[str, Any] | None = None


def baseline_backtest_report(
    *,
    base_config_path: Path,
    research_config_path: Path,
    raw_input_path: Path | None = None,
    backtest_runner: BacktestRunner | None = None,
) -> Path:
    base_config, research_config, dataset, registry = _bootstrap(
        base_config_path=base_config_path,
        research_config_path=research_config_path,
        raw_input_path=raw_input_path,
    )
    run_id = _research_run_id(prefix="baseline")
    artifacts_dir = _artifact_dir(research_config, run_id)
    baseline_params = _baseline_params(base_config=base_config, research_config=research_config)

    results = {
        "train": _run_variant(
            backtest_runner,
            base_config,
            research_config.strategy_name,
            baseline_params,
            dataset.train.frame,
            "baseline_train",
            None,
        ),
        "validation": _run_variant(
            backtest_runner,
            base_config,
            research_config.strategy_name,
            baseline_params,
            dataset.validation.frame,
            "baseline_validation",
            None,
        ),
        "test": _run_variant(
            backtest_runner,
            base_config,
            research_config.strategy_name,
            baseline_params,
            dataset.test.frame,
            "baseline_test",
            None,
        ),
    }
    summary = {
        "research_run_id": run_id,
        "mode": "baseline_backtest",
        "strategy_name": research_config.strategy_name,
        "baseline_params": baseline_params,
        "dataset": _dataset_summary(research_config, dataset),
        "results": {name: result.metrics for name, result in results.items()},
    }
    report_path = _write_summary(artifacts_dir=artifacts_dir, summary=summary)
    registry.record_experiment(
        {
            "research_run_id": run_id,
            "stage": "baseline_backtest",
            "strategy_name": research_config.strategy_name,
            "params": baseline_params,
            "dataset": summary["dataset"],
            "metrics": summary["results"],
            "decision": "baseline_only",
            "artifact_path": str(report_path),
        }
    )
    return report_path


def run_research_cycle(
    *,
    base_config_path: Path,
    research_config_path: Path,
    raw_input_path: Path | None = None,
    backtest_runner: BacktestRunner | None = None,
) -> dict[str, Any]:
    base_config, research_config, dataset, registry = _bootstrap(
        base_config_path=base_config_path,
        research_config_path=research_config_path,
        raw_input_path=raw_input_path,
    )
    run_id = _research_run_id(prefix="research")
    artifacts_dir = _artifact_dir(research_config, run_id)
    baseline_params = _baseline_params(base_config=base_config, research_config=research_config)
    baseline_validation = _run_variant(
        backtest_runner,
        base_config,
        research_config.strategy_name,
        baseline_params,
        dataset.validation.frame,
        "baseline_validation",
        None,
    )
    baseline_test = _run_variant(
        backtest_runner,
        base_config,
        research_config.strategy_name,
        baseline_params,
        dataset.test.frame,
        "baseline_test",
        None,
    )
    baseline_subperiod_ratio = _subperiod_positive_ratio(
        backtest_runner=backtest_runner,
        base_config=base_config,
        strategy_name=research_config.strategy_name,
        params=baseline_params,
        frame=dataset.validation.frame,
        research_config=research_config,
        split_label_prefix="baseline_validation",
    )
    baseline_validation_scorecard = build_score_card(
        metrics=baseline_validation.metrics,
        config=research_config.scoring,
        subperiod_positive_ratio=baseline_subperiod_ratio,
    )
    baseline_test_scorecard = build_score_card(
        metrics=baseline_test.metrics,
        config=research_config.scoring,
        subperiod_positive_ratio=baseline_subperiod_ratio,
    )

    ranked_variants = _evaluate_train_stage(
        base_config=base_config,
        research_config=research_config,
        dataset=dataset,
        baseline_params=baseline_params,
        backtest_runner=backtest_runner,
        registry=registry,
        run_id=run_id,
    )
    shortlisted = ranked_variants[: research_config.generation.top_k_train]

    validated: list[RankedVariant] = []
    for variant in shortlisted:
        validation_result = _run_variant(
            backtest_runner,
            base_config,
            research_config.strategy_name,
            variant.params,
            dataset.validation.frame,
            f"{variant.variant_id}_validation",
            None,
        )
        robustness = run_robustness_suite(
            params=variant.params,
            validation_frame=dataset.validation.frame,
            train_validation_frame=dataset.train_validation_frame,
            dataset_config=research_config.dataset,
            robustness_config=research_config.robustness,
            param_spaces=_param_space_values(research_config),
            run_backtest=lambda params, frame, split_label, overrides: _run_variant(
                backtest_runner,
                base_config,
                research_config.strategy_name,
                params,
                frame,
                split_label,
                _resolve_execution_overrides(base_config, overrides),
            ),
            split_label_prefix=variant.variant_id,
        )
        scorecard = build_score_card(
            metrics=validation_result.metrics,
            config=research_config.scoring,
            subperiod_positive_ratio=robustness.subperiod_positive_ratio,
        )
        failures = list(scorecard.reasons) + robustness_gate_failures(
            report=robustness,
            config=research_config.robustness,
        )
        accepted = (
            scorecard.accepted
            and not failures
            and scorecard.score
            >= baseline_validation_scorecard.score + research_config.selection.min_validation_score_delta
            and abs(float(validation_result.metrics.get("max_drawdown", 0.0) or 0.0))
            <= abs(float(baseline_validation.metrics.get("max_drawdown", 0.0) or 0.0))
            + research_config.selection.max_drawdown_increase_abs
        )
        variant.validation_result = validation_result
        variant.validation_scorecard = ScoreCard(
            score=scorecard.score,
            components=scorecard.components,
            gates=scorecard.gates,
            accepted=accepted,
            reasons=failures,
        )
        variant.robustness = asdict(robustness)
        registry.record_experiment(
            {
                "research_run_id": run_id,
                "variant_id": variant.variant_id,
                "stage": "validation",
                "strategy_name": research_config.strategy_name,
                "params": variant.params,
                "metrics": validation_result.metrics,
                "score": variant.validation_scorecard.score,
                "decision": "accepted" if accepted else "rejected",
                "reason": failures,
                "artifact_path": str(artifacts_dir / "summary.json"),
            }
        )
        if accepted:
            validated.append(variant)

    validated.sort(
        key=lambda item: (
            -(item.validation_scorecard.score if item.validation_scorecard is not None else float("-inf")),
            -float(item.validation_result.metrics.get("net_pnl", 0.0) if item.validation_result is not None else 0.0),
        )
    )

    status = "no_candidate"
    selected_summary: dict[str, Any] | None = None
    if validated:
        candidate = validated[0]
        candidate_test = _run_variant(
            backtest_runner,
            base_config,
            research_config.strategy_name,
            candidate.params,
            dataset.test.frame,
            f"{candidate.variant_id}_test",
            None,
        )
        candidate_test_scorecard = build_score_card(
            metrics=candidate_test.metrics,
            config=research_config.scoring,
            subperiod_positive_ratio=_subperiod_positive_ratio(
                backtest_runner=backtest_runner,
                base_config=base_config,
                strategy_name=research_config.strategy_name,
                params=candidate.params,
                frame=dataset.test.frame,
                research_config=research_config,
                split_label_prefix=f"{candidate.variant_id}_test",
            ),
        )
        score_delta = candidate_test_scorecard.score - baseline_test_scorecard.score
        passes_test = (
            (not research_config.selection.require_test_improvement or score_delta >= research_config.selection.min_test_score_delta)
            and candidate_test_scorecard.accepted
        )
        status = "validated_for_paper" if passes_test else "rejected_on_test"
        selected_summary = {
            "candidate_id": _candidate_id(run_id=run_id, strategy_name=research_config.strategy_name, params=candidate.params),
            "variant_id": candidate.variant_id,
            "strategy_name": research_config.strategy_name,
            "params": candidate.params,
            "status": status,
            "train_score": candidate.train_score,
            "validation_metrics": candidate.validation_result.metrics if candidate.validation_result is not None else {},
            "validation_score": candidate.validation_scorecard.score if candidate.validation_scorecard is not None else 0.0,
            "validation_failures": candidate.validation_scorecard.reasons if candidate.validation_scorecard is not None else [],
            "robustness": candidate.robustness,
            "test_metrics": candidate_test.metrics,
            "test_score": candidate_test_scorecard.score,
            "test_score_delta_vs_baseline": score_delta,
            "paper_window_minutes": research_config.selection.paper_window_minutes,
        }
        registry.record_candidate(
            {
                **selected_summary,
                "research_run_id": run_id,
                "baseline_params": baseline_params,
                "raw_input_path": str(research_config.dataset.raw_input_path),
                "artifact_path": str(artifacts_dir / "summary.json"),
            }
        )

    summary = {
        "research_run_id": run_id,
        "status": status,
        "strategy_name": research_config.strategy_name,
        "baseline_params": baseline_params,
        "dataset": _dataset_summary(research_config, dataset),
        "baseline": {
            "validation_metrics": baseline_validation.metrics,
            "validation_score": baseline_validation_scorecard.score,
            "test_metrics": baseline_test.metrics,
            "test_score": baseline_test_scorecard.score,
        },
        "variants_evaluated": len(ranked_variants),
        "shortlisted_train": [
            {
                "variant_id": item.variant_id,
                "params": item.params,
                "train_score": item.train_score,
                "train_metrics": item.train_result.metrics,
            }
            for item in shortlisted
        ],
        "selected_candidate": selected_summary,
    }
    report_path = _write_summary(artifacts_dir=artifacts_dir, summary=summary)
    registry.record_experiment(
        {
            "research_run_id": run_id,
            "stage": "research_summary",
            "strategy_name": research_config.strategy_name,
            "params": baseline_params,
            "metrics": summary["baseline"],
            "decision": status,
            "artifact_path": str(report_path),
        }
    )
    return {"summary": summary, "report_path": report_path}


def validate_candidate(
    *,
    base_config_path: Path,
    research_config_path: Path,
    candidate_id: str,
    raw_input_path: Path | None = None,
    backtest_runner: BacktestRunner | None = None,
) -> dict[str, Any]:
    base_config, research_config, dataset, registry = _bootstrap(
        base_config_path=base_config_path,
        research_config_path=research_config_path,
        raw_input_path=raw_input_path,
    )
    candidate = registry.get_candidate(candidate_id)
    if candidate is None:
        raise ValueError(f"candidate not found: {candidate_id}")
    baseline_params = dict(candidate.get("baseline_params") or _baseline_params(base_config=base_config, research_config=research_config))
    candidate_params = dict(candidate.get("params") or {})
    run_id = _research_run_id(prefix="validate")
    artifacts_dir = _artifact_dir(research_config, run_id)

    baseline_validation = _run_variant(
        backtest_runner,
        base_config,
        research_config.strategy_name,
        baseline_params,
        dataset.validation.frame,
        "baseline_validation",
        None,
    )
    candidate_validation = _run_variant(
        backtest_runner,
        base_config,
        research_config.strategy_name,
        candidate_params,
        dataset.validation.frame,
        "candidate_validation",
        None,
    )
    candidate_test = _run_variant(
        backtest_runner,
        base_config,
        research_config.strategy_name,
        candidate_params,
        dataset.test.frame,
        "candidate_test",
        None,
    )
    validation_robustness = run_robustness_suite(
        params=candidate_params,
        validation_frame=dataset.validation.frame,
        train_validation_frame=dataset.train_validation_frame,
        dataset_config=research_config.dataset,
        robustness_config=research_config.robustness,
        param_spaces=_param_space_values(research_config),
        run_backtest=lambda params, frame, split_label, overrides: _run_variant(
            backtest_runner,
            base_config,
            research_config.strategy_name,
            params,
            frame,
            split_label,
            _resolve_execution_overrides(base_config, overrides),
        ),
        split_label_prefix="candidate_validation",
    )
    validation_scorecard = build_score_card(
        metrics=candidate_validation.metrics,
        config=research_config.scoring,
        subperiod_positive_ratio=validation_robustness.subperiod_positive_ratio,
    )
    failures = validation_scorecard.reasons + robustness_gate_failures(
        report=validation_robustness,
        config=research_config.robustness,
    )
    status = "validated_for_paper" if not failures else "rejected"
    updated = {
        **candidate,
        "status": status,
        "validation_metrics": candidate_validation.metrics,
        "test_metrics": candidate_test.metrics,
        "validation_score": validation_scorecard.score,
        "validation_failures": failures,
        "artifact_path": str(artifacts_dir / "summary.json"),
    }
    registry.record_candidate(updated)
    summary = {
        "research_run_id": run_id,
        "mode": "candidate_validation",
        "candidate_id": candidate_id,
        "status": status,
        "baseline_validation": baseline_validation.metrics,
        "candidate_validation": candidate_validation.metrics,
        "candidate_test": candidate_test.metrics,
        "validation_failures": failures,
    }
    report_path = _write_summary(artifacts_dir=artifacts_dir, summary=summary)
    return {"summary": summary, "report_path": report_path}


def _evaluate_train_stage(
    *,
    base_config: AppConfig,
    research_config: ResearchConfig,
    dataset: ReplayDatasetSplit,
    baseline_params: dict[str, Any],
    backtest_runner: BacktestRunner | None,
    registry: ResearchRegistry,
    run_id: str,
) -> list[RankedVariant]:
    ranked: list[RankedVariant] = []
    for params in generate_variants(
        baseline_params=baseline_params,
        param_spaces=_param_space_values(research_config),
        method=research_config.generation.method,
        max_variants=research_config.generation.max_variants,
        seed=research_config.generation.seed,
    ):
        variant_id = _variant_signature(research_config.strategy_name, params)
        train_result = _run_variant(
            backtest_runner,
            base_config,
            research_config.strategy_name,
            params,
            dataset.train.frame,
            f"{variant_id}_train",
            None,
        )
        scorecard = build_score_card(
            metrics=train_result.metrics,
            config=research_config.scoring,
            subperiod_positive_ratio=0.5,
        )
        ranked_variant = RankedVariant(
            variant_id=variant_id,
            params=params,
            train_result=train_result,
            train_score=scorecard.score,
        )
        ranked.append(ranked_variant)
        registry.record_experiment(
            {
                "research_run_id": run_id,
                "variant_id": variant_id,
                "stage": "train",
                "strategy_name": research_config.strategy_name,
                "params": params,
                "metrics": train_result.metrics,
                "score": scorecard.score,
                "decision": "shortlist" if scorecard.accepted else "screened_out",
            }
        )
    ranked.sort(
        key=lambda item: (
            -item.train_score,
            -float(item.train_result.metrics.get("net_pnl", 0.0) or 0.0),
        )
    )
    return ranked


def generate_variants(
    *,
    baseline_params: dict[str, Any],
    param_spaces: dict[str, list[Any]],
    method: str,
    max_variants: int,
    seed: int,
) -> list[dict[str, Any]]:
    items = sorted(param_spaces.items())
    if not items:
        return [dict(baseline_params)]

    variants: list[dict[str, Any]] = []
    if method == "mutate":
        for key, values in items:
            base_value = baseline_params.get(key)
            if base_value not in values:
                continue
            index = values.index(base_value)
            for offset in (-1, 1):
                neighbor_index = index + offset
                if neighbor_index < 0 or neighbor_index >= len(values):
                    continue
                updated = dict(baseline_params)
                updated[key] = values[neighbor_index]
                variants.append(updated)
    else:
        combinations = [dict(baseline_params)]
        for key, values in items:
            next_combinations: list[dict[str, Any]] = []
            for combination in combinations:
                for value in values:
                    updated = dict(combination)
                    updated[key] = value
                    next_combinations.append(updated)
            combinations = next_combinations
        variants = combinations
        if method == "random":
            rng = random.Random(seed)
            rng.shuffle(variants)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    baseline_signature = _stable_payload(baseline_params)
    for params in variants:
        signature = _stable_payload(params)
        if signature == baseline_signature or signature in seen:
            continue
        seen.add(signature)
        deduped.append(params)
        if len(deduped) >= max_variants:
            break
    return deduped


def _bootstrap(
    *,
    base_config_path: Path,
    research_config_path: Path,
    raw_input_path: Path | None,
) -> tuple[AppConfig, ResearchConfig, ReplayDatasetSplit, ResearchRegistry]:
    base_config = load_config(base_config_path)
    research_config = load_research_config(research_config_path)
    if raw_input_path is not None:
        research_config.dataset.raw_input_path = raw_input_path
    frame = load_replay_frame(research_config.dataset.raw_input_path)
    dataset = split_replay_dataset(frame, research_config.dataset)
    registry = ResearchRegistry(research_config.paths.registry_dir)
    return base_config, research_config, dataset, registry


def _baseline_params(*, base_config: AppConfig, research_config: ResearchConfig) -> dict[str, Any]:
    params = dict(getattr(base_config.strategy_library, research_config.strategy_name).params)
    params.update(research_config.baseline_params)
    return params


def _run_variant(
    backtest_runner: BacktestRunner | None,
    base_config: AppConfig,
    strategy_name: str,
    params: dict[str, Any],
    frame: pl.DataFrame,
    split_name: str,
    execution_overrides: dict[str, Any] | None,
) -> ReplayBacktestResult:
    runner = backtest_runner or _default_backtest_runner
    return runner(base_config, strategy_name, params, frame, split_name, execution_overrides)


def _default_backtest_runner(
    base_config: AppConfig,
    strategy_name: str,
    params: dict[str, Any],
    frame: pl.DataFrame,
    split_name: str,
    execution_overrides: dict[str, Any] | None,
) -> ReplayBacktestResult:
    return run_replay_backtest(
        base_config=base_config,
        strategy_name=strategy_name,
        params=params,
        frame=frame,
        split_name=split_name,
        execution_overrides=execution_overrides,
    )


def _resolve_execution_overrides(
    base_config: AppConfig,
    overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not overrides:
        return None
    resolved: dict[str, Any] = {}
    for key, value in overrides.items():
        if not isinstance(value, tuple) or len(value) != 2:
            resolved[key] = value
            continue
        mode, operand = value
        current = getattr(base_config.execution, key)
        if mode == "mul":
            resolved[key] = float(current) * float(operand)
        elif mode == "add":
            resolved[key] = float(current) + float(operand)
        else:
            resolved[key] = operand
    return resolved


def _subperiod_positive_ratio(
    *,
    backtest_runner: BacktestRunner | None,
    base_config: AppConfig,
    strategy_name: str,
    params: dict[str, Any],
    frame: pl.DataFrame,
    research_config: ResearchConfig,
    split_label_prefix: str,
) -> float:
    from ofbot.research.dataset import split_subperiods

    periods = split_subperiods(
        frame,
        count=research_config.dataset.stability_subperiods,
        timestamp_column=research_config.dataset.timestamp_column,
    )
    if not periods:
        return 0.0
    positives = 0
    for period in periods:
        result = _run_variant(
            backtest_runner,
            base_config,
            strategy_name,
            params,
            period.frame,
            f"{split_label_prefix}_{period.name}",
            None,
        )
        if float(result.metrics.get("net_pnl", 0.0) or 0.0) > 0.0:
            positives += 1
    return positives / len(periods)


def _param_space_values(config: ResearchConfig) -> dict[str, list[Any]]:
    return {key: list(spec.values) for key, spec in config.param_spaces.items()}


def _artifact_dir(config: ResearchConfig, run_id: str) -> Path:
    target = config.paths.artifacts_dir / run_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_summary(*, artifacts_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = artifacts_dir / "summary.json"
    report_path = artifacts_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    report_path.write_text(_render_report(summary))
    return report_path


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Controlled Research Report",
        "",
        f"- research_run_id: `{summary.get('research_run_id', 'n/a')}`",
        f"- status: `{summary.get('status', summary.get('mode', 'n/a'))}`",
        f"- strategy_name: `{summary.get('strategy_name', 'n/a')}`",
    ]
    if "baseline_params" in summary:
        lines.append(f"- baseline_params: `{json.dumps(summary['baseline_params'], sort_keys=True)}`")
    if "selected_candidate" in summary and summary["selected_candidate"] is not None:
        candidate = summary["selected_candidate"]
        lines.extend(
            [
                "",
                "## Selected Candidate",
                f"- candidate_id: `{candidate.get('candidate_id', 'n/a')}`",
                f"- status: `{candidate.get('status', 'n/a')}`",
                f"- params: `{json.dumps(candidate.get('params', {}), sort_keys=True)}`",
                f"- validation_score: `{candidate.get('validation_score', 0.0):.4f}`",
                f"- test_score: `{candidate.get('test_score', 0.0):.4f}`",
            ]
        )
    if "results" in summary:
        lines.extend(["", "## Baseline Results"])
        for split_name, metrics in summary["results"].items():
            lines.append(f"- {split_name}: `{json.dumps(metrics, sort_keys=True)}`")
    elif "baseline" in summary:
        lines.extend(
            [
                "",
                "## Baseline",
                f"- validation: `{json.dumps(summary['baseline']['validation_metrics'], sort_keys=True)}`",
                f"- test: `{json.dumps(summary['baseline']['test_metrics'], sort_keys=True)}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _candidate_id(*, run_id: str, strategy_name: str, params: dict[str, Any]) -> str:
    digest = hashlib.sha1(f"{run_id}|{strategy_name}|{_stable_payload(params)}".encode("utf-8")).hexdigest()[:10]
    return f"{strategy_name}_{digest}"


def _variant_signature(strategy_name: str, params: dict[str, Any]) -> str:
    digest = hashlib.sha1(f"{strategy_name}|{_stable_payload(params)}".encode("utf-8")).hexdigest()[:8]
    return f"variant_{digest}"


def _stable_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _dataset_summary(config: ResearchConfig, dataset: ReplayDatasetSplit) -> dict[str, Any]:
    return {
        "raw_input_path": str(config.dataset.raw_input_path),
        "train": {
            "rows": dataset.train.frame.height,
            "start": dataset.train.start_time.isoformat(),
            "end": dataset.train.end_time.isoformat(),
        },
        "validation": {
            "rows": dataset.validation.frame.height,
            "start": dataset.validation.start_time.isoformat(),
            "end": dataset.validation.end_time.isoformat(),
        },
        "test": {
            "rows": dataset.test.frame.height,
            "start": dataset.test.start_time.isoformat(),
            "end": dataset.test.end_time.isoformat(),
        },
    }


def _research_run_id(*, prefix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
