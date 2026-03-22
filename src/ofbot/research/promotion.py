from __future__ import annotations

from pathlib import Path
from typing import Any

from ofbot.config import dump_config_yaml, load_config
from ofbot.research.backtest import build_single_strategy_config
from ofbot.research.registry import ResearchRegistry


def promote_candidate_to_paper(
    *,
    base_config_path: Path,
    registry: ResearchRegistry,
    deployment_dir: Path,
    active_config_path: Path,
    candidate_id: str,
    paper_window_minutes: int,
) -> Path:
    candidate = registry.get_candidate(candidate_id)
    if candidate is None:
        raise ValueError(f"candidate not found: {candidate_id}")
    if candidate.get("status") != "validated_for_paper":
        raise ValueError(f"candidate {candidate_id} is not validated_for_paper")

    base_config = load_config(base_config_path)
    deployed = build_single_strategy_config(
        base_config=base_config,
        strategy_name=str(candidate["strategy_name"]),
        params=dict(candidate.get("params") or {}),
        disable_learning=True,
    )
    deployment_dir.mkdir(parents=True, exist_ok=True)
    target_path = deployment_dir / f"{candidate_id}.paper.yaml"
    target_path.write_text(dump_config_yaml(deployed))
    active_config_path.parent.mkdir(parents=True, exist_ok=True)
    active_config_path.write_text(dump_config_yaml(deployed))

    registry.record_candidate(
        {
            **candidate,
            "status": "paper_candidate",
            "paper_config_path": str(target_path),
            "paper_window_minutes": int(paper_window_minutes),
            "decision_reason": "promoted_to_paper",
        }
    )
    return target_path


def rollback_active_paper_candidate(
    *,
    base_config_path: Path,
    registry: ResearchRegistry,
    active_config_path: Path,
    candidate_id: str | None,
    reason: str,
) -> Path:
    baseline_config = load_config(base_config_path)
    baseline_config.learning.enabled = False
    baseline_config.learning.enable_promotion = False
    active_config_path.parent.mkdir(parents=True, exist_ok=True)
    active_config_path.write_text(dump_config_yaml(baseline_config))

    if candidate_id is not None:
        candidate = registry.get_candidate(candidate_id)
        if candidate is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        registry.record_candidate(
            {
                **candidate,
                "status": "rolled_back",
                "decision_reason": reason,
            }
        )
    return active_config_path
