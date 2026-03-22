from __future__ import annotations

from ofbot.research.campaign import run_paper_research_campaign
from ofbot.research.config import ResearchConfig, load_research_config
from ofbot.research.promotion import promote_candidate_to_paper, rollback_active_paper_candidate
from ofbot.research.runner import (
    baseline_backtest_report,
    run_research_cycle,
    validate_candidate,
)

__all__ = [
    "ResearchConfig",
    "baseline_backtest_report",
    "load_research_config",
    "promote_candidate_to_paper",
    "rollback_active_paper_candidate",
    "run_paper_research_campaign",
    "run_research_cycle",
    "validate_candidate",
]
