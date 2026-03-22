from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ResearchPathsConfig(BaseModel):
    registry_dir: Path = Path("data/research/registry")
    artifacts_dir: Path = Path("data/research/runs")
    deployment_dir: Path = Path("data/research/deployments")
    active_paper_config_path: Path = Path("data/research/deployments/active.paper.yaml")


class DatasetConfig(BaseModel):
    raw_input_path: Path
    timestamp_column: str = "event_time"
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2
    stability_subperiods: int = 4
    walk_forward_windows: int = 3
    walk_forward_train_ratio: float = 0.6
    min_unique_timestamps: int = 40

    @model_validator(mode="after")
    def _validate_ratios(self) -> "DatasetConfig":
        total = float(self.train_ratio) + float(self.validation_ratio) + float(self.test_ratio)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")
        if min(self.train_ratio, self.validation_ratio, self.test_ratio) <= 0.0:
            raise ValueError("all dataset split ratios must be strictly positive")
        return self


class VariantGenerationConfig(BaseModel):
    method: Literal["grid", "random", "mutate"] = "grid"
    max_variants: int = 12
    top_k_train: int = 4
    top_k_validation: int = 2
    seed: int = 42


class SearchSpaceParamConfig(BaseModel):
    values: list[Any]

    @field_validator("values")
    @classmethod
    def _validate_values(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("search space values must not be empty")
        return value


class CompositeScoreWeights(BaseModel):
    net_pnl: float = 0.30
    max_drawdown: float = 0.20
    sharpe_like: float = 0.15
    win_rate: float = 0.10
    profit_factor: float = 0.10
    expectancy: float = 0.05
    stability: float = 0.10


class CompositeScoreConfig(BaseModel):
    weights: CompositeScoreWeights = Field(default_factory=CompositeScoreWeights)
    min_trades: int = 5
    pnl_scale: float = 5.0
    drawdown_scale: float = 5.0
    expectancy_scale: float = 0.25
    max_drawdown_abs: float = 10.0
    min_profit_factor: float = 0.85
    min_win_rate: float = 0.40


class RobustnessConfig(BaseModel):
    min_subperiod_positive_ratio: float = 0.50
    min_walk_forward_positive_ratio: float = 0.50
    min_stressed_positive_ratio: float = 0.50
    min_neighbor_positive_ratio: float = 0.50
    stressed_maker_fee_multipliers: list[float] = Field(default_factory=lambda: [1.0, 1.25])
    stressed_taker_fee_multipliers: list[float] = Field(default_factory=lambda: [1.0, 1.25])
    stressed_impact_multipliers: list[float] = Field(default_factory=lambda: [1.0, 1.25])
    stressed_latency_ms_add: list[int] = Field(default_factory=lambda: [0, 100])


class SelectionConfig(BaseModel):
    min_validation_score_delta: float = 0.05
    min_test_score_delta: float = 0.0
    max_drawdown_increase_abs: float = 3.0
    require_test_improvement: bool = True
    paper_window_minutes: int = 60


class ResearchConfig(BaseModel):
    strategy_name: Literal["continuation", "exhaustion", "hybrid"] = "continuation"
    notes: str | None = None
    paths: ResearchPathsConfig = Field(default_factory=ResearchPathsConfig)
    dataset: DatasetConfig
    generation: VariantGenerationConfig = Field(default_factory=VariantGenerationConfig)
    scoring: CompositeScoreConfig = Field(default_factory=CompositeScoreConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    param_spaces: dict[str, SearchSpaceParamConfig] = Field(default_factory=dict)
    baseline_params: dict[str, Any] = Field(default_factory=dict)

    def resolve(self, repo_root: Path) -> "ResearchConfig":
        self.paths = ResearchPathsConfig(
            registry_dir=_resolve_path(repo_root, self.paths.registry_dir),
            artifacts_dir=_resolve_path(repo_root, self.paths.artifacts_dir),
            deployment_dir=_resolve_path(repo_root, self.paths.deployment_dir),
            active_paper_config_path=_resolve_path(repo_root, self.paths.active_paper_config_path),
        )
        self.dataset.raw_input_path = _resolve_path(repo_root, self.dataset.raw_input_path)
        return self


def _resolve_path(repo_root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return repo_root / value


def _safe_yaml_load(raw_text: str) -> dict[str, Any]:
    parsed = yaml.safe_load(raw_text) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Research config must contain a YAML object")
    return parsed


def load_research_config(path: str | Path, *, repo_root: Path | None = None) -> ResearchConfig:
    config_path = Path(path)
    root = repo_root or config_path.resolve().parent.parent if config_path.exists() else Path.cwd()
    raw = _safe_yaml_load(config_path.read_text())
    config = ResearchConfig(**raw)
    return config.resolve(root)
