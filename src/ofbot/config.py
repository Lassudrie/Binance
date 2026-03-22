from __future__ import annotations

from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class PathsConfig(BaseModel):
    root: Path
    raw_dir: Path
    reports_dir: Path
    memory_dir: Path


class GatewayConfig(BaseModel):
    public_ws_base: str = "wss://stream.binance.com:9443/stream"
    public_rest_base: str = "https://api.binance.com/api"
    public_ws_reconnect_initial_delay_s: float = 1.0
    public_ws_reconnect_max_delay_s: float = 30.0
    heartbeat_timeout_s: float = 8.0
    request_timeout_s: int = 20
    depth_stream: Literal["100ms", "1000ms"] = "100ms"
    depth_snapshot_limit: int = 1000
    book_ticker_divergence_bps: float = 50.0
    include_kline_1m: bool = False
    max_queue_size: int = 2000


class FeaturesConfig(BaseModel):
    horizons_sec: list[int] = Field(default_factory=lambda: [1, 5, 15, 30, 60, 180])
    feature_window_sec: int = 600
    large_trade_quantile: float = 0.90
    volatility_window_sec: int = 60
    zscore_window: int = 200
    microprice_window_sec: int = 120


class StrategyModeConfig(BaseModel):
    name: Literal["continuation", "exhaustion", "hybrid"]
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

class PolicyLibraryConfig(BaseModel):
    continuation: StrategyModeConfig = Field(default_factory=lambda: StrategyModeConfig(name="continuation"))
    exhaustion: StrategyModeConfig = Field(default_factory=lambda: StrategyModeConfig(name="exhaustion"))
    hybrid: StrategyModeConfig = Field(default_factory=lambda: StrategyModeConfig(name="hybrid"))
    default: Literal["continuation", "exhaustion", "hybrid"] = "continuation"


class ExecutionConfig(BaseModel):
    mode: Literal["paper_local", "paper_testnet"] = "paper_local"
    allow_short: bool = False
    initial_cash: float = 100_000.0
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 1.0
    default_target_notional_usd: float = 500.0
    min_target_notional_usd: float = 250.0
    max_target_notional_usd: float = 1_000.0
    base_spread_bps: float = 2.0
    fallback_half_spread_bps: float = 1.0
    impact_bps_per_unit_participation: float = 10.0
    market_sweep_depth_bps: float = 25.0
    latency_ms: int = 150
    market_participation_cap: float = 0.25
    limit_order_book_depth_bps: float = 3.0
    max_order_lifetime_s: int = 8
    allow_limit_orders: bool = True
    soft_exit_limit_ttl_s: float = 2.0
    soft_exit_limit_adverse_bps: float = 1.0

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in {"paper_local", "paper_testnet"}:
            return "paper_local"
        return value


class RiskKillSwitchConfig(BaseModel):
    stale_data_s: float = 6.0
    broken_ws_s: float = 12.0
    abnormal_spread_bps: float = 250.0
    abnormal_volatility_bps: float = 900.0
    max_position_size: float = 8.0
    max_notional: float = 50_000.0
    max_concurrent_exposure: int = 2
    cooldown_after_loss_s: int = 300
    cooldown_after_trade_s: int = 0
    daily_loss_limit: float = 0.02
    max_holding_time_s: int = 900
    catastrophic_stop_loss_bps: float = 45.0
    min_hold_before_soft_exit_s: int = 20
    stop_loss_bps: float = 70.0
    take_profit_bps: float = 120.0
    trailing_stop_bps: float = 35.0
    spread_gate_enabled: bool = True
    volatility_gate_enabled: bool = True
    max_consecutive_losses: int = 5


class LearningConfig(BaseModel):
    enabled: bool = True
    duckdb_path: Path = Path("data/memory/learning.duckdb")
    min_samples_per_regime: int = 12
    min_samples_total: int = 24
    min_improvement_bps: float = 8.0
    max_drawdown_bps: float = 120.0
    rolling_window: int = 60
    confidence: float = 0.80
    enable_promotion: bool = True


class HousekeepingConfig(BaseModel):
    enabled: bool = True
    keep_recent_run_raw: int = 12
    keep_recent_report_csv: int = 8
    preserve_pending_candidate_sources: bool = True
    remove_tmp_pytest: bool = True
    status_markdown_path: Path = Path("docs/live_learning_status.md")


class AppConfig(BaseModel):
    mode: Literal["paper_local", "paper_testnet"] = "paper_local"
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT"])
    feature_profile: str = "orderflow_hft_v1"
    universe_seed: int = 31
    live_telemetry_interval_s: int = 30
    live_telemetry_top_k: int = 3
    paths: PathsConfig
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskKillSwitchConfig = Field(default_factory=RiskKillSwitchConfig)
    strategy_library: PolicyLibraryConfig = Field(default_factory=PolicyLibraryConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    housekeeping: HousekeepingConfig = Field(default_factory=HousekeepingConfig)
    paper_local_record_raw: bool = True
    seed: int = 42
    report_time: str = "today"
    enable_testnet_signed_endpoints: bool = False

    @property
    def symbol_set(self) -> tuple[str, ...]:
        return tuple(self.symbols)

    @model_validator(mode="after")
    def _sync_modes(self) -> "AppConfig":
        if self.execution.mode != self.mode:
            self.execution.mode = self.mode
        return self

    def resolve(self, repo_root: Path | None = None) -> "AppConfig":
        root = repo_root or Path.cwd()
        self.paths = PathsConfig(
            root=root,
            raw_dir=self._resolve_path(root, self.paths.raw_dir),
            reports_dir=self._resolve_path(root, self.paths.reports_dir),
            memory_dir=self._resolve_path(root, self.paths.memory_dir),
        )
        self.learning.duckdb_path = self._resolve_path(root, self.learning.duckdb_path)
        self.housekeeping.status_markdown_path = self._resolve_path(root, self.housekeeping.status_markdown_path)
        return self

    @staticmethod
    def _resolve_path(base: Path, value: Path) -> Path:
        if value.is_absolute():
            return value
        return base / value


def _safe_yaml_load(raw_text: str) -> dict[str, Any]:
    parsed = yaml.safe_load(raw_text) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Config file must contain a YAML object")
    return parsed


def load_config(path: str | Path, *, repo_root: Path | None = None) -> AppConfig:
    config_path = Path(path)
    root = repo_root or config_path.resolve().parent.parent if config_path.exists() else Path.cwd()
    data = _safe_yaml_load(config_path.read_text())
    data = {"mode": data.get("mode", "paper_local"), **data}
    base = root if root is not None else Path.cwd()

    resolved = AppConfig(
        mode=data.get("mode", "paper_local"),
        symbols=list(data.get("symbols", ["BTCUSDT"])),
        feature_profile=str(data.get("feature_profile", "orderflow_hft_v1")),
        universe_seed=int(data.get("universe_seed", 31)),
        live_telemetry_interval_s=int(data.get("live_telemetry_interval_s", 30)),
        live_telemetry_top_k=int(data.get("live_telemetry_top_k", 3)),
        paths=PathsConfig(
            root=base,
            raw_dir=Path(data.get("raw_dir", data.get("paths", {}).get("raw_dir", "data/raw"))),
            reports_dir=Path(data.get("reports_dir", data.get("paths", {}).get("reports_dir", "data/reports"))),
            memory_dir=Path(data.get("memory_dir", data.get("paths", {}).get("memory_dir", "data/memory"))),
        ),
        gateway=GatewayConfig(**data.get("gateway", {})),
        features=FeaturesConfig(**data.get("features", {})),
        execution=ExecutionConfig(
            mode=data.get("mode", "paper_local"),
            **{k: v for k, v in data.get("execution", {}).items() if k != "mode"},
        ),
        risk=RiskKillSwitchConfig(**data.get("risk", {})),
        strategy_library=PolicyLibraryConfig(**data.get("strategy_library", {})),
        learning=LearningConfig(**data.get("learning", {})),
        housekeeping=HousekeepingConfig(**data.get("housekeeping", {})),
        paper_local_record_raw=bool(data.get("paper_local_record_raw", True)),
        seed=int(data.get("seed", data.get("universe_seed", 42))),
        report_time=str(data.get("report_time", "today")),
        enable_testnet_signed_endpoints=bool(
            data.get("enable_testnet_signed_endpoints", False)
        ),
    )
    return resolved.resolve(base)


def dump_config_yaml(config: AppConfig) -> str:
    payload = config.model_dump(mode="json")
    return yaml.safe_dump(payload, sort_keys=False)
